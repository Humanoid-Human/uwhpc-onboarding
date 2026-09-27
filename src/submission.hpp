#pragma once

#include <algorithm>
#include <complex>
#include <vector>

// for convenience
using std::size_t, std::vector;
using complex = std::complex<double>;

const double TAU = 6.283185307179586;

void rfft2(const vector<double>& src, vector<complex>& dst, size_t n, size_t rows, complex * scratch);

class Grid {
	size_t rows;
	size_t cols;
	size_t pad_n;

	vector<double> data;
	vector<complex> kern_fft;
	mutable vector<complex> scratch1;
	mutable vector<complex> scratch2;

	public:
	Grid(size_t r, size_t c): rows{r}, cols{c} {
		// padding for fft (next power of 2)
		pad_n = (r < c ? c : r) + 1; // would be -1 but kernel is 3-wide so +2 on top
		for (size_t s = pad_n >> 1; s != 0; s >>= 1) { pad_n |= s; }
		pad_n++;

		// this doesn't actually have to be square, since the dimensions just have to be powers of 2
		// but non-square transpose algorithms are more complicated
		size_t data_size = pad_n * pad_n;
		data = vector<double>(data_size, 0);

		// scratch arrays so apply_stencil doesnt have to allocate
		scratch1 = vector<complex>(data_size);
		scratch2 = vector<complex>(data_size);

		// precompute kernel fft bc it's constant
		vector<double> kernel(data_size, 0);
		kernel[0] = 0.5;
		kernel[1] = 0.125;
		kernel[pad_n - 1] = 0.125;
		kernel[pad_n] = 0.125;
		kernel[pad_n * (pad_n - 1)] = 0.125;

		kern_fft = vector<complex>(data_size);
		rfft2(kernel, kern_fft, pad_n, pad_n, scratch1.data());
	}

	double& operator()(size_t i, size_t j)       { return data[i * pad_n + j]; }
	double  operator()(size_t i, size_t j) const { return data[i * pad_n + j]; }

	friend void apply_stencil(const Grid& old_, Grid& new_);
};

// best thresholds for parallelism and base case are probably hardware-dependent
// idk what the github actions benchmark runs on
void transpose(complex *m, size_t n, size_t full_n) {
	if (n <= 8) {
		#pragma omp simd
		for (size_t row = 0; row < n; row++) {
			#pragma omp simd
			for (size_t col = row + 1; col < n; col++) {
				std::swap(m[row * full_n + col], m[col * full_n + row]);
			}
		}
		return;
	}

	if (n >= 512) {
		// swapping the top right/bottom left quadrants
		#pragma omp parallel for
		for (size_t i = 0; i < (full_n * n / 2); i += full_n) {
			std::swap_ranges(m + i + (n/2), m + i + n, m + (full_n * n / 2) + i);
		}

		// unsure if this actually does anything,
		// documentation on omp orphan directives is pretty sparse
		#pragma omp task
		transpose(m, n/2, full_n);
		#pragma omp task
		transpose(m + n/2, n/2, full_n);
		#pragma omp task
		transpose(m + (full_n * n/2), n/2, full_n);
		#pragma omp task
		transpose(m + (full_n * n/2) + (n/2), n/2, full_n);
	} else {
		#pragma omp simd
		for (size_t i = 0; i < (full_n * n / 2); i += full_n) {
			std::swap_ranges(m + i + (n/2), m + i + n, m + (full_n * n / 2) + i);
		}

		transpose(m, n/2, full_n);
		transpose(m + n/2, n/2, full_n);
		transpose(m + (full_n * n/2), n/2, full_n);
		transpose(m + (full_n * n/2) + (n/2), n/2, full_n);
	}
}

void transpose(complex *m, size_t n) { transpose(m, n, n); }

// not parallelised because it will already be in the 2d fft
void fft(const complex *src, complex *dst, size_t n, size_t stride=1, double expnt=-1) {
	if (n == 4) {
		// 4-point butterfly
		// there is a way to apply this to the whole thing instead of just base case
		// ("split-radix" fft) but i am lazy
		complex lp = src[0];
		complex lk = src[2 * stride];
		complex rp = src[stride];
		complex rk = src[3 * stride];

		complex p1 = lp + lk;
		complex k1 = rp + rk;
		complex p2 = lp - lk;
		complex k2 = (rp - rk) * expnt * complex{0, 1};

		dst[0] = p1 + k1;
		dst[1] = p2 + k2;
		dst[2] = p1 - k1;
		dst[3] = p2 - k2;
	} else if (n == 2) { // safety
		dst[0] = src[0] + src[stride];
		dst[1] = src[0] - src[stride];
	} else {
		fft(src, dst, n / 2, stride * 2, expnt);
		fft(src + stride, dst + (n/2), n / 2, stride * 2, expnt);

		#pragma omp simd
		for (size_t i = 0; i < n / 2; i++) {
			complex p = dst[i];
			complex k = dst[i + (n/2)] * std::polar(1.0, expnt * TAU * i / n);
			dst[i] = p + k;
			dst[i + (n/2)] = p - k;
		}
	}
}

void ifft(const complex * src, complex * dst, size_t n) { fft(src, dst, n, 1, 1); }

void rfft(const double * src, complex * dst, size_t n) {
	fft(reinterpret_cast<const complex *>(src), dst, n / 2);

	// https://doi.org/10.1016/0022-460X(70)90075-1

	size_t half = n / 2;

	complex first = dst[0];
	dst[0] = first.real() + first.imag();
	dst[half] = first.real() - first.imag();

	#pragma omp simd
	for (size_t i = 1; i <= n / 4; i++) {
		complex z1 = dst[i];
		complex z2 = dst[half - i];
		complex a1 = 0.5 * (z1 + std::conj(z2));
		complex a2 = complex{0, 1} * std::polar(0.5, (-TAU * i / n)) * (std::conj(z2) - z1);

		dst[i] = a1 + a2;
		dst[half + i] = a1 - a2;
		dst[n - i] = std::conj(dst[i]);
		dst[half - i] = std::conj(dst[half + i]);
	}
}

// n is side len
// row-column fft2. vector-radix might have been faster, but this is simpler (i am lazy)
// and allows the algorithm to skip unnecessary parts
void rfft2(const vector<double>& src, vector<complex>& dst, size_t n, size_t rows, complex * scratch) {
	#pragma omp parallel
	{
		#pragma omp for
		for (size_t row = 0; row < rows; row++) {
			rfft(&src[row * n], scratch + (row * n), n);
		}

		#pragma omp single
		std::fill_n(scratch + (rows * n), (n - rows) * n, 0);
		#pragma omp single
		transpose(scratch, n);

		#pragma omp for
		for (size_t row = 0; row < n; row++) {
			fft(scratch + (row * n), &dst[row * n], n);
		}
	}
}

void ifft2(const vector<complex>& src, vector<complex>& dst, size_t n, size_t rows, size_t cols, complex * scratch) {
	#pragma omp parallel
	{
		#pragma omp for
		for (size_t row = 0; row < n; row++) {
			ifft(&src[row * n], scratch + (row * n), n);
		}

		#pragma omp single
		transpose(scratch, n);

		#pragma omp for
		for (size_t row = 0; row < rows; row++) {
			ifft(scratch + (row * n), &dst[row * n], n);
			for (size_t col = 0; col < cols; col++) {
				dst[row * n + col] /= n * n;
			}
		}
	}
}

// maybe there's some way to use less memory in grids?
// rn they have 4 scratch bufs where only 3 are needed
// in-place fft2 could do it but afaik that's slower
void apply_stencil(const Grid& old_, Grid& new_) {
	size_t n = new_.pad_n;
	size_t rows = new_.rows;
	size_t cols = new_.cols;

	rfft2(old_.data, new_.scratch1, n, rows, old_.scratch1.data());

	for (size_t i = 0; i < n * n; i++) { new_.scratch1[i] *= new_.kern_fft[i]; }

	ifft2(new_.scratch1, new_.scratch2, n, rows, cols, old_.scratch1.data());
	
	#pragma omp parallel
	{
		// copy only middle part
		#pragma omp for
		for (size_t row = 1; row < rows-1; row++) {
			for (size_t col = 1; col < cols-1; col++) {
				new_.data[n * row + col] = new_.scratch2[n * row + col].real();
			}
		}

		// copy sides
		#pragma omp for
		for (size_t row = 1; row < rows - 1; row++) {
			new_.data[n * row] = old_.data[n * row];
			new_.data[n * row + cols - 1] = old_.data[n * row + cols - 1];
		}

		// copy top and bottom
		#pragma omp task
		std::copy_n(&old_.data[0], cols, &new_.data[0]);
		#pragma omp task
		std::copy_n(&old_.data[n * (rows-1)], cols, &new_.data[n * (rows-1)]);
	}
}
