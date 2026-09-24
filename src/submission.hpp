#pragma once

#include <algorithm>
#include <complex>
#include <climits>

using std::size_t;
using complex = std::complex<double>;

const double TAU = 6.283185307179586;

class Grid {
	size_t rows;
	size_t cols;
	size_t pad_n;

	double * data;
	double * kernel;
	complex * scratch1;
	complex * scratch2;

	public:
	Grid(size_t r, size_t c): rows{r}, cols{c} {
		// padding for fft (next power of 2)
		pad_n = (r < c ? c : r) + 1; // would be -1 but kernel is 3-wide so +2 on top
		for (size_t shift = 1; shift < sizeof(size_t) * CHAR_BIT; shift++) {
			pad_n |= pad_n >> shift;
		}
		pad_n++;

		size_t data_size = pad_n * pad_n;
		data = new double[data_size];

		// scratch arrays & kernel, so main thing doesnt have to allocate
		// TODO: find a way to not allocate so much memory?
		scratch1 = new complex[data_size];
		scratch2 = new complex[data_size];

		kernel = new double[data_size];
		std::fill_n(kernel, data_size, 0);
		kernel[0] = 0.5;
		kernel[1] = 0.125;
		kernel[pad_n - 1] = 0.125;
		kernel[pad_n] = 0.125;
		kernel[pad_n * (pad_n - 1)] = 0.125;
	}

	~Grid() {
		delete [] data;
		delete [] scratch1;
		delete [] scratch2;
		delete [] kernel;
	}

	inline double& operator()(size_t i, size_t j) { return data[i * pad_n + j]; }
	inline double operator()(size_t i, size_t j) const { return data[i * pad_n + j]; }

	friend void apply_stencil(const Grid& old_grid, Grid& new_grid);
};

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
		#pragma omp parallel
		{
			#pragma omp for
			for (size_t i = 0; i < (full_n * n / 2); i += full_n) {
				std::swap_ranges(m + i + (n/2), m + i + n, m + (full_n * n / 2) + i);
			}

			#pragma omp single
			{
				#pragma omp task
				transpose(m, n/2, full_n);
				#pragma omp task
				transpose(m + n/2, n/2, full_n);
				#pragma omp task
				transpose(m + (full_n * n/2), n/2, full_n);
				#pragma omp task
				transpose(m + (full_n * n/2) + (n/2), n/2, full_n);
			}
		}	
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

void transpose(complex *m, size_t n) {
	transpose(m, n, n);
}

// not parallelised because it will already be in the 2d fft
void fft(const complex *src, complex *dst, size_t n, size_t stride=1, double expnt=-1) {
	if (n == 4) {
		// 4-point butterfly
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
		complex a1 = 0.5 * (dst[i] + std::conj(dst[half - i]));
		complex a2 = complex(0, 0.5) * std::polar(1.0, -TAU * i / n) * (std::conj(dst[half - i]) - dst[i]);

		dst[i] = a1 + a2;
		dst[half + i] = a1 - a2;
		dst[n - i] = std::conj(dst[i]);
		dst[half - i] = std::conj(dst[half + i]);
	}
}

// n is side len
void rfft2(const double * src, complex * dst, size_t n, complex * scratch) {
	#pragma omp parallel for
	for (size_t row = 0; row < n; row++) {
		rfft(src + (row * n), scratch + (row * n), n);
	}

	transpose(scratch, n);

	#pragma omp parallel for
	for (size_t row = 0; row < n; row++) {
		fft(scratch + (row * n), dst + (row * n), n);
	}
	
	transpose(dst, n);
}

void ifft2(const complex * src, complex * dst, size_t n, complex * scratch) {
	#pragma omp parallel for
	for (size_t row = 0; row < n; row++) {
		ifft(src + (row * n), scratch + (row * n), n);
	}

	transpose(scratch, n);

	#pragma omp parallel for
	for (size_t row = 0; row < n; row++) {
		ifft(scratch + (row * n), dst + (row * n), n);
	}

	transpose(dst, n);

	size_t len = n * n;
	complex div = 1.0 / len;
	#pragma omp simd
	for (size_t i = 0; i < len; i++) { dst[i] *= div; }
}

// TODO:
// - try fake padding thing (per-row instead of whole thing?)
// - some way to use less memory in grids? rn they have 4 scratch bufs where only 3 are needed
// in-place fft works but probly slower

// NOTE: i know this is slow af for small grids, but i didn't feel like writing
// the generic nested loop thing just for small grids that don't matter anyway
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
	size_t full_n = new_grid.pad_n;
	size_t rows = new_grid.rows;
	size_t cols = new_grid.cols;
	
	double * dst = new_grid.data;
	double * src = old_grid.data;

	complex * scratch = old_grid.scratch1;
	complex * kern_fft = new_grid.scratch1;
	complex * src_fft = new_grid.scratch2;

	// fft kern and src matrix
	rfft2(old_grid.kernel, kern_fft, full_n, scratch);
	rfft2(src, src_fft, full_n, scratch);

	#pragma omp simd
	for (size_t i = 0; i < full_n * full_n; i++) { src_fft[i] *= kern_fft[i]; }

	// reuse kern_fft, ifft
	ifft2(src_fft, kern_fft, full_n, scratch);
	
	// copy only middle part
	#pragma omp parallel for
	for (size_t row = 1; row < rows-1; row++) {
		#pragma omp simd
		for (size_t col = 1; col < cols-1; col++) {
			dst[full_n * row + col] = kern_fft[full_n * row + col].real();
		}
	}

	// copy sides
	// TODO: find a way to skip this part after the first loop?
	// since they stay the same
	#pragma omp parallel for
	for (size_t row = 1; row < rows - 1; row++) {
		dst[full_n * row] = src[full_n * row];
		dst[(row+1) * full_n - 1] = src[(row+1) * full_n - 1];
	}

	// copy top and bottom
	std::copy_n(dst, cols, src);
	std::copy_n(dst + full_n * (rows-1), cols, src + full_n * (rows-1));
}
