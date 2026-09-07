#pragma once

#include <vector>
#include <algorithm>
#include <complex>

using std::size_t, std::vector;
using complex = std::complex<double>;

const double TAU = 6.283185307179586;
const double stencil[9] = {
	0,     0.125, 0,
	0.125, 0.5,   0.125,
    0,     0.125, 0};

class Grid {
	size_t rows_;
	size_t cols_;
	vector<double> data;

	public:
	Grid(size_t rows, size_t cols):
		rows_{rows}, cols_{cols}, data{vector<double>(rows * cols, 0)} {}

	inline double& operator()(size_t i, size_t j) { return data[i * cols_ + j]; }
	inline double operator()(size_t i, size_t j) const { return data[i * cols_ + j]; }

	friend void apply_stencil(const Grid& old_grid, Grid& new_grid);
};

void boring_stencil(const vector<double>& old_data, vector<double>& new_data, size_t rows, size_t cols) {
	// copy top and bottom row
	std::copy_n(&old_data[0], cols, &new_data[0]);
	std::copy_n(&old_data[(rows - 1) * cols], cols,
				&new_data[(rows - 1) * cols]);

	#pragma omp parallel for
	for (size_t r = 1; r < rows - 1; r++) {
		// copy ends
		new_data[r * cols] = old_data[r * cols];
		new_data[(r+1) * cols - 1] = old_data[(r+1) * cols - 1];

		// apply stencil
		#pragma omp simd
		for (size_t i = r * cols + 1; i < (r+1) * cols - 1; i++) {
			const double u = old_data[i + cols];
			const double d = old_data[i - cols];
			const double l = old_data[i - 1];
			const double r = old_data[i + 1];
			new_data[i] = 0.5 * old_data[i] + 0.125 * (u + d + l + r);
		}
	}
}

// only works on square grids
// n represents side length
template<typename T>
void transpose(T *m, size_t n, size_t full_n) {
	// base case
	if (n == 2) {
		std::swap(*(m + 1), *(m + full_n));
		return;
	}

	#pragma omp parallel
	{
		// transpose four quadrants
		#pragma omp single
		{
			#pragma omp task
			transpose(m, n/2, full_n);
			#pragma omp task
			transpose(m + n/2, n/2, full_n);
			#pragma omp task
			transpose(m + (n * n / 2), n/2, full_n);
			#pragma omp task
			transpose(m + (n * n / 2) + (n / 2), n/2, full_n);
		}

		// swap top-right and bottom-left quadrants
		#pragma omp for
		for (size_t i = 0; i < (n * n / 2); i += n) {
			std::swap_ranges(m + (n/2) + i, m + n + i, m + (n * n / 2) + i);
		}
	}
	
}

template<typename T>
void transpose(T *m, size_t n) { transpose(m, n, n); }

// super basic rad2 fft
// not parallelised because it will already be parallelised in the top level
void fft(complex *src, complex *dst, size_t n, size_t stride=1, double expnt=-TAU) {
	if (n == 1) {
		*dst = *src;
		return;
	}

	fft(src, dst, n / 2, stride * 2, expnt);
	fft(src + stride, dst + (n/2), n / 2, stride * 2, expnt);

	complex w = std::polar(1.0, expnt / n);

	#pragma omp simd
	for (size_t i = 0; i < n / 2; i++) {
		complex p = dst[i];
		complex k = dst[i + (n/2)] * std::pow(w, i);
		dst[i] = p + k;
		dst[i + (n/2)] = p - k;
	}
}

void ifft(complex * src, complex * dst, size_t n) {
	fft(src, dst, n, 1, TAU);
	
	#pragma omp simd
	for (size_t i = 0; i < n; i++) { dst[i] /= n; }
}

// n is half the size of src
void rfft(double * src, complex * dst, size_t n) {
	fft((complex *) src, dst, n);

	dst[0] = 0.5 * (dst[0].real() + dst[0].imag());

	complex w = std::polar(1.0, TAU / n);
	for (size_t i = 1; i < n / 4; i++) {
		complex a1 = 0.5 * (dst[i] + std::conj(dst[(n/2) - i]));
		complex a2 = std::pow(w, -i) * complex(0, 0.5) * (std::conj(dst[(n/2)-i]) - dst[i]);

		dst[i] = 0.5 * (a1 + a2);
		dst[(n/2) + i] = 0.5 * (a1 - a2);
	}

	std::copy_n(dst, n/4, dst + n/4);
	std::copy_n(dst + (n/2), n/4, dst + (3*n/4));
}

// n is side len
void rfft2d(double * src, complex * dst, size_t n) {
	complex * scratch = new complex[n * n];

	#pragma omp parallel
	{
		#pragma omp for
		for (size_t i = 0; i < n * n; i += n) {
			rfft(src + i, scratch + i, n);
		}

		#pragma omp single
		transpose(scratch, n);

		#pragma omp for
		for (size_t i = 0; i < n * n; i += n) {
			fft(scratch + i, dst + i, n);
		}
		
		#pragma omp single
		transpose(dst, n);
	}

	delete [] scratch;
}

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
	const size_t rows = old_grid.rows_;
	const size_t cols = old_grid.cols_;

	// convenience
	const vector<double>& old_data = old_grid.data;
	vector<double>& new_data = new_grid.data;

	// only do the fft thing if we have a std::power-of-2 square grid. other grids
	// could be padded to do the same thing, but i can't be bothered because the
	// benchmark is on a 1024x1024 grid :)
	if (rows != cols || rows & (rows - 1) || cols & (cols - 1)) {
		boring_stencil(old_data, new_data, rows, cols);
	} else {
		
	}
}
