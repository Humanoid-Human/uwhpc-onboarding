#pragma once

#include <vector>
#include <algorithm>
#include <complex>
#include <iostream>

using std::size_t, std::vector;
using complex = std::complex<double>;

const double TAU = 6.283185307179586;

// assumes cols >= rows
template <typename T>
class PaddedGrid {
	T *data;
	size_t rows;
	size_t cols;

	PaddedGrid(T * data, size_t rows, size_t cols):
		data{data}, rows{rows}, cols{cols} {}

	T operator[](size_t i) {
		if (i / cols >= rows) { return 0; }
		return data[i];
	}
};

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

	//#pragma omp parallel for
	for (size_t r = 1; r < rows - 1; r++) {
		// copy ends
		new_data[r * cols] = old_data[r * cols];
		new_data[(r+1) * cols - 1] = old_data[(r+1) * cols - 1];

		// apply stencil
		//#pragma omp simd
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
void transpose(complex *m, size_t n, size_t full_n) {
	// base case
	if (n == 2) {
		complex tmp = m[1];
		m[1] = m[full_n];
		m[full_n] = tmp;
		return;
	}

	//#pragma omp parallel
	{
		// transpose four quadrants
		//#pragma omp single
		{
			//#pragma omp task
			transpose(m, n/2, full_n);
			//#pragma omp task
			transpose(m + n/2, n/2, full_n);
			//#pragma omp task
			transpose(m + (full_n * n/2), n/2, full_n);
			//#pragma omp task
			transpose(m + (full_n * n/2) + (n/2), n/2, full_n);
		}

		// swap top-right and bottom-left quadrants
		//#pragma omp for
		for (size_t i = 0; i < (full_n * n / 2); i += full_n) {
			std::swap_ranges(m + i + (n/2), m + i + n, m + (full_n * n / 2) + i);
		}
	}
}

void transpose(complex *m, size_t n) { transpose(m, n, n); }

// super basic rad2 fft
// not parallelised because it will already be in the 2d fft
void fft(complex *src, complex *dst, size_t n, size_t stride=1, double expnt=-TAU) {
	if (n == 1) {
		*dst = *src;
		return;
	}

	fft(src, dst, n / 2, stride * 2, expnt);
	fft(src + stride, dst + (n/2), n / 2, stride * 2, expnt);

	complex w = std::polar(1.0, expnt / n);

	//#pragma omp simd
	for (size_t i = 0; i < n / 2; i++) {
		complex p = dst[i];
		complex k = dst[i + (n/2)] * std::pow(w, i);
		dst[i] = p + k;
		dst[i + (n/2)] = p - k;
	}
}

void ifft(complex * src, complex * dst, size_t n) {
	fft(src, dst, n, 1, TAU);
}

void rfft(double * src, complex * dst, size_t n) {
	fft(reinterpret_cast<complex*>(src), dst, n / 2);

	// https://doi.org/10.1016/0022-460X(70)90075-1
	
	complex first = dst[0];
	dst[0] = first.real() + first.imag();
	dst[n / 2] = first.real() - first.imag();

	complex w = std::polar(1.0, -TAU / n);
	for (size_t i = 1; i <= n / 4; i++) {
		complex a1 = 0.5 * (dst[i] + std::conj(dst[n / 2 - i]));
		complex a2 = complex(0, 0.5) * std::pow(w, i) * (std::conj(dst[n / 2 - i]) - dst[i]);

		// according to the paper these should be multiplied by 0.5
		// i have no idea why but that doesn't agree with scipy's algorithm
		dst[i] = a1 + a2;
		dst[(n/2) + i] = a1 - a2;
		dst[n - i] = std::conj(a1 + a2);
		dst[(n/2) - i] = std::conj(a1 - a2);
	}
}

// n is side len
void rfft2(double * src, complex * dst, size_t n, complex * scratch) {
	//#pragma omp parallel
	{
		//#pragma omp for
		for (size_t row = 0; row < n; row++) {
			rfft(src + (row * n), scratch + (row * n), n);
		}

		//#pragma omp single
		transpose(scratch, n);

		//#pragma omp for
		for (size_t row = 0; row < n; row++) {
			fft(scratch + (row * n), dst + (row * n), n);
		}
		
		//#pragma omp single
		transpose(dst, n);
	}
}

void ifft2(complex * src, complex * dst, size_t n, complex * scratch) {
	//#pragma omp parallel
	{
		//#pragma omp for
		for (size_t row = 0; row < n; row++) {
			ifft(src + (row * n), scratch + (row * n), n);
		}

		//#pragma omp single
		transpose(scratch, n);

		//#pragma omp for
		for (size_t row = 0; row < n; row++) {
			ifft(scratch + (row * n), dst + (row * n), n);
		}

		//#pragma omp single
		transpose(dst, n);

		//#pragma omp for
		for (size_t i = 0; i < n * n; i++) {
			dst[i] /= n * n;
		}
	}
}

// TODO: don't allocate so much memory
// - store it in the Grid itself so it can be reused?
void fft_stencil(double * src, double * dst, size_t n) {
	auto res = new complex[4 * n * n];
	auto scratch = new complex[4 * n * n];
	auto src_fft = new complex[4 * n * n];
	auto padded = new double[4 * n * n];

	// fill in the stencil
	std::fill_n(padded, n * n, 0);
	padded[1] = 0.125;
	padded[n] = 0.125;
	padded[n+1] = 0.5;
	padded[n+2] = 0.125;
	padded[2 * n + 1] = 0.125;

	rfft2(padded, res, 2 * n, scratch);

	std::fill_n(padded, n * n, 0);
	for (size_t i = 0; i < n; i++) {
		std::copy_n(src + (i*n), n, padded + (i*n*2));
	}

	rfft2(padded, src_fft, 2 * n, scratch);

	//#pragma omp parallel
	{
		//#pragma omp for
		for (size_t i = 0; i < n * n; i++) {
			src_fft[i] *= res[i];
		}

		//#pragma omp single
		ifft2(src_fft, res, 2 * n, scratch);
		
		//#pragma omp for
		for (size_t row = 1; row < n-1; row++) {
			//#pragma omp simd
			for (size_t col = 1; col < n-1; col++) {
				dst[n * row + col] = res[2 * n * row + col].real();	
			}
		}

		//#pragma omp for
		for (size_t i = 1; i < n - 1; i++) {
			dst[i*n] = src[i*n];
			dst[(i+1) * n - 1] = src[(i+1) * n - 1];
		}
	}

	std::copy_n(src, n, dst);
	std::copy_n(src + n * (n-1), n, dst + n * (n-1));
	
	delete [] padded;
	delete [] res;
	delete [] src_fft;
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
