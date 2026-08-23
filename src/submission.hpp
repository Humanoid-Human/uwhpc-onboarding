#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
	private:
	  std::size_t rows_;
	  std::size_t cols_;
	  std::vector<double> data;

	public:
	  Grid(std::size_t rows, std::size_t cols);

	  double& operator()(std::size_t i, std::size_t j);
	  double  operator()(std::size_t i, std::size_t j) const;

	  const std::vector<double>& get_const_data() const { return data; }
	  std::vector<double>& get_data() { return data; }
	  std::size_t get_rows() const { return rows_; }
	  std::size_t get_cols() const { return cols_; }

	  friend void apply_stencil(const Grid& old_grid, Grid& new_grid);
};

Grid::Grid(std::size_t rows, std::size_t cols):
	rows_{rows}, cols_{cols}, data{std::vector<double>(rows * cols, 0)} {}

inline double& Grid::operator()(std::size_t i, std::size_t j) { return data[i * cols_ + j]; }
inline double Grid::operator()(std::size_t i, std::size_t j) const { return data[i * cols_ + j]; }

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
	const std::size_t rows = old_grid.rows_;
	const std::size_t cols = old_grid.cols_;

	// copy top and bottom row
	std::copy_n(&old_grid.data[0], cols, &new_grid.data[0]);
	std::copy_n(&old_grid.data[(rows - 1) * cols], cols,
				&new_grid.data[(rows - 1) * cols]);

	#pragma omp parallel for
	for (int r = 1; r < rows - 1; r++) {
		// copy ends
		new_grid.data[r * cols] = old_grid.data[r * cols];
		new_grid.data[(r+1) * cols - 1] = old_grid.data[(r+1) * cols - 1];

		// actual stencil
		// pragma omp simd doesn't seem to do anything here
		for (int i = r * cols + 1; i < (r+1) * cols - 1; i++) {
			const double u = old_grid.data[i + cols];
			const double d = old_grid.data[i - cols];
			const double l = old_grid.data[i - 1];
			const double r = old_grid.data[i + 1];
			new_grid.data[i] = 0.5 * old_grid.data[i] + 0.125 * (u + d + l + r);
		}
	}
}
