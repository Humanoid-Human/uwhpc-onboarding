#include <iostream>
#include <cmath>
#include "../src/submission.hpp"
#include "test.hpp"

void test_fft();
void test_transpose();

int main(void) {
	test_fft();
	test_transpose();
}

bool far_apart(std::complex<double> a, std::complex<double> b) {
	const double DELTA_SQ = 0.001;
	auto d = a - b;
	return (d.real() * d.real()) > DELTA_SQ || (d.imag() * d.imag()) > DELTA_SQ;
}

void test_fft() {
	std::size_t pass = NUM_TESTS;
	auto res = new std::complex<double>[1024];

	// test fft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = fft_inps[i].size();

		fft(fft_inps[i].data(), res, len);
		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(fft_expects[i][j], res[j])) {
				pass--;
				std::cout << "[FAIL] (fft) at " << j << "/" << len
					<< " expected: " << fft_expects[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " fft tests passed" << std::endl;

	pass = NUM_TESTS;
	
	// test ifft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = fft_inps[i].size();

		ifft(fft_expects[i].data(), res, len);
		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(fft_inps[i][j], res[j])) {
				pass--;
				std::cout << "[FAIL] (ifft) at " << j << "/" << len
					<< " expected: " << fft_inps[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " ifft tests passed" << std::endl;

	// test rfft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = rfft_inps[i].size();

		rfft(rfft_inps[i].data(), res, len);
		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(rfft_expects[i][j], res[j])) {
				pass--;
				std::cout << "[FAIL] (rfft) at " << j << "/" << len
					<< " expected: " << rfft_expects[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " rfft tests passed" << std::endl;

	delete[] res;
}

void test_transpose() {
	std::size_t pass = NUM_TESTS;
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = transpose_inps[i].size();
		transpose(transpose_inps[i].data(), sqrt(len));
		for (std::size_t j = 0; j < NUM_TESTS; j++) {
			if (transpose_inps[i][j] != transpose_expects[i][j]) {
				pass--;
				std::cout << "[FAIL] (transpose) at " << j << "/" << len
					<< " expected: " << transpose_expects[i][j]
					<< ", got: " << transpose_inps[i][j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " transpose tests passed" << std::endl;
}
