#include <iostream>
#include <cmath>
#include "../src/submission.hpp"
#include "test.hpp"

using complex = std::complex<double>;

void test_fft();
void test_ifft();
void test_rfft();
void test_transpose();
void test_rfft2();
void test_ifft2();

int main(void) {
	test_fft();
	test_ifft();
	test_rfft();
	test_transpose();
	test_rfft2();
	test_ifft2();
}

bool far_apart(complex a, complex b) {
	const double DELTA_SQ = 0.001;
	auto d = a - b;
	return (d.real() * d.real()) > DELTA_SQ || (d.imag() * d.imag()) > DELTA_SQ;
}

void test_fft() {
	std::size_t pass = NUM_TESTS;
	auto res = new complex[1024];

	// test fft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = fft_inps[i].size();

		fft(fft_inps[i].data(), res, len);
		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(fft_expects[i][j], res[j])) {
				pass--;
				std::cout << "[FAIL] (fft " << i << ") at index " << j << " (len " << len
					<< ") expected: " << fft_expects[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " fft tests passed" << std::endl;

	delete[] res;
}

void test_ifft() {
	std::size_t pass = NUM_TESTS;
	auto res = new complex[1024];
	
	// test ifft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = fft_inps[i].size();
		ifft(fft_expects[i].data(), res, len);

		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(fft_inps[i][j], res[j] / (double) len)) {
				pass--;
				std::cout << "[FAIL] (ifft " << i << ") at index " << j << " (len " << len
					<< ") expected: " << rfft_inps[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " ifft tests passed" << std::endl;

	delete [] res;
}

void test_rfft() {
	std::size_t pass = NUM_TESTS;
	auto res = new complex[1024];

	// test rfft
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = rfft_inps[i].size();

		rfft(rfft_inps[i].data(), res, len);
		for (std::size_t j = 0; j < len; j++) {
			if (far_apart(rfft_expects[i][j], res[j])) {
				pass--;
				std::cout << "[FAIL] (rfft " << i << ") at index " << j << " (len " << len
					<< ") expected: " << rfft_expects[i][j]
					<< ", got: " << res[j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " rfft tests passed" << std::endl;

	delete [] res;
}

void test_transpose() {
	std::size_t pass = NUM_TESTS;
	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t len = transpose_inps[i].size();
		transpose(transpose_inps[i].data(), sqrt(len));
		for (std::size_t j = 0; j < len; j++) {
			if (transpose_inps[i][j] != transpose_expects[i][j]) {
				pass--;
				std::cout << "[FAIL] (transpose " << i << ") at index " << j << " (len " << len
					<< ") expected: " << transpose_expects[i][j]
					<< ", got: " << transpose_inps[i][j] << std::endl;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " transpose tests passed" << std::endl;
}

void test_rfft2() {
	std::size_t pass = NUM_TESTS;
	std::vector<complex> res(256);
	auto scratch = new complex[256];

	for (std::size_t i = 0; i < NUM_TESTS; i++) {
		std::size_t n = sqrt(rfft_inps[i].size());
		rfft2(rfft_inps[i], res, n, n, scratch);
		transpose(res.data(), n);

		for (size_t j = 0; j < n * n; j++) {
			if (far_apart(res[j], rfft2_expects[i][j])) {
				std::cout << "[FAIL] (rfft2 " << i << ") at index " << j << " (len " << n * n
					<< ") expected: " << rfft2_expects[i][j]
					<< ", got: " << res[j] << std::endl;
				pass--;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " rfft2 tests passed" << std::endl;

	delete [] scratch;
}

void test_ifft2() {
	std::size_t pass = NUM_TESTS;
	std::vector<complex> res(256);
	auto scratch = new complex[256];
	
	for (std::size_t i = 1; i < NUM_TESTS; i++) {
		std::size_t n = sqrt(rfft2_expects[i].size());
		ifft2(rfft2_expects[i], res, n, n, scratch);
		transpose(res.data(), n);

		for (size_t j = 0; j < n * n; j++) {
			if (far_apart(rfft_inps[i][j], res[j])) {
				std::cout << "[FAIL] (ifft2 " << i << ") at index " << j << " (len " << n * n
					<< ") expected: " << rfft_inps[i][j]
					<< ", got: " << res[j] << std::endl;
				pass--;
				break;
			}
		}
	}
	std::cout << pass << "/" << NUM_TESTS << " ifft2 tests passed" << std::endl;

	delete [] scratch;
}
