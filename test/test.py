import numpy as np
from sys import maxsize
from math import sqrt
import scipy

NUM_TESTS = 20

def print_cmplx(z):
    return f"({z.real}, {z.imag})"

formatter = { 'complex_kind': print_cmplx }

np.set_printoptions(formatter=formatter, threshold=maxsize, linewidth=120, suppress=True, floatmode="fixed")

def print_fmt(a):
    string = np.array2string(a, separator=",")
    print(string.replace("[","{").replace("]","}").replace("(","{").replace(")","}") + ",")

inp = []

rng = np.random.default_rng(seed=42)

print("#include <vector>")
print("#include <complex>\n")

print(f"const std::size_t NUM_TESTS = {NUM_TESTS};\n")

# fft
print(f"std::vector<std::complex<double>> fft_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    inp.append(np.zeros(2 ** rng.integers(2, 6), dtype=np.complex128))
    for j in range(len(inp[i])):
        inp[i][j] = complex(rng.uniform(0, 32), rng.uniform(0, 32))
    print_fmt(inp[i])
print("};\n")

print(f"std::vector<std::complex<double>> fft_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    print_fmt(scipy.fft.fft(inp[i]))
print("};")

inp.clear()

# rfft
print(f"std::vector<double> rfft_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    inp.append(np.zeros(2 ** (2 * rng.integers(3, 5))))
    for j in range(len(inp[i])):
        inp[i][j] = rng.uniform(0, 32)
    print_fmt(inp[i])
print("};\n")

print(f"std::vector<std::complex<double>> rfft_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    print_fmt(scipy.fft.fft(inp[i]))
print("};")

# transpose
t_inp = []
print(f"std::vector<std::complex<double>> transpose_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    t_inp.append(np.zeros(2 ** (2 * rng.integers(2, 5)), dtype=np.complex128))
    for j in range(len(t_inp[i])):
        t_inp[i][j] = complex(rng.uniform(0, 32), rng.uniform(0, 32))
    print_fmt(t_inp[i])
print("};")

print(f"std::vector<std::complex<double>> transpose_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    n = int(sqrt(len(t_inp[i])))
    print_fmt(np.transpose(t_inp[i].reshape(n, n)).flatten())
print("};")

# rfft2 & ifft2
print(f"std::vector<std::complex<double>> rfft2_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    n = int(sqrt(len(inp[i])))
    print_fmt(scipy.fft.fft2(inp[i].reshape((n, n))).flatten())
print("};")