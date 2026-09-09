import numpy as np
from sys import maxsize
from math import sqrt
import scipy

NUM_TESTS = 16

np.set_printoptions(threshold=maxsize, linewidth=120,  suppress=True, floatmode="fixed")

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
    inp.append(np.zeros(2 ** rng.integers(3, 6), dtype=np.complex128))
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
    inp.append(np.zeros(2 ** (2 * rng.integers(2, 4))))
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
print(f"std::vector<double> transpose_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    sidelen = 2 ** rng.integers(3, 5)
    t_inp.append(rng.random((sidelen, sidelen))) 
    print_fmt(t_inp[i].flatten())
print("};")

print(f"std::vector<double> transpose_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    print_fmt(np.transpose(t_inp[i]).flatten())
print("};")

# 2d rfft
print(f"std::vector<std::complex<double>> rfft2_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    n = int(sqrt(len(inp[i])))
    print_fmt(scipy.fft.fft2(inp[i].reshape((n, n))).flatten())
print("};")

# stencil
kern = [[0, 1/8, 0], [1/8, 1/2, 1/8], [0, 1/8, 0]]
print(f"std::vector<double> stencil_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    n = int(sqrt(len(inp[i])))
    ans = scipy.signal.convolve2d(inp[i].reshape((n, n)), kern, "valid");
    for j in range(1, n-1):
        for k in range(1, n-1):
            inp[i][j * n + k] = ans[j-1][k-1]
    print_fmt(inp[i])
print("};")