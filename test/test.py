import numpy as np
from sys import maxsize
from math import sqrt

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
    print_fmt(np.fft.fft(inp[i]))
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
    print_fmt(np.fft.fft(inp[i]))
print("};")

# 2d rfft
print(f"std::vector<std::complex<double>> rfft2_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    n = int(sqrt(len(inp[i])))
    print_fmt(np.fft.fft2(inp[i].reshape((n, n))).flatten())
print("};")

inp.clear();

# transpose
print(f"std::vector<double> transpose_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    sidelen = 2 ** rng.integers(3, 5)
    inp.append(rng.random((sidelen, sidelen))) 
    print_fmt(inp[i].flatten())
print("};")

print(f"std::vector<double> transpose_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    print_fmt(np.transpose(inp[i]).flatten())
print("};")
