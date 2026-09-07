import numpy as np

NUM_TESTS = 32

def fmt(string):
    return string.replace("[","{").replace("]","}").replace("(","{").replace(")","}")

inp = []

rng = np.random.default_rng(seed=42)

print("#include <vector>")
print("#include <complex>\n")

print(f"const std::size_t NUM_TESTS = {NUM_TESTS};\n")

# fft
print(f"std::vector<std::complex<double>> fft_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    inp.append(np.zeros(2 ** rng.integers(3, 11), dtype=np.complex128))
    for j in range(len(inp[i])):
        inp[i][j] = complex(rng.uniform(0, 32), rng.uniform(0, 32))
    print(fmt(str([(z.real, z.imag) for z in inp[i]])) + ",")
print("};\n")

print(f"std::vector<std::complex<double>> fft_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    ans_fmt = [(z.real, z.imag) for z in np.fft.fft(inp[i])]
    print(fmt(str(ans_fmt)) + ",")
print("};")

inp.clear()

# rfft
print(f"std::vector<double> rfft_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    inp.append(np.zeros(2 ** rng.integers(3, 11)))
    for j in range(len(inp[i])):
        inp[i][j] = rng.uniform(0, 32)
    print(fmt(str(list(inp[i]))) + ",")
print("};\n")

print(f"std::vector<std::complex<double>> rfft_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    ans_fmt = [(z.real, z.imag) for z in np.fft.fft(inp[i])]
    print(fmt(str(ans_fmt)) + ",")
print("};")

inp.clear();

# transpose
print(f"std::vector<double> transpose_inps[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    sidelen = 2 ** rng.integers(3, 5)
    inp.append(np.zeros((sidelen, sidelen), dtype=np.ubyte))
    for row in inp[i]:
        for x in row:
            x = rng.integers(0, 256)
    print(fmt(str(list(inp[i].flatten()))) + ",")
print("};")

print(f"std::vector<double> transpose_expects[{NUM_TESTS}] = {{")
for i in range(NUM_TESTS):
    ans = list(np.transpose(inp[i]).flatten())
    print(fmt(str(ans)) + ",")
print("};")
