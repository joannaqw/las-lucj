import ffsim
GIB_PER_AMPLITUDE = 128 / 8 / 1024**3

norb = 24
nelec = (13, 13)

dim = ffsim.dim(norb, nelec)
gib = GIB_PER_AMPLITUDE * dim

print(f"Storage required for state vector: {gib:.2f} GiB")
