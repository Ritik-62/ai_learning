import torch
import time

size = 5000

# CPU par matrices
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# CPU benchmark
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start

print(f"CPU: {cpu_time:.3f}s")


# Apple GPU (MPS) benchmark
if torch.backends.mps.is_available():

    # CPU se Apple GPU par data bhejo
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    start = time.time()

    # Matrix multiplication GPU par
    c_mps = a_mps @ b_mps

    # GPU ko calculation finish karne do
    torch.mps.synchronize()

    mps_time = time.time() - start

    print(f"MPS: {mps_time:.3f}s")
    print(f"Speedup: {cpu_time / mps_time:.1f}x")

else:
    print("MPS is not available.")