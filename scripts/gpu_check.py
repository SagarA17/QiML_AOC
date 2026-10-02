"""Compute-node sanity check: torch/CUDA compatibility, visible GPUs and their load, twin import, a timed batch."""
import os, subprocess, sys, time
import torch

print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda}); python {sys.version.split()[0]}")
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
                      "--format=csv"], capture_output=True, text=True).stdout)
print(f"cuda available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
for d in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(d)
    print(f"  cuda:{d} {torch.cuda.get_device_name(d)}  free {free/2**30:.1f}/{total/2**30:.1f} GiB")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "qiml_aoc"))
import common  # noqa: F401  (imports the twin)
print("twin import ok")

# Batched dSB-sized workload: B problems x R runs x N spins.
if torch.cuda.is_available():
    B, R, N, T = 4096, 100, 16, 1000
    J = torch.randn(B, N, N, device="cuda"); J = (J + J.transpose(1, 2)) / 2
    x = 0.1 * torch.randn(B, R, N, device="cuda"); y = torch.zeros_like(x)
    torch.cuda.synchronize(); t = time.time()
    for k in range(T):
        y = y + (-(1 - k / T) * x + 0.05 * torch.bmm(torch.sign(x), J)); x = x + y
        w = x.abs() > 1; x = torch.where(w, torch.sign(x), x); y = torch.where(w, torch.zeros_like(y), y)
    torch.cuda.synchronize()
    print(f"batched dSB: {B} problems x {R} runs x N={N}, {T} steps in {time.time()-t:.2f} s")
