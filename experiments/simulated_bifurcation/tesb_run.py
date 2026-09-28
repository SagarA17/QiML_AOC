import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import os, time, torch
from ising import energy, TwinIsing
from sb_tabu import tesb_ideal, tesb_twin
res = torch.load("step25_results.pt", weights_only=False)
idx = [i for i, r in enumerate(res) if r["MN"] == 0.5 and r["N"] in (32, 44)]
f = "tesb_results.pt"; out = torch.load(f, weights_only=False) if os.path.exists(f) else {}
t0 = time.time()
for n, i in enumerate(idx):
    if i in out: continue
    r = res[i]; J = r["J"]; h = torch.zeros(r["N"])
    out[i] = {"TE_paper_ideal": energy(tesb_ideal(J, h, variant="paper", beta=0.1), J, h),
              "TE_hw_ideal": energy(tesb_ideal(J, h, variant="hardware", beta=0.05), J, h),
              "TE_hw_twin": energy(tesb_twin(TwinIsing(J, h, programming="new"), beta=0.05), J, h)}
    torch.save(out, f)
    if n % 5 == 4: print(f"[{time.time()-t0:4.0f}s] {n+1}/{len(idx)}", flush=True)
print("done", flush=True)
