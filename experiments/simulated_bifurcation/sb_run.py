import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import os, time, torch
from ising import *
from sb import sb_ideal, sb_twin
res = torch.load("step25_results.pt", weights_only=False)
f = "sb_results.pt"; out = torch.load(f, weights_only=False) if os.path.exists(f) else []
t0 = time.time()
for i, r in enumerate(res):
    if i < len(out): continue
    J = r["J"]; h = torch.zeros(r["N"])
    E = {"dSB_ideal": energy(sb_ideal(J, h, runs=1000, variant="dSB"), J, h),
         "dSB_twin_g1": energy(sb_twin(TwinIsing(J, h, programming="new"), runs=1000, gamma=1.0), J, h),
         "dSB_twin_g095": energy(sb_twin(TwinIsing(J, h, programming="new"), runs=1000, gamma=0.95), J, h)}
    out.append(E); torch.save(out, f)
    if i % 5 == 4: print(f"[{time.time()-t0:4.0f}s] {i+1}/90", flush=True)
print("done", flush=True)
