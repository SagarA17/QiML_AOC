import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time, torch, numpy as np
from ising import *
st = torch.load("qubo_stage1.pt"); fam, refs = st["fam"], st["refs"]
A0, B0, G0 = 0.6, 0.2, 0.1
def stats(s, J, h, Er):
    E = energy(s, J, h)
    return float((E <= Er + 1e-5).float().mean()), float(((E - Er) / abs(Er)).mean())
t0 = time.time()
print(f"schedule a0={A0}, beta0={B0}, gamma={G0}; T=1000 iterations; 100 runs per instance; 10 test instances per row")
print(f"{'family':>8s} {'N':>3s} | {'idealised':>22s} | {'twin, OLD programming':>22s} | {'twin, NEW programming':>22s}")
print(f"{'':>12s} | {'succ/run  solved  gap':>22s} | {'succ/run  solved  gap':>22s} | {'succ/run  solved  gap':>22s}")
for key in [("wishart", 16), ("wishart", 24), ("wishart", 32), ("wishart", 44), ("3reg", 16), ("3reg", 24)]:
    row = {"ideal": [], "old": [], "new": []}
    for (J, h, _), Er in zip(fam[key], refs[key]):
        row["ideal"].append(stats(aoc_paper(J, h, a0=A0, beta0=B0, gamma=G0), J, h, Er))
        for prog in ["old", "new"]:
            tw = TwinIsing(J, h, programming=prog)
            row[prog].append(stats(twin_run_paper(tw, a0=A0, beta0=B0, gamma=G0), J, h, Er))
    cells = []
    for k in ["ideal", "old", "new"]:
        p = np.array([r[0] for r in row[k]]); gap = np.array([r[1] for r in row[k]])
        cells.append(f"{p.mean():6.3f}  {int((p > 0).sum()):2d}/10  {gap.mean()*100:5.1f}%")
    print(f"{key[0]:>8s} {key[1]:3d} | " + " | ".join(f"{c:>22s}" for c in cells), flush=True)
print(f"({time.time()-t0:.0f}s)  gap = mean relative energy gap to the optimum over all runs")
