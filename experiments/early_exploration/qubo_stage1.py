"""Stage 1+2: instances, exact references, SA baseline, and idealized-AOC schedule search."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import time, torch, numpy as np
from ising import *
t0 = time.time()
# ---- references on test instances
fam = {}
for N in [16, 24, 32, 44]:
    fam[("wishart", N)] = [wishart_planted(N, seed=100 + i) for i in range(10)]
for N in [16, 24]:
    fam[("3reg", N)] = [three_regular(N, seed=100 + i) for i in range(10)]
refs = {}
for key, insts in fam.items():
    E = []
    for J, h, t in insts:
        E.append(float(energy(t, J, h)) if t is not None else brute_force(J, h))
    refs[key] = E
# sanity: planted state is optimal where brute force is feasible
for N in [16, 24]:
    J, h, t = fam[("wishart", N)][0]
    print(f"check Wishart N={N}: planted E {float(energy(t, J, h)):.5f} vs brute force {brute_force(J, h):.5f}")
print(f"references ready ({time.time()-t0:.0f}s)")
# ---- schedule search on separate tuning instances (idealised dynamics)
tune = [wishart_planted(32, seed=s) for s in range(5)]
tref = [float(energy(t, J, h)) for J, h, t in tune]
best = None
print("\nidealised AOC, tuning grid on Wishart N=32 (T=1000 iterations, 100 runs each):")
for alpha0 in [0.5, 1.0, 2.0]:
    for beta0 in [0.1, 0.25, 0.5, 0.75]:
        for gamma in [0.0, 0.5]:
            ps = []
            for (J, h, _), Er in zip(tune, tref):
                s = aoc_ideal(J, h, alpha0=alpha0, beta0=beta0, gamma=gamma)
                ps.append(float((energy(s, J, h) <= Er + 1e-5).float().mean()))
            p = float(np.mean(ps))
            if best is None or p > best[0]: best = (p, alpha0, beta0, gamma)
            print(f"  alpha0={alpha0:3.1f} beta0={beta0:4.2f} gamma={gamma:3.1f}: success/run {p:.3f}")
print(f"best: success {best[0]:.3f} at alpha0={best[1]} beta0={best[2]} gamma={best[3]}")
torch.save({"fam": fam, "refs": refs, "best": best}, "qubo_stage1.pt")
