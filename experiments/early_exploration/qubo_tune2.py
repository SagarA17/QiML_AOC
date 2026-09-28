import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np, itertools
from ising import *
tune = [wishart_planted(32, seed=s) for s in range(5)]
tref = [float(energy(t, J, h)) for J, h, t in tune]
res = []
for a0, beta0, gamma in itertools.product([0.1, 0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], [0.0, 0.1, 0.3]):
    ps = [float((energy(aoc_paper(J, h, a0=a0, beta0=beta0, gamma=gamma), J, h) <= Er + 1e-5).float().mean())
          for (J, h, _), Er in zip(tune, tref)]
    res.append((np.mean(ps), np.min(ps), a0, beta0, gamma))
res.sort(reverse=True)
print("SA reference on this tuning set: mean success/run 0.494 (1000 sweeps)")
print("paper-faithful idealised AOC (T=1000, 100 runs) -- mean success, worst instance, a0, beta0, gamma:")
for r in res[:8]: print("  ", tuple(round(float(v), 3) for v in r))
torch.save(res[0], "qubo_best_paper.pt")
