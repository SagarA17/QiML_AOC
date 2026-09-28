import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np, itertools
from ising import *
tune = [wishart_planted(32, seed=s) for s in range(5)]
tref = [float(energy(t, J, h)) for J, h, t in tune]
sa = [float((energy(simulated_annealing(J, h, runs=100, sweeps=1000), J, h) <= Er + 1e-5).float().mean()) for (J,h,_),Er in zip(tune,tref)]
print(f"SA reference on tuning set (1000 sweeps): mean success/run {np.mean(sa):.3f}  per-instance {np.round(sa,2)}")
res = []
for alpha0, beta0, gamma, init in itertools.product([0.5, 1.0, 1.5], [0.25, 0.5, 0.75, 1.0], [0.0, 0.5], [0.3, 1.0, 3.0]):
    ps = [float((energy(aoc_ideal(J, h, alpha0=alpha0, beta0=beta0, gamma=gamma, init_scale=init), J, h) <= Er + 1e-5).float().mean())
          for (J, h, _), Er in zip(tune, tref)]
    res.append((np.mean(ps), np.min(ps), alpha0, beta0, gamma, init))
res.sort(reverse=True)
print("top schedules (idealised AOC, T=1000, 100 runs): mean success, worst-instance success, alpha0, beta0, gamma, init")
for r in res[:8]: print("  ", tuple(round(float(v), 3) for v in r))
torch.save(res[0], "qubo_best_ideal.pt")
