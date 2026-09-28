import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from ising import *
for mn in [0.25, 0.5, 0.75, 1.5]:
    tune = [wishart_planted(32, M_over_N=mn, seed=s) for s in range(5)]
    tref = [float(energy(t, J, h)) for J, h, t in tune]
    sa = [float((energy(simulated_annealing(J, h, runs=100, sweeps=1000), J, h) <= Er + 1e-5).float().mean()) for (J,h,_),Er in zip(tune,tref)]
    aoc = [float((energy(aoc_ideal(J, h, alpha0=0.5, beta0=0.5, T=1000), J, h) <= Er + 1e-5).float().mean()) for (J,h,_),Er in zip(tune,tref)]
    aoc5 = [float((energy(aoc_ideal(J, h, alpha0=1.0, beta0=0.75, gamma=0.5, T=5000), J, h) <= Er + 1e-5).float().mean()) for (J,h,_),Er in zip(tune,tref)]
    print(f"Wishart N=32, M/N={mn}: SA(1000 sweeps) per-instance {np.round(sa,2)} | AOC-ideal T=1000 {np.round(aoc,2)} | T=5000 {np.round(aoc5,2)}")
