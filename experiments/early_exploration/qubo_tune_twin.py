import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np, itertools, time
from ising import *
from common import make_cell
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
tune = [wishart_planted(32, seed=s) for s in range(5)]; tref = [float(energy(t, J, h)) for J, h, t in tune]
tws = [TwinIsing(J, h, programming="new", cell=make_cell(**OFF)) for J, h, _ in tune]
t0 = time.time(); res = []
for a0, beta0, gamma, init in itertools.product([0.4, 0.6, 0.8], [0.02, 0.05, 0.1], [0.0], [None]):
    ps = [float((energy(twin_run_paper(tw, a0=a0, beta0=beta0, gamma=gamma, init=init), J, h) <= Er + 1e-5).float().mean())
          for tw, (J, h, _), Er in zip(tws, tune, tref)]
    res.append((np.mean(ps), np.min(ps), a0, beta0, gamma, init if init else 0.0))
res.sort(key=lambda r: -(r[0] if r[1] > 0 else r[0] - 1))   # prefer schedules that never collapse (worst > 0)
print(f"twin-dynamics tuning (ideal loop, Wishart N=32 tuning set; {time.time()-t0:.0f}s). mean succ, worst, a0, beta0, gamma, init amp [V] (0 = wall/sqrt(N)):")
for r in res[:8]: print("  ", tuple(round(float(v), 3) for v in r))
torch.save(res[0], "qubo_best_twin.pt")
