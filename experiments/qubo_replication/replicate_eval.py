import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time, itertools, torch, numpy as np
from ising import *
from common import make_cell
import qubo_instances as rq   # re-uses generator (module-level code regenerates identically; harmless)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
insts = torch.load("replicate_instances.pt")
# ---- separate tuning instances
rng = np.random.default_rng(7); tune = []
for kind in ["dense", "sparse"]:
    while sum(1 for t in tune if t[0] == kind) < 5:
        bits = int(rng.integers(3, 9)); J = rq.make_instance(kind, 16, bits, rng); E, s = rq.ground_state(J)
        if not rq.eig_trivial(J, s, E): tune.append((kind, bits, J, E))
h0 = torch.zeros(16)
def p_ideal(J, E, **kw): return float((energy(aoc_paper(J, h0, runs=100, **kw), J, h0) <= E + 1e-6).float().mean())
def p_twin(J, E, flags, **kw):
    tw = TwinIsing(J, h0, programming="new", cell=make_cell(**flags))
    return float((energy(twin_run_paper(tw, runs=100, **kw), J, h0) <= E + 1e-6).float().mean())
grid = list(itertools.product([0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.5], [0.0, 0.1]))
def tune_best(fn):
    res = []
    for a0, b0, g0 in grid:
        ps = [fn(J, E, a0=a0, beta0=b0, gamma=g0) for _, _, J, E in tune]
        res.append((np.mean(ps), np.min(ps), a0, b0, g0))
    res.sort(key=lambda r: -(r[0] if r[1] > 0 else r[0] - 1)); return res[0]
t0 = time.time()
bi = tune_best(p_ideal); print(f"tuned idealised: mean {bi[0]:.3f} worst {bi[1]:.3f}  a0={bi[2]} beta0={bi[3]} gamma={bi[4]}  ({time.time()-t0:.0f}s)")
bt = tune_best(lambda J, E, **kw: p_twin(J, E, OFF, **kw)); print(f"tuned twin loop: mean {bt[0]:.3f} worst {bt[1]:.3f}  a0={bt[2]} beta0={bt[3]} gamma={bt[4]}  ({time.time()-t0:.0f}s)")
torch.save({"ideal": bi, "twin": bt}, "replicate_schedules.pt")
