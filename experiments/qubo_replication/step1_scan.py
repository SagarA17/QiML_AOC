import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import sys, itertools, torch, numpy as np
from ising import *
from common import make_cell
insts = torch.load("replicate_instances.pt", weights_only=False); h0 = torch.zeros(16)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
grid = list(itertools.product([0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.5], [0.1]))       # 12 schedules
def scan(run_fn, J, Eopt):
    used = 0
    for a0, b0, g0 in grid:
        E = energy(run_fn(J, a0, b0, g0), J, h0); hit = E <= Eopt + 1e-6
        if hit.any(): return used + int(hit.nonzero()[0]) + 1, (a0, b0, g0)
        used += len(E)
    return None, None
solvers = {
  "idealised": lambda J, a, b, g: aoc_paper(J, h0, runs=100, a0=a, beta0=b, gamma=g),
  "twin_no_hw": lambda J, a, b, g: twin_run_paper(TwinIsing(J, h0, programming="new", cell=make_cell(**OFF)), runs=100, a0=a, beta0=b, gamma=g),
  "twin_new+fieldcal": lambda J, a, b, g: twin_run_paper(TwinIsing(J, h0, programming="new"), runs=100, a0=a, beta0=b, gamma=g, field_cal=True),
}
targets = {k: [int(x) for x in v.split(",")] for k, v in (a.split("=") for a in sys.argv[1:])}
for k, ids in targets.items():
    for i in ids:
        kind, bits, J, Eopt = insts[i]
        n, sch = scan(solvers[k], J, Eopt)
        print(f"{k:18s} inst {i:3d} ({kind}, {bits}-bit): " + (f"optimum found after {n} samples with a0={sch[0]}, beta0={sch[1]}" if n else "NOT found within 1200 samples"), flush=True)
