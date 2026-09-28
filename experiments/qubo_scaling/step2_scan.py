import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import itertools, torch, numpy as np
from ising import *
from common import make_cell
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
res = torch.load("step2_results.pt", weights_only=False); Js = torch.load("step2_J.pt", weights_only=False)
grid = list(itertools.product([0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.5], [0.1]))
def miss(r, k):
    tol = 0.25 / (2 ** (r["bits"] - 1) - 1); return not bool((r["E"][k] <= r["Eref"] + tol).any())
solvers = {"full twin": lambda J, h, a, b, g: twin_run_paper(TwinIsing(J, h, programming="new"), runs=100, a0=a, beta0=b, gamma=g, field_cal=True),
           "twin no hw": lambda J, h, a, b, g: twin_run_paper(TwinIsing(J, h, programming="new", cell=make_cell(**OFF)), runs=100, a0=a, beta0=b, gamma=g)}
key = {"full twin": "twin_new_fc", "twin no hw": "twin_no_hw"}
for name, fn in solvers.items():
    for i, r in enumerate(res):
        if not miss(r, key[name]): continue
        J = Js[i]; h = torch.zeros(r["N"]); tol = 0.25 / (2 ** (r["bits"] - 1) - 1); used = 0; found = None
        for a0, b0, g0 in grid:
            E = energy(fn(J, h, a0, b0, g0), J, h); hit = E <= r["Eref"] + tol
            if hit.any(): found = (used + int(hit.nonzero()[0]) + 1, a0, b0); break
            used += len(E)
        print(f"{name:10s} #{i:3d} N={r['N']} {r['kind']:6s} {r['bits']}-bit: " +
              (f"found after {found[0]} samples (a0={found[1]}, beta0={found[2]})" if found else "NOT found within 1200 samples"), flush=True)
