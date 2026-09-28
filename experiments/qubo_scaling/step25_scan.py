import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import itertools, torch, numpy as np
from ising import *
from common import make_cell
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
res = torch.load("step25_results.pt", weights_only=False)
grid = list(itertools.product([0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.5], [0.1]))
solvers = {"full twin": ("twin_new_fc", lambda J, h, a, b, g: twin_run_paper(TwinIsing(J, h, programming="new"), runs=100, a0=a, beta0=b, gamma=g, field_cal=True)),
           "idealised": ("idealised", lambda J, h, a, b, g: aoc_paper(J, h, runs=100, a0=a, beta0=b, gamma=g))}
out = {}
for name, (key, fn) in solvers.items():
    for i, r in enumerate(res):
        if bool((r["E"][key] <= r["Eref"] + 1e-4).any()): continue
        J = r["J"]; h = torch.zeros(r["N"]); used = 0; found = None
        for a0, b0, g0 in grid:
            E = energy(fn(J, h, a0, b0, g0), J, h); hit = E <= r["Eref"] + 1e-4
            if hit.any(): found = used + int(hit.nonzero()[0]) + 1; break
            used += len(E)
        out[(name, i)] = found
        print(f"{name:10s} #{i:2d} M/N={r['MN']} N={r['N']}: " + (f"found after {found} samples" if found else "NOT found within 1200"), flush=True)
torch.save(out, "step25_scan.pt"); print("done", flush=True)
