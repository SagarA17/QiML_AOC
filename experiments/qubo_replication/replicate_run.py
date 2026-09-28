import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time, torch, numpy as np
from ising import *
from common import make_cell
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
insts = torch.load("replicate_instances.pt"); sch = torch.load("replicate_schedules.pt", weights_only=False)
KI = dict(a0=sch["ideal"][2], beta0=sch["ideal"][3], gamma=sch["ideal"][4])
KT = dict(a0=sch["twin"][2], beta0=sch["twin"][3], gamma=sch["twin"][4])
h0 = torch.zeros(16); R = 1000
solvers = {
    "idealised": lambda J: aoc_paper(J, h0, runs=R, **KI),
    "twin_no_hw": lambda J: twin_run_paper(TwinIsing(J, h0, programming="new", cell=make_cell(**OFF)), runs=R, **KT),
    "twin_old": lambda J: twin_run_paper(TwinIsing(J, h0, programming="old"), runs=R, **KT),
    "twin_new": lambda J: twin_run_paper(TwinIsing(J, h0, programming="new"), runs=R, **KT),
}
import os
out = torch.load("replicate_results.pt", weights_only=False) if os.path.exists("replicate_results.pt") else {k: [] for k in solvers}
start = min(len(v) for v in out.values()); out = {k: v[:start] for k, v in out.items()}
print(f"resuming at instance {start}", flush=True); t0 = time.time()
for n, (kind, bits, J, Eopt) in enumerate(insts):
    if n < start: continue
    for k, fn in solvers.items():
        E = energy(fn(J), J, h0)
        hit = (E <= Eopt + 1e-6)
        first = int(hit.nonzero()[0]) + 1 if hit.any() else None       # samples until first optimum
        prox = float(E.min() / Eopt)                                    # best-of-1000 objective proximity
        out[k].append(dict(kind=kind, bits=bits, p=float(hit.float().mean()), first=first, prox=prox))
    if n % 10 == 9:
        torch.save(out, "replicate_results.pt")
        print(f"{n+1}/100 instances done ({time.time()-t0:.0f}s)", flush=True)
torch.save(out, "replicate_results.pt"); print("done")
