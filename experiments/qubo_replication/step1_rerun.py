import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import os, time, torch, numpy as np
from ising import *
insts = torch.load("replicate_instances.pt", weights_only=False); sch = torch.load("replicate_schedules.pt", weights_only=False)
KT = dict(a0=sch["twin"][2], beta0=sch["twin"][3], gamma=sch["twin"][4]); h0 = torch.zeros(16)
f = "step1_fieldcal.pt"; out = torch.load(f, weights_only=False) if os.path.exists(f) else []
t0 = time.time()
for n, (kind, bits, J, Eopt) in enumerate(insts):
    if n < len(out): continue
    E = energy(twin_run_paper(TwinIsing(J, h0, programming="new"), runs=1000, field_cal=True, **KT), J, h0)
    hit = E <= Eopt + 1e-6
    out.append(dict(kind=kind, bits=bits, p=float(hit.float().mean()), first=int(hit.nonzero()[0]) + 1 if hit.any() else None, prox=float(E.min() / Eopt)))
    if n % 10 == 9: torch.save(out, f); print(f"{n+1}/100 ({time.time()-t0:.0f}s)", flush=True)
torch.save(out, f); print("done", flush=True)
