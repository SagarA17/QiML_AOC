import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time, torch, numpy as np
from ising import *
from common import make_cell
st = torch.load("qubo_stage1.pt"); fam, refs = st["fam"], st["refs"]
IDEAL = dict(a0=0.6, beta0=0.2, gamma=0.1)      # tuned on the idealised algorithm
TWIN = dict(a0=0.6, beta0=0.1, gamma=0.0)       # tuned on the twin's loop dynamics
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
def stats(s, J, h, Er):
    E = energy(s, J, h); return float((E <= Er + 1e-5).float().mean()), float(((E - Er) / abs(Er)).mean())
cols = [("idealised algorithm", None), ("twin, no hw effects", ("new", OFF)), ("twin full, OLD prog.", ("old", {})), ("twin full, NEW prog.", ("new", {}))]
print("100 runs x 1000 iterations per instance; 10 held-out test instances per row; cell = success/run, instances solved, mean energy gap")
print(f"{'':12s} | " + " | ".join(f"{c[0]:>24s}" for c in cols))
t0 = time.time(); summary = {}
for key in [("wishart", 16), ("wishart", 24), ("wishart", 32), ("wishart", 44), ("3reg", 16), ("3reg", 24)]:
    cells = []
    for name, spec in cols:
        r = []
        for (J, h, _), Er in zip(fam[key], refs[key]):
            if spec is None: s = aoc_paper(J, h, **IDEAL)
            else: s = twin_run_paper(TwinIsing(J, h, programming=spec[0], cell=make_cell(**spec[1])), **TWIN)
            r.append(stats(s, J, h, Er))
        p = np.array([x[0] for x in r]); g = np.array([x[1] for x in r]); summary[(key, name)] = (p, g)
        cells.append(f"{p.mean():5.3f} {int((p>0).sum()):2d}/10 {g.mean()*100:5.1f}%")
    print(f"{key[0]:>8s} {key[1]:3d} | " + " | ".join(f"{c:>24s}" for c in cells), flush=True)
print(f"({time.time()-t0:.0f}s)")
torch.save(summary, "qubo_final.pt")
