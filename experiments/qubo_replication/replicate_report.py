import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
out = torch.load("replicate_results.pt", weights_only=False)
names = {"idealised": "idealised algorithm", "twin_no_hw": "twin, no hw effects", "twin_old": "twin full, OLD prog.", "twin_new": "twin full, NEW prog."}
n = len(out["idealised"]); print(f"{n} instances, 1000 samples each (16 variables, 3-8 bit, eigenvector-filtered)\n")
print(f"{'solver':24s} {'kind':7s} | {'solved<=1000':>12s} | {'samples to optimum: median / 90% / max':>40s} | {'succ/run':>8s} | {'worst best-of-1000 proximity':>28s}")
for k, lbl in names.items():
    for kind in ["dense", "sparse"]:
        rs = [r for r in out[k] if r["kind"] == kind]
        firsts = [r["first"] for r in rs if r["first"] is not None]
        solved = len(firsts)
        f = np.array(firsts) if firsts else np.array([np.nan])
        print(f"{lbl:24s} {kind:7s} | {solved:5d}/{len(rs):<6d} | {np.median(f):12.0f} / {np.percentile(f,90):8.0f} / {np.max(f):8.0f}      | "
              f"{np.mean([r['p'] for r in rs]):8.3f} | {min(r['prox'] for r in rs)*100:27.1f}%")
