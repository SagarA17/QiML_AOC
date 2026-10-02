import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
"""Compare TE-dSB variants with plain dSB and SA on the hard Wishart set (M/N = 0.5, N = 32/44).
Cell: instances solved at least once / total (mean per-run success probability)."""
import torch, numpy as np
res = torch.load("step25_results.pt", weights_only=False); sb = torch.load("sb_results.pt", weights_only=False)
te = torch.load("tesb_results.pt", weights_only=False)
cols = [("dSB ideal", lambda i: sb[i]["dSB_ideal"])] + \
       [(k, lambda i, k=k: te[i][k]) for k in ("TE_ideal", "TE_ideal_noshift", "TE_hw_ideal")] + \
       [("dSB twin", lambda i: sb[i]["dSB_twin_g095"])] + \
       [(k, lambda i, k=k: te[i][k]) for k in ("TE_hw_twin", "TE_iter_twin")] + \
       [("SA", lambda i: res[i]["E"]["SA"])]
print(f"{'N':>3s} | " + " | ".join(f"{c:>17s}" for c, _ in cols))
for N in (32, 44):
    ids = [i for i in te if res[i]["N"] == N]; cells = []
    for _, get in cols:
        hits = [get(i) <= res[i]["Eref"] + 1e-4 for i in ids]
        cells.append(f"{sum(bool(x.any()) for x in hits):2d}/{len(ids)} ({np.mean([float(x.float().mean()) for x in hits]):.4f})")
    print(f"{N:3d} | " + " | ".join(f"{c:>17s}" for c in cells))
nt = [te[i]["TE_hw_ideal_ntabu"] for i in te]
print(f"tabu list size (ideal warm-up, of 100): min {min(nt)}, median {int(np.median(nt))}, max {max(nt)}")
