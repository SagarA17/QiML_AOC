import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
res = torch.load("step25_results.pt", weights_only=False); sb = torch.load("sb_results.pt", weights_only=False)
cols = [("idealised", "AOC-alg ideal", "old"), ("twin_new_fc", "AOC-alg twin", "old"), ("dSB_ideal", "dSB ideal", "sb"),
        ("dSB_twin_g1", "dSB twin g=1", "sb"), ("dSB_twin_g095", "dSB twin g=.95", "sb"), ("SA", "SA", "old")]
print(f"{len(sb)} Wishart instances with dSB results; cell = solved/15 within 1000 samples (success/run)\n")
print(f"{'M/N':>4s} {'N':>3s} | " + " | ".join(f"{c[1]:>15s}" for c in cols))
for MN in [0.75, 0.5]:
    for N in [24, 32, 44]:
        idx = [i for i, r in enumerate(res[:len(sb)]) if r["MN"] == MN and r["N"] == N]
        if not idx: continue
        cells = []
        for k, lbl, src in cols:
            hits = [(res[i]["E"][k] if src == "old" else sb[i][k]) <= res[i]["Eref"] + 1e-4 for i in idx]
            cells.append(f"{sum(bool(x.any()) for x in hits):2d}/{len(idx)} ({np.mean([float(x.float().mean()) for x in hits]):.3f})")
        print(f"{MN:4.2f} {N:3d} | " + " | ".join(f"{c:>15s}" for c in cells))
