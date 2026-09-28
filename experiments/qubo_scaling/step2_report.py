import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
res = torch.load("step2_results.pt", weights_only=False)
names = [("idealised", "idealised algorithm"), ("twin_no_hw", "twin, no hw effects"), ("twin_new_fc", "full twin (new prog + field cal)"), ("SA", "simulated annealing")]
print("20 dense + 20 sparse instances per size; 1000 samples each; reference exact for N<=32, best-known (any solver) for N=44\n")
print(f"{'N':>3s} {'kind':6s} {'solver':33s} | solved | samples to opt: median / max | succ/run | worst proximity")
for N in [24, 32, 44]:
    for kind in ["dense", "sparse"]:
        rs = [r for r in res if r["N"] == N and r["kind"] == kind]
        if not rs: continue
        for k, lbl in names:
            firsts, ps, prox = [], [], []
            for r in rs:
                tol = 0.25 / (2 ** (r["bits"] - 1) - 1)          # quarter of the energy quantum
                hit = r["E"][k] <= r["Eref"] + tol
                if hit.any(): firsts.append(int(hit.nonzero()[0]) + 1)
                ps.append(float(hit.float().mean())); prox.append(float(r["E"][k].min() / r["Eref"]))
            f = np.array(firsts) if firsts else np.array([np.nan])
            print(f"{N:3d} {kind:6s} {lbl:33s} | {len(firsts):2d}/{len(rs):<2d}  | {np.nanmedian(f):8.0f} / {np.nanmax(f):5.0f}       | {np.mean(ps):8.3f} | {min(prox)*100:6.1f}%")
        print()
