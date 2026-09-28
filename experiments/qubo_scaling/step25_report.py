import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
res = torch.load("step25_results.pt", weights_only=False)
names = [("idealised", "idealised algorithm"), ("twin_no_hw", "twin, no hw effects"), ("twin_new_fc", "full twin (new prog + field cal)"), ("SA", "simulated annealing")]
print("Wishart planted (exact optimum known); 15 instances per row; 1000 samples each; hit tolerance 1e-4\n")
print(f"{'M/N':>4s} {'N':>3s} {'solver':33s} | solved | samples to opt: median / max | succ/run | worst proximity")
for MN in [0.75, 0.5]:
    for N in [24, 32, 44]:
        rs = [r for r in res if r["MN"] == MN and r["N"] == N]
        if not rs: continue
        for k, lbl in names:
            firsts, ps, prox = [], [], []
            for r in rs:
                hit = r["E"][k] <= r["Eref"] + 1e-4
                if hit.any(): firsts.append(int(hit.nonzero()[0]) + 1)
                ps.append(float(hit.float().mean())); prox.append(float(r["E"][k].min() / r["Eref"]))
            f = np.array(firsts, float) if firsts else np.array([np.nan])
            print(f"{MN:4.2f} {N:3d} {lbl:33s} | {len(firsts):2d}/{len(rs):<2d}  | {np.nanmedian(f):8.0f} / {np.nanmax(f):5.0f}       | {np.mean(ps):8.3f} | {min(prox)*100:6.1f}%")
        print()
