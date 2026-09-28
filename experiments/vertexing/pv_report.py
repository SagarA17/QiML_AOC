import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from sklearn.metrics import adjusted_rand_score as ari
out = torch.load("pv_results.pt", weights_only=False)
print(f"{len(out)} windows; 11 tracks x 4 slots (44 vars); 2-4 true vertices\n")
print(f"{'method':30s} {'budget':>6s} | {'QUBO optimum hit':>16s} | {'valid':>6s} | {'mean ARI vs truth':>17s} | {'perfect windows':>15s}")
ex = [ari(r["w"]["lab"], r["a_exact"].numpy()) for r in out]
print(f"{'exact QUBO optimum (ceiling)':30s} {'-':>6s} | {'-':>16s} | {'-':>6s} | {np.mean(ex):17.3f} | {sum(a > 0.999 for a in ex):9d}/{len(out)}")
da = [ari(r["w"]["lab"], r["DA"].numpy()) for r in out]
print(f"{'deterministic annealing (DA)':30s} {'-':>6s} | {'-':>16s} | {'-':>6s} | {np.mean(da):17.3f} | {sum(a > 0.999 for a in da):9d}/{len(out)}")
for k, lbl in [("dSB_twin", "dSB, full twin"), ("dSB_ideal", "dSB, idealised"), ("SA", "simulated annealing")]:
    for b in [1, 10, 100]:
        hit = val = 0; A = []
        for r in out:
            E, As = r["sol"][k]["E"][:b], r["sol"][k]["A"][:b]
            i = int(E.argmin()); a = As[i]
            hit += float(E[i]) <= r["Eref"] + 1e-3 * abs(r["Eref"])
            ok = bool((a >= 0).all()); val += ok
            A.append(ari(r["w"]["lab"], a.numpy()) if ok else 0.0)
        print(f"{lbl:30s} {b:6d} | {hit:10d}/{len(out):<5d} | {val:3d}/{len(out):<2d} | {np.mean(A):17.3f} | {sum(x > 0.999 for x in A):9d}/{len(out)}")
