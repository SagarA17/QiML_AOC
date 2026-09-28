"""Rebuild the step-2 coupling matrices by replaying the deterministic generator; match accepted
instances by bit precision + eigenvector filter + a quick SA search reproducing the saved reference energy."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import torch, numpy as np
from ising import energy, simulated_annealing
import qubo_instances as rq
res = torch.load("step2_results.pt", weights_only=False)
rng = np.random.default_rng(4242); count = 0; Js = []
for N in [24, 32, 44]:
    for kind in ["dense", "sparse"]:
        made = 0
        while made < 20:
            bits = int(rng.integers(3, 9)); J = rq.make_instance(kind, N, bits, rng); seed_sa = int(rng.integers(1 << 30))
            r = res[count]; tol = 0.25 / (2 ** (bits - 1) - 1); h = torch.zeros(N)
            if bits != r["bits"]: continue
            v = torch.linalg.eigh(J).eigenvectors[:, -1]; sv = torch.sign(v) + (v == 0).float()
            if float(energy(sv, J, h)) <= r["Eref"] + tol: continue
            Emin = float(energy(simulated_annealing(J, h, runs=300, sweeps=1000, seed=1), J, h).min())
            if abs(Emin - r["Eref"]) > tol and Emin < r["Eref"] - tol: continue      # cannot beat a true reference
            if abs(Emin - r["Eref"]) > 0.05 * abs(r["Eref"]): continue               # clearly a different instance
            Js.append(J); made += 1; count += 1
            print(f"matched #{count-1} N={N} {kind} ({bits}-bit): quick-SA {Emin:.4f} vs saved ref {r['Eref']:.4f}", flush=True)
torch.save(Js, "step2_J.pt"); print("done", flush=True)
