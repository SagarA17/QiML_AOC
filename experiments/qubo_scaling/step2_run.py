"""Step 2 (resumable): generate paper-family instance -> reference -> run all solvers -> save. One instance at a time."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import os, time, math, torch, numpy as np
exec(open(_os.path.join(_ROOT, "experiments", "qubo_scaling", "step2_gen.py")).read().split("rng = np.random.default_rng(4242)")[0])      # exact_mitm (no generation)
from ising import *
from common import make_cell
import qubo_instances as rq
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
sch = torch.load("replicate_schedules.pt", weights_only=False)
KI = dict(a0=sch["ideal"][2], beta0=sch["ideal"][3], gamma=sch["ideal"][4]); KT = dict(a0=sch["twin"][2], beta0=sch["twin"][3], gamma=sch["twin"][4])
R = 1000; PER = 20
f = "step2_results.pt"; res = torch.load(f, weights_only=False) if os.path.exists(f) else []
rng = np.random.default_rng(4242); t0 = time.time(); count = 0
for N in [24, 32, 44]:
    for kind in ["dense", "sparse"]:
        made = 0
        while made < PER:
            bits = int(rng.integers(3, 9)); J = rq.make_instance(kind, N, bits, rng); h = torch.zeros(N)
            seed_sa = int(rng.integers(1 << 30))
            if count < len(res):                       # already done (generator is deterministic): skip work
                if res[count]["N"] == N and res[count]["kind"] == kind and res[count]["bits"] == bits:
                    made += 1; count += 1; continue
            Eref = exact_mitm(J) if N <= 32 else None
            runs = {"idealised": aoc_paper(J, h, runs=R, **KI),
                    "twin_no_hw": twin_run_paper(TwinIsing(J, h, programming="new", cell=make_cell(**OFF)), runs=R, **KT),
                    "twin_new_fc": twin_run_paper(TwinIsing(J, h, programming="new"), runs=R, field_cal=True, **KT),
                    "SA": simulated_annealing(J, h, runs=R, sweeps=1000, seed=seed_sa)}
            E = {k: energy(v, J, h) for k, v in runs.items()}
            if Eref is None: Eref = min(float(e.min()) for e in E.values())    # best known (N = 44)
            v = torch.linalg.eigh(J).eigenvectors[:, -1]; sv = torch.sign(v) + (v == 0).float()
            if float(energy(sv, J, h)) <= Eref + 1e-6: continue                 # eigenvector filter
            rec = dict(N=N, kind=kind, bits=bits, Eref=Eref, exact=N <= 32, E={k: e for k, e in E.items()})
            res.append(rec); made += 1; count += 1
            torch.save(res, f)
            print(f"[{time.time()-t0:5.0f}s] N={N} {kind} {made}/{PER}", flush=True)
print("done", flush=True)
