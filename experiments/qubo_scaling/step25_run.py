"""Step 2.5 (resumable): rugged Wishart planted instances at N = 24, 32, 44; same solvers & protocol as step 2."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import os, time, torch, numpy as np
from ising import *
from common import make_cell
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
sch = torch.load("replicate_schedules.pt", weights_only=False)
KI = dict(a0=sch["ideal"][2], beta0=sch["ideal"][3], gamma=sch["ideal"][4]); KT = dict(a0=sch["twin"][2], beta0=sch["twin"][3], gamma=sch["twin"][4])
R, PER = 1000, 15
f = "step25_results.pt"; res = torch.load(f, weights_only=False) if os.path.exists(f) else []
t0 = time.time(); n = 0
for MN in [0.75, 0.5]:
    for N in [24, 32, 44]:
        for i in range(PER):
            if n < len(res): n += 1; continue
            J, h, t = wishart_planted(N, M_over_N=MN, seed=5000 + 100 * N + i + int(MN * 1000))
            Eref = float(energy(t, J, h))
            runs = {"idealised": aoc_paper(J, h, runs=R, **KI),
                    "twin_no_hw": twin_run_paper(TwinIsing(J, h, programming="new", cell=make_cell(**OFF)), runs=R, **KT),
                    "twin_new_fc": twin_run_paper(TwinIsing(J, h, programming="new"), runs=R, field_cal=True, **KT),
                    "SA": simulated_annealing(J, h, runs=R, sweeps=1000, seed=i)}
            E = {k: energy(v, J, h) for k, v in runs.items()}
            assert all(float(e.min()) >= Eref - 1e-4 for e in E.values()), "found below planted optimum?!"
            res.append(dict(MN=MN, N=N, J=J, Eref=Eref, E=E)); n += 1; torch.save(res, f)
            print(f"[{time.time()-t0:5.0f}s] M/N={MN} N={N} {i+1}/{PER}", flush=True)
print("done", flush=True)
