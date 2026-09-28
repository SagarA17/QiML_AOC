import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import os, time, torch, numpy as np
from pv_toy import *
from ising import energy, simulated_annealing, TwinIsing
from sb import sb_ideal, sb_twin
R = 100
f = "pv_results.pt"; out = torch.load(f, weights_only=False) if os.path.exists(f) else []
rng = np.random.default_rng(2026); t0 = time.time()
for n in range(100):
    w = make_window(rng)
    if n < len(out): continue
    D = pair_matrix(w); Q, q, lam = build_qubo(D); J, h = qubo_to_ising(Q, q)
    Eex, aex = exact_assignment(D)
    xopt = torch.zeros(44); xopt[[idx(i, int(aex[i])) for i in range(N_TRK)]] = 1
    Eref = float(energy(2 * xopt - 1, J, h))
    rec = dict(w=w, a_exact=aex, Eref=Eref, sol={})
    # twin: linear terms folded in via a reference spin (45 channels)
    Ja = with_reference_spin(J, h)
    sa_ = sb_twin(TwinIsing(Ja, torch.zeros(45), idle=(8, 1, 46), programming="new"), runs=R, gamma=0.95)
    s_tw = sa_[:, :44] * sa_[:, 44:45]
    sols = {"dSB_twin": s_tw, "dSB_ideal": sb_ideal(J, h, runs=R, variant="dSB"),
            "SA": simulated_annealing(J, h, runs=R, sweeps=1000, seed=n)}
    for k, s in sols.items():
        rec["sol"][k] = dict(E=energy(s, J, h), A=decode(s))
    rec["DA"] = torch.tensor(deterministic_annealing(w))
    out.append(rec); torch.save(out, f)
    if n % 10 == 9: print(f"[{time.time()-t0:4.0f}s] {n+1}/100", flush=True)
print("done", flush=True)
