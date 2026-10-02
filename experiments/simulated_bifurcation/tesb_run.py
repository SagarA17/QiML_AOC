import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
"""TE-dSB variants on the hard Wishart set (M/N = 0.5, N = 32/44). Readout window 40, as for the dSB baselines.
  TE_ideal          paper equations: per-iteration shared mini-batch + pump shift
  TE_ideal_noshift  authors' released code: per-iteration shared mini-batch, no pump shift
  TE_hw_ideal       per-sample mini-batch, no pump shift (static per-run injection)
  TE_hw_twin        same, on the full twin
  TE_iter_twin      per-iteration mini-batch on the full twin (needs per-iteration injection updates)"""
import os, time, torch
from ising import energy, TwinIsing
from sb_tabu import tesb_ideal, tesb_twin
res = torch.load("step25_results.pt", weights_only=False)
idx = [i for i, r in enumerate(res) if r["MN"] == 0.5 and r["N"] in (32, 44)]
f = "tesb_results.pt"; out = torch.load(f, weights_only=False) if os.path.exists(f) else {}
runs = {
    "TE_ideal": lambda J, h: tesb_ideal(J, h, beta=0.1, minibatch="iteration", pump_shift=True, window=40),
    "TE_ideal_noshift": lambda J, h: tesb_ideal(J, h, beta=0.1, minibatch="iteration", pump_shift=False, window=40),
    "TE_hw_ideal": lambda J, h: tesb_ideal(J, h, beta=0.05, minibatch="sample", pump_shift=False, window=40),
    "TE_hw_twin": lambda J, h: tesb_twin(TwinIsing(J, h, programming="new"), beta=0.05, minibatch="sample", window=40),
    "TE_iter_twin": lambda J, h: tesb_twin(TwinIsing(J, h, programming="new"), beta=0.1, minibatch="iteration", window=40),
}
t0 = time.time()
for n, i in enumerate(idx):
    if i in out: continue
    r = res[i]; J = r["J"]; h = torch.zeros(r["N"]); rec = {}
    for name, fn in runs.items():
        s, M = fn(J, h)
        rec[name] = energy(s, J, h); rec[name + "_ntabu"] = len(M)
    out[i] = rec; torch.save(out, f)
    print(f"[{time.time()-t0:5.0f}s] {n+1}/{len(idx)}", flush=True)
print("done", flush=True)
