import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from ising import *
from common import make_cell
st = torch.load("qubo_stage1.pt"); fam, refs = st["fam"], st["refs"]
A0, B0, G0 = 0.6, 0.2, 0.1
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
def succ(s, J, h, Er): return float((energy(s, J, h) <= Er + 1e-5).float().mean())
print("mean success/run over 10 test instances (NEW programming)")
for key in [("wishart", 32), ("3reg", 24)]:
    out = {}
    for lbl, flags in [("twin, ideal loop (no hardware effects)", OFF), ("twin, full", {})]:
        ps = []
        for (J, h, _), Er in zip(fam[key], refs[key]):
            tw = TwinIsing(J, h, programming="new", cell=make_cell(**flags))
            ps.append(succ(twin_run_paper(tw, a0=A0, beta0=B0, gamma=G0), J, h, Er))
        out[lbl] = np.mean(ps)
    ideal = np.mean([succ(aoc_paper(J, h, a0=A0, beta0=B0, gamma=G0), J, h, Er) for (J, h, _), Er in zip(fam[key], refs[key])])
    print(f"  {key}: idealised algorithm {ideal:.3f} | " + " | ".join(f"{k} {v:.3f}" for k, v in out.items()))
