import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps
Xtr, ytr, Xte, yte = load_data(); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt"))
cores, r = gauge_normalise(mps, Ptr)
eps = choose_eps(cores, Ptr, 0.02)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
for label, flags in [("ideal loop", OFF), ("only weight distortion", {**OFF, "apply_weight_distortion": True}), ("full twin", {})]:
    m = AOCMPS(cores, r, eps, make_cell(**flags)); m.exact_dc = True
    x = Xtr[:1]
    W = m.build_W(); m.cell.matrix = W
    c, b, _ = m.injection(x, W)
    z = torch.clip(c, -1.3, 1.3); hist = []
    for t in range(400):
        zn = m.cell(z, x=c); hist.append(float((zn - z).abs().max())); z = zn
    # Jacobian at the fixed point
    J = torch.autograd.functional.jacobian(lambda zz: m.cell(zz.unsqueeze(0), x=c)[0], z[0].detach())
    ev = torch.linalg.eigvals(J[:44, :44]).abs()
    print(f"{label:24s} max|dz| @t=50,100,200,400: {hist[49]:.1e} {hist[99]:.1e} {hist[199]:.1e} {hist[399]:.1e} | spectral radius of loop Jacobian {float(ev.max()):.3f}")
