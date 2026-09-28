import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps
Xtr, ytr, Xte, yte = load_data()
set_feature_mode("complement"); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt")); cores, r = gauge_normalise(mps, Ptr)
set_feature_mode("balanced")
b = biases(Xtr)
tn = F_neg(b[...,0]) + F_neg(b[...,1])
print(f"negative-LED-path per-site total: event-to-event std {float(tn.std(0).mean())*1e3:.3f} mV (pos path is exactly 0)")
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
for label, flags in [("ideal loop", OFF)] + [(f"only {n}", {**OFF, n: True, **({"apply_pd_crosstalk": True} if n=="apply_slm_darkness" else {})})
                     for n in ["apply_input_efficiencies","apply_output_efficiencies","apply_pd_crosstalk","apply_slm_darkness","apply_weight_distortion"]] + [("full twin", {})]:
    m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), make_cell(**flags))
    W = m.build_W(); m.cell.matrix = W
    with torch.no_grad():
        bop, _ = m.operating_points(Xtr)
        G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
    print(f"  {label:32s} G(b(x)) event-to-event std: mean {float(G[:, :44].std(0).mean())*1e3:6.3f} mV, max {float(G[:, :44].std(0).max())*1e3:6.3f} mV")
