import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps, site_efficiency_ratios
Xtr, ytr, Xte, yte = load_data()
set_feature_mode("complement"); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt")); cores, r = gauge_normalise(mps, Ptr)
cell = make_cell()
ie = cell.hardware_parameters.input_efficiency
print(f"input efficiencies: pos {float(ie[0].min()):.3f}-{float(ie[0].max()):.3f}, neg {float(ie[1].min()):.3f}-{float(ie[1].max()):.3f}")
rho = site_efficiency_ratios(cell); print("per-site group efficiency ratio rho:", [round(float(v),3) for v in rho])
for label, mode, rh, req in [("complement features", "complement", None, False), ("seesaw", "balanced", None, False),
                             ("seesaw + row equalisation", "balanced", None, True),
                             ("efficiency-aware seesaw + row equalisation", "balanced", rho, True)]:
    set_feature_mode(mode, rh)
    m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), cell); m.row_equalise = req
    W = m.build_W(); m.cell.matrix = W
    with torch.no_grad():
        bop, _ = m.operating_points(Xtr)
        G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
    print(f"  {label:44s} G(b(x)) event-to-event std: mean {float(G[:, :44].std(0).mean())*1e3:6.3f} mV, max {float(G[:, :44].std(0).max())*1e3:6.3f} mV")
