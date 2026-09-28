import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps, site_efficiency_ratios
Xtr, ytr, Xte, yte = load_data()
set_feature_mode("complement"); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt")); cores, r = gauge_normalise(mps, Ptr)
cell = make_cell()
for idle in [[44, 45, 46, 47], [8, 1, 46, 3]]:
    cm.PLACEMENT["idle"] = idle
    used = cm.layout()["used_ch"]
    rho = site_efficiency_ratios(cell)
    print(f"\nidle channels {idle}; per-site rho {[round(float(v),3) for v in rho]}")
    for label, mode, rh, req in [("complement features", "complement", None, False), ("seesaw", "balanced", None, False),
                                 ("seesaw + row equalisation", "balanced", None, True),
                                 ("efficiency-aware seesaw + row equalisation", "balanced", rho, True)]:
        set_feature_mode(mode, rh)
        m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), cell); m.row_equalise = req
        W = m.build_W(); m.cell.matrix = W
        with torch.no_grad():
            bop, _ = m.operating_points(Xtr)
            G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
        sd = G[:, used].std(0)
        print(f"  {label:44s} DC event-to-event std: mean {float(sd.mean())*1e3:6.3f} mV, max {float(sd.max())*1e3:6.3f} mV")
print("\nwith efficiency matching (idle [8, 1, 46, 3]):")
cm.PLACEMENT["idle"] = [8, 1, 46, 3]; used = cm.layout()["used_ch"]; rho = site_efficiency_ratios(cell)
for flags_off, lbl in [({}, "full twin"), (dict(apply_weight_distortion=False), "full twin minus weight distortion (no floor)")]:
    set_feature_mode("balanced", rho)
    cellx = make_cell(**flags_off)
    m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), cellx); m.row_equalise = True; m.eff_match = True
    W = m.build_W(); m.cell.matrix = W
    with torch.no_grad():
        bop, _ = m.operating_points(Xtr); G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
    sd = G[:, used].std(0)
    print(f"  {lbl:50s} DC event-to-event std: mean {float(sd.mean())*1e3:6.3f} mV, max {float(sd.max())*1e3:6.3f} mV")
