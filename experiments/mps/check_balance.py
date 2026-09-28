import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps
Xtr, ytr, Xte, yte = load_data()
x = Xtr
for mode in ["complement", "balanced"]:
    set_feature_mode(mode)
    b = biases(x)
    tot = (F_pos(b[..., 0]) + F_pos(b[..., 1]))                   # per-site total light (pos path)
    print(f"[{mode}] per-site total LED level: mean {float(tot.mean())*1e3:.2f} mV, event-to-event std {float(tot.std(0).mean())*1e3:.4f} mV")
# loop's zero-signal DC response G(b) variation, using the same (old) matrix for a like-for-like check
set_feature_mode("complement"); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt")); cores, r = gauge_normalise(mps, Ptr)
m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), make_cell())
W = m.build_W(); m.cell.matrix = W
for mode in ["complement", "balanced"]:
    set_feature_mode(mode)
    with torch.no_grad():
        bop, _ = m.operating_points(x)
        G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
    print(f"[{mode}] loop DC response G(b(x)): event-to-event std per channel {float(G[:, :44].std(0).mean())*1e3:.3f} mV "
          f"(max over channels {float(G[:, :44].std(0).max())*1e3:.3f} mV)")
