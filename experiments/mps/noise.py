import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps
Xtr, ytr, Xte, yte = load_data(); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt"))
cores, r = gauge_normalise(mps, Ptr)
print("noise sigma per channel per iteration; full-scale swing of the nonlinearity ~ +-227 mV")
for umax in [0.005, 0.02]:
    eps = choose_eps(cores, Ptr, umax)
    m = AOCMPS(cores, r, eps, make_cell(), n_iter=60); m.exact_dc = True; m.init_at_target = True
    m.load_state_dict(torch.load(f"aocmps_ft_{int(umax*1e3)}mV.pt"))
    row = []
    for sig in [0.0, 1e-4, 3e-4, 1e-3, 3e-3]:
        for navg in ([0, 40] if sig > 0 else [0]):
            m.noise_std, m.n_avg = sig, navg
            accs = []
            for rep in range(3):
                torch.manual_seed(rep)
                with torch.no_grad(): accs.append(((m(Xte)[0] > 0).float() == yte).float().mean().item())
            row.append(f"sigma={sig*1e3:4.1f}mV avg={navg:2d}: {np.mean(accs):.3f}+-{np.std(accs):.3f}")
    print(f"\nmodel fine-tuned at u_max={umax*1e3:.0f} mV (fixed readout from fine-tune):")
    for s in row: print("   ", s)
