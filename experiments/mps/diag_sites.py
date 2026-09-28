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
x = Xtr[:600]; P = Ptr[:600]
# ideal pre-gate drives u_{k,s,j} = eps * (A_k[s]^T l~_{k-1})
with torch.no_grad():
    l = torch.einsum("bs,slr->blr", P[:,0], cores[0])[:,0,:]; ideal = {0: eps*l}
    for k in range(1, N_SITES):
        drive = torch.einsum("bl,slr->bsr", l, cores[k]); ideal[k] = eps*drive
        l = torch.einsum("bs,bsr->br", P[:,k], drive)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
for label, flags in [("only weight distortion", {**OFF,"apply_weight_distortion":True}), ("only input eff.", {**OFF,"apply_input_efficiencies":True}), ("full twin", {})]:
    m = AOCMPS(cores, r, eps, make_cell(**flags)); m.exact_dc = True; m.init_at_target = True
    with torch.no_grad():
        _, z, b = m.run(x, return_state=True)
        m.signal_on = False; _, z0, _ = m.run(x, return_state=True); m.signal_on = True
    print(f"== {label}: zero-signal DC error rms {float((z0-b)[:, :44].pow(2).mean().sqrt())*1e3:.4f} mV")
    for k in range(N_SITES):
        ch = m.idx["e1"] if k == 0 else m.idx["gates"][k].reshape(-1)
        meas = (z - b)[:, ch].reshape(-1); idl = ideal[k].reshape(-1)
        g = float((meas*idl).sum()/(idl*idl).sum()); res = float((meas - g*idl).pow(2).mean().sqrt()/idl.pow(2).mean().sqrt())
        print(f"   site {k+1}: rms ideal {float(idl.pow(2).mean().sqrt())*1e3:6.3f} mV  best gain {g:6.3f}  residual after gain {res:6.3f}")
