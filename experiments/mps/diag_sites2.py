import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps, site_efficiency_ratios
Xtr, ytr, Xte, yte = load_data()
cm.PLACEMENT["idle"] = [8, 1, 46, 3]
cellfull = make_cell(); set_feature_mode("balanced", site_efficiency_ratios(cellfull))
Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal_balanced.pt")); cores, r = gauge_normalise(mps, Ptr)
eps = choose_eps(cores, Ptr, 0.02); x = Xtr[:600]; P = Ptr[:600]
with torch.no_grad():
    l = torch.einsum("bs,slr->blr", P[:,0], cores[0])[:,0,:]; ideal = {0: eps*l}
    for k in range(1, N_SITES):
        drive = torch.einsum("bl,slr->bsr", l, cores[k]); ideal[k] = eps*drive; l = torch.einsum("bs,bsr->br", P[:,k], drive)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
def trace(lbl, flags, req, em, gc=False):
    m = AOCMPS(cores, r, eps, make_cell(**flags)); m.exact_dc = True; m.init_at_target = True; m.row_equalise = req; m.eff_match = em; m.gain_cal = gc
    W = m.program_slm(m.build_W())
    with torch.no_grad(): _, z, b = m.run(x, return_state=True)
    out = []
    for k in range(N_SITES):
        ch = m.idx["e1"] if k == 0 else m.idx["gates"][k].reshape(-1)
        meas = (z-b)[:, ch]
        if em and 1 <= k <= N_SITES-2:        # eff-match pre-divides each group's drive by kappa
            kap = m._kappa()[k]; meas = meas * kap.repeat_interleave(m.idx["gates"][k].shape[1]).unsqueeze(0)
        meas = meas.reshape(-1); idl = ideal[k].reshape(-1)
        g = float((meas*idl).sum()/(idl*idl).sum()); res = float((meas-g*idl).pow(2).mean().sqrt()/idl.pow(2).mean().sqrt())
        out.append(f"{g:5.2f}/{res:4.2f}")
    print(f"{lbl:46s} amax(W)={float(W.abs().max()):6.2f} | per-site gain/resid: " + " ".join(out))
trace("ideal loop", OFF, False, False)
trace("full twin, row-eq + eff-match", {}, True, True)
trace("full twin, row-eq + eff-match + gain cal", {}, True, True, True)
# isolate the darkness floor: zero the constant term of the per-column SLM response, keep everything else
import types
def trace_nofloor(lbl, gc):
    cellnf = make_cell()
    cellnf.hardware_parameters.weight_distortion_pos[:, 2] = 0.0
    cellnf.hardware_parameters.weight_distortion_neg[:, 2] = 0.0
    m = AOCMPS(cores, r, eps, cellnf); m.exact_dc = True; m.init_at_target = True
    m.row_equalise = True; m.eff_match = True; m.gain_cal = gc
    W = m.program_slm(m.build_W())
    with torch.no_grad(): _, z, b = m.run(x, return_state=True)
    out = []
    for k in range(N_SITES):
        ch = m.idx["e1"] if k == 0 else m.idx["gates"][k].reshape(-1)
        meas = (z-b)[:, ch]
        if 1 <= k <= N_SITES-2:
            kap = m._kappa()[k]; meas = meas * kap.repeat_interleave(m.idx["gates"][k].shape[1]).unsqueeze(0)
        meas = meas.reshape(-1); idl = ideal[k].reshape(-1)
        g = float((meas*idl).sum()/(idl*idl).sum()); res = float((meas-g*idl).pow(2).mean().sqrt()/idl.pow(2).mean().sqrt())
        out.append(f"{g:5.2f}/{res:4.2f}")
    print(f"{lbl:46s} amax(W)={float(W.abs().max()):6.2f} | per-site gain/resid: " + " ".join(out))
trace_nofloor("full twin, floor zeroed, + gain cal", True)
