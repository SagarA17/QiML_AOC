import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps
cm.PLACEMENT["idle"] = [8, 1, 46, 3]; set_feature_mode("balanced", None)
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal_seesaw.pt")); cores, r = gauge_normalise(mps, Ptr)
eps = choose_eps(cores, Ptr, 0.02); probe = torch.rand(256, N_SITES)
def drives(P):
    with torch.no_grad():
        l = torch.einsum("bs,slr->blr", P[:,0], cores[0])[:,0,:]; out = {0: eps*l}
        for k in range(1, N_SITES):
            d = torch.einsum("bl,slr->bsr", l, cores[k]); out[k] = eps*d; l = torch.einsum("bs,bsr->br", P[:,k], d)
    return out, l[:,0]
idl, _ = drives(Ptr[:600]); _, lt_te = drives(Pte)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
def run(lbl, flags, comp):
    m = AOCMPS(cores, r, eps, make_cell(**flags), n_iter=60); m.init_at_target = True; m.floor_cancel = True
    m.static_dc = True; m.static_probe = probe; m.xtalk_comp = comp
    with torch.no_grad():
        _, z, b = m.run(Xtr[:600], return_state=True); fh_tr, fh_te = m.run(Xtr), m.run(Xte)
    out = []
    for k in range(N_SITES):
        ch = m.idx["e1"] if k == 0 else m.idx["gates"][k].reshape(-1)
        meas = (z-b)[:, ch].reshape(-1); i = idl[k].reshape(-1)
        g = float((meas*i).sum()/(i*i).sum()); res = float((meas-g*i).pow(2).mean().sqrt()/i.pow(2).mean().sqrt())
        out.append(f"{g:4.2f}/{res:4.2f}")
    acc = LogisticRegression(C=1e4, max_iter=5000).fit(fh_tr[:, None].numpy(), ytr.numpy()).score(fh_te[:, None].numpy(), yte.numpy())
    print(f"{lbl:40s} acc {acc:.3f} corr {float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0,1]):+.3f} | {' '.join(out)}")
XT = {**OFF, "apply_pd_crosstalk": True}
run("crosstalk only, no compensation", XT, False)
run("crosstalk only, with compensation", XT, True)
for extra in ["apply_slm_darkness", "apply_input_efficiencies", "apply_output_efficiencies", "apply_weight_distortion"]:
    run(f"crosstalk + {extra.replace('apply_','')}", {**XT, extra: True}, True)
print()
run("FULL TWIN (all repo-default effects)", {}, True)
