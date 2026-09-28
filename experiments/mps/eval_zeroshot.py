import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps, layout
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt"))
with torch.no_grad():
    acc_ideal = ((mps(Pte)[0] > 0).float() == yte).float().mean().item()
cores, r = gauge_normalise(mps, Ptr)
print(f"ideal MPS test acc {acc_ideal:.3f}; channels used {layout()['used']}/48; env scales r={np.round(r.numpy(),3)}")
def ideal_ltilde(P):
    with torch.no_grad():
        l = torch.ones(P.shape[0],1)
        for k,A in enumerate(cores): l = torch.einsum("bl,blr->br", l, torch.einsum("bs,slr->blr", P[:,k], A))
    return l[:,0]
lt_tr, lt_te = ideal_ltilde(Ptr), ideal_ltilde(Pte)
def score(fh_tr, fh_te):
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(fh_tr[:,None].numpy(), ytr.numpy())
    return lr.score(fh_te[:,None].numpy(), yte.numpy())
print(f"sanity: refit readout on exact l~_N -> test acc {score(lt_tr, lt_te):.3f}")
configs = {"ideal loop (all non-idealities off)": dict(apply_weight_distortion=False, apply_input_efficiencies=False,
                 apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False),
           "full twin (repo defaults)": {}}
for name, flags in configs.items():
    print(f"\n== {name}")
    cell = make_cell(**flags)
    for umax in [0.002, 0.005, 0.01, 0.02, 0.04]:
        eps = choose_eps(cores, Ptr, umax)
        m = AOCMPS(cores, r, eps, cell, n_iter=100)
        with torch.no_grad():
            fh_tr = m.run(Xtr); fh_te, z, b = m.run(Xte, return_state=True)
            z2 = m.cell(z, x=m.injection(Xte, m.build_W())[0])
            resid = float((z2-z).norm()/z.norm())
        rel = float((fh_te-lt_te).pow(2).mean().sqrt()/lt_te.pow(2).mean().sqrt())
        corr = float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0,1])
        print(f"  u_max={umax*1e3:4.0f} mV eps={eps*1e3:6.2f} mV | rel.err vs exact MPS {rel:7.3f} corr {corr:6.3f} | "
              f"test acc {score(fh_tr, fh_te):.3f} | clip {m.clip_frac:.3f} | fp resid {resid:.1e}")
