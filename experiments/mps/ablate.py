import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt"))
cores, r = gauge_normalise(mps, Ptr)
with torch.no_grad():
    l = torch.ones(Pte.shape[0],1)
    for k,A in enumerate(cores): l = torch.einsum("bl,blr->br", l, torch.einsum("bs,slr->blr", Pte[:,k], A))
lt_te = l[:,0]
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False,
           apply_pd_crosstalk=False, apply_slm_darkness=False)
names = ["apply_input_efficiencies","apply_output_efficiencies","apply_pd_crosstalk","apply_slm_darkness","apply_weight_distortion"]
eps = choose_eps(cores, Ptr, 0.005)
print(f"u_max = 5 mV (eps={eps*1e3:.2f} mV); one non-ideality ON at a time")
for nm in [None] + names:
    flags = dict(OFF); 
    if nm: flags[nm] = True
    if nm == "apply_slm_darkness": flags["apply_pd_crosstalk"] = True   # darkness only acts inside PD crosstalk
    m = AOCMPS(cores, r, eps, make_cell(**flags))
    with torch.no_grad(): fh_tr, fh_te = m.run(Xtr), m.run(Xte)
    rel = float((fh_te-lt_te).pow(2).mean().sqrt()/lt_te.pow(2).mean().sqrt())
    corr = float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0,1])
    acc = LogisticRegression(C=1e4,max_iter=5000).fit(fh_tr[:,None].numpy(), ytr.numpy()).score(fh_te[:,None].numpy(), yte.numpy())
    print(f"  {str(nm or 'none (ideal loop)'):28s} rel.err {rel:9.3f}  corr {corr:6.3f}  acc {acc:.3f}")
