import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from common import *
from compile_mps import AOCMPS, gauge_normalise, choose_eps

torch.manual_seed(0)
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal.pt"))
cores, r = gauge_normalise(mps, Ptr)
with torch.no_grad():
    l = torch.ones(Pte.shape[0], 1)
    for k, A in enumerate(cores):
        l = torch.einsum("bl,blr->br", l, torch.einsum("bs,slr->blr", Pte[:, k], A))
lt_te = l[:, 0]


def evaluate(m, tag):
    with torch.no_grad():
        fh_tr, fh_te = m.run(Xtr), m.run(Xte)
        acc_readout = ((m.a * fh_te + m.c > 0).float() == yte).float().mean().item()
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(fh_tr[:, None].numpy(), ytr.numpy())
    acc_refit = lr.score(fh_te[:, None].numpy(), yte.numpy())
    corr = float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0, 1])
    print(f"  [{tag}] test acc (refit 2-param readout) {acc_refit:.3f} | corr with ideal MPS {corr:.3f}")
    return acc_refit


results = {}
for umax in [0.005, 0.02]:
    print(f"\n=== u_max = {umax*1e3:.0f} mV, full twin (repo-default non-idealities) ===")
    eps = choose_eps(cores, Ptr, umax)
    m = AOCMPS(cores, r, eps, make_cell(), n_iter=60)
    m.exact_dc = True; m.init_at_target = True
    a0 = evaluate(m, "zero-shot")
    # init readout from refit
    with torch.no_grad():
        fh = m.run(Xtr)
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(fh[:, None].numpy(), ytr.numpy())
    m.a.data = torch.tensor(float(lr.coef_[0, 0])); m.c.data = torch.tensor(float(lr.intercept_[0]))
    opt = torch.optim.Adam(list(m.cores.parameters()) + [m.a, m.c], lr=3e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for step in range(301):
        idx = torch.randperm(Xtr.shape[0])[:256]
        opt.zero_grad()
        logit, _ = m(Xtr[idx])
        loss = bce(logit, ytr[idx]); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step:3d} loss {loss.item():.3f}  ({time.time()-t0:.0f}s)")
    a1 = evaluate(m, "after HW-aware fine-tune")
    results[umax] = (a0, a1)
    torch.save(m.state_dict(), f"aocmps_ft_{int(umax*1e3)}mV.pt")
print("\nsummary:", {f"{k*1e3:.0f}mV": v for k, v in results.items()})
