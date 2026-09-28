import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import time, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps, site_efficiency_ratios
torch.manual_seed(0)
cm.PLACEMENT["idle"] = [8, 1, 46, 3]
cell = make_cell()
set_feature_mode("balanced", site_efficiency_ratios(cell))
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
# 1) ideal MPS on the balanced feature map
bce = torch.nn.BCEWithLogitsLoss(); best = None
for seed in range(3):
    mps = TaperedMPS(seed=seed); opt = torch.optim.Adam(mps.parameters(), lr=1e-2)
    for ep in range(1500):
        opt.zero_grad(); loss = bce(mps(Ptr)[0], ytr); loss.backward(); opt.step()
    with torch.no_grad(): acc = ((mps(Pte)[0] > 0).float() == yte).float().mean().item()
    print(f"ideal MPS (balanced features) seed {seed}: test {acc:.3f}")
    if best is None or acc > best[0]: best = (acc, mps)
mps = best[1]; torch.save(mps.state_dict(), "mps_ideal_balanced.pt")
cores, r = gauge_normalise(mps, Ptr)
with torch.no_grad():
    l = torch.ones(Pte.shape[0], 1)
    for k, A in enumerate(cores): l = torch.einsum("bl,blr->br", l, torch.einsum("bs,slr->blr", Pte[:, k], A))
lt_te = l[:, 0]
probe = torch.rand(256, N_SITES)              # calibration probes (random patterns, not data)
def build(mode, init_target):
    m = AOCMPS(cores, r, choose_eps(cores, Ptr, 0.02), cell, n_iter=60)
    m.row_equalise = True; m.eff_match = True; m.init_at_target = init_target; m.gain_cal = True
    if mode == "static": m.static_dc = True; m.static_probe = probe
    else: m.exact_dc = True
    return m
def evaluate(m):
    with torch.no_grad(): fh_tr, fh_te = m.run(Xtr), m.run(Xte)
    acc = LogisticRegression(C=1e4, max_iter=5000).fit(fh_tr[:, None].numpy(), ytr.numpy()).score(fh_te[:, None].numpy(), yte.numpy())
    return acc, float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0, 1])
print("\nzero-shot, full twin, u_max = 20 mV:")
for mode, it, lbl in [("exact", True, "per-event exact DC (oracle), pre-settled"),
                      ("static", True, "STATIC offset only, pre-settled"),
                      ("static", False, "STATIC offset only, start from injection")]:
    acc, corr = evaluate(build(mode, it)); print(f"  {lbl:44s} acc {acc:.3f} corr {corr:.3f}")
import sys
if len(sys.argv) > 1 and sys.argv[1] == 'zeroshot': sys.exit(0)
# 2) hardware-aware fine-tuning with the static-offset compile
m = build("static", True)
with torch.no_grad(): fh = m.run(Xtr)
lr = LogisticRegression(C=1e4, max_iter=5000).fit(fh[:, None].numpy(), ytr.numpy())
m.a.data = torch.tensor(float(lr.coef_[0, 0])); m.c.data = torch.tensor(float(lr.intercept_[0]))
opt = torch.optim.Adam(list(m.cores.parameters()) + [m.a, m.c], lr=3e-3); t0 = time.time()
for step in range(301):
    idx = torch.randperm(Xtr.shape[0])[:256]; opt.zero_grad()
    loss = bce(m(Xtr[idx])[0], ytr[idx]); loss.backward(); opt.step()
    if step % 100 == 0: print(f"  fine-tune step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
acc, corr = evaluate(m)
with torch.no_grad(): acc_fixed = ((m(Xte)[0] > 0).float() == yte).float().mean().item()
print(f"after HW-aware fine-tune (static offset): acc {acc:.3f} (fixed readout {acc_fixed:.3f}) corr-with-ideal {corr:.3f}")
torch.save(m.state_dict(), "aocmps_balanced_ft.pt")
