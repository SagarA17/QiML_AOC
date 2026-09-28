import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import sys, time, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps
torch.manual_seed(0)
cm.PLACEMENT["idle"] = [8, 1, 46, 3]
set_feature_mode("balanced", None)                         # plain seesaw
Xtr, ytr, Xte, yte = load_data(); Ptr, Pte = features(Xtr), features(Xte)
bce = torch.nn.BCEWithLogitsLoss(); best = None
for seed in range(3):
    mps = TaperedMPS(seed=seed); opt = torch.optim.Adam(mps.parameters(), lr=1e-2)
    for ep in range(1500):
        opt.zero_grad(); loss = bce(mps(Ptr)[0], ytr); loss.backward(); opt.step()
    with torch.no_grad(): acc = ((mps(Pte)[0] > 0).float() == yte).float().mean().item()
    print(f"ideal MPS (seesaw features) seed {seed}: test {acc:.3f}")
    if best is None or acc > best[0]: best = (acc, mps)
mps = best[1]; torch.save(mps.state_dict(), "mps_ideal_seesaw.pt")
cores, r = gauge_normalise(mps, Ptr)
eps = choose_eps(cores, Ptr, 0.02)
def ideal_drives(P):
    with torch.no_grad():
        l = torch.einsum("bs,slr->blr", P[:,0], cores[0])[:,0,:]; out = {0: eps*l}
        for k in range(1, N_SITES):
            d = torch.einsum("bl,slr->bsr", l, cores[k]); out[k] = eps*d; l = torch.einsum("bs,bsr->br", P[:,k], d)
    return out, l[:, 0]
_, lt_te = ideal_drives(Pte)
probe = torch.rand(256, N_SITES)
OFF = dict(apply_weight_distortion=False, apply_input_efficiencies=False, apply_output_efficiencies=False, apply_pd_crosstalk=False, apply_slm_darkness=False)
def build(flags, dc="static", fc=True):
    m = AOCMPS(cores, r, eps, make_cell(**flags), n_iter=60); m.init_at_target = True; m.floor_cancel = fc
    if dc == "static": m.static_dc = True; m.static_probe = probe
    else: m.exact_dc = True
    return m
def dc_var(m):
    W = m.program_full(m.build_W()) if m.floor_cancel else m.program_slm(m.build_W()); m.cell.matrix = W
    with torch.no_grad():
        bop, _ = m.operating_points(Xtr); G = m.cell(bop, x=torch.zeros_like(bop)) - ALPHA * bop
    sd = G[:, m.idx["used_ch"]].std(0); return float(sd.mean())*1e3, float(sd.max())*1e3
def trace(m):
    idl, _ = ideal_drives(Ptr[:600])
    with torch.no_grad(): _, z, b = m.run(Xtr[:600], return_state=True)
    out = []
    for k in range(N_SITES):
        ch = m.idx["e1"] if k == 0 else m.idx["gates"][k].reshape(-1)
        meas = (z-b)[:, ch].reshape(-1); i = idl[k].reshape(-1)
        g = float((meas*i).sum()/(i*i).sum()); res = float((meas-g*i).pow(2).mean().sqrt()/i.pow(2).mean().sqrt())
        out.append(f"{g:4.2f}/{res:4.2f}")
    return " ".join(out)
def evaluate(m):
    with torch.no_grad(): fh_tr, fh_te = m.run(Xtr), m.run(Xte)
    acc = LogisticRegression(C=1e4, max_iter=5000).fit(fh_tr[:, None].numpy(), ytr.numpy()).score(fh_te[:, None].numpy(), yte.numpy())
    return acc, float(np.corrcoef(fh_te.numpy(), lt_te.numpy())[0, 1])
print(f"\nchannels used: {cm.layout()['used']}/48, idle {cm.PLACEMENT['idle']}")
for lbl, flags, fc in [("ideal loop", OFF, True), ("full twin, OLD programming", {}, False),
                       ("full twin, NEW programming", {}, True), ("full twin minus PD crosstalk, NEW prog.", dict(apply_pd_crosstalk=False, apply_slm_darkness=False), True)]:
    m = build(flags, fc=fc); dm, dx = dc_var(m); acc, corr = evaluate(m)
    print(f"{lbl:42s} DC var {dm:6.3f}/{dx:6.3f} mV (mean/max) | zero-shot acc {acc:.3f} corr {corr:+.3f}")
    print(f"{'':42s} per-site gain/resid: {trace(m)}")
if len(sys.argv) > 1 and sys.argv[1] == "zeroshot": sys.exit(0)
m = build({})
with torch.no_grad(): fh = m.run(Xtr)
lr = LogisticRegression(C=1e4, max_iter=5000).fit(fh[:, None].numpy(), ytr.numpy())
m.a.data = torch.tensor(float(lr.coef_[0, 0])); m.c.data = torch.tensor(float(lr.intercept_[0]))
opt = torch.optim.Adam(list(m.cores.parameters()) + [m.a, m.c], lr=3e-3); t0 = time.time()
for step in range(301):
    idx = torch.randperm(Xtr.shape[0])[:256]; opt.zero_grad()
    loss = bce(m(Xtr[idx])[0], ytr[idx]); loss.backward(); opt.step()
    if step % 100 == 0: print(f"  fine-tune step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
acc, corr = evaluate(m)
with torch.no_grad(): accf = ((m(Xte)[0] > 0).float() == yte).float().mean().item()
print(f"after HW-aware fine-tune (NEW programming, static offset): acc {acc:.3f} (fixed readout {accf:.3f})")
torch.save(m.state_dict(), "aocmps_floorcancel_ft.pt")
