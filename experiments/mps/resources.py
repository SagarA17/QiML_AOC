import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
import compile_mps as cm
from compile_mps import AOCMPS, gauge_normalise, choose_eps
cm.PLACEMENT["idle"] = [8, 1, 46, 3]; set_feature_mode("balanced", None)
Xtr, ytr, Xte, yte = load_data(); Ptr = features(Xtr)
mps = TaperedMPS(); mps.load_state_dict(torch.load("mps_ideal_seesaw.pt")); cores, r = gauge_normalise(mps, Ptr)
eps = choose_eps(cores, Ptr, 0.02); probe = torch.rand(256, N_SITES)
m = AOCMPS(cores, r, eps, make_cell(), n_iter=60); m.floor_cancel = True; m.static_dc = True; m.static_probe = probe
W = m.program_full(m.build_W()); m.cell.matrix = W
idx = m.idx; last = idx["gates"][N_SITES-1][:, 0]
# ---- static footprint
nz_target = int((m.build_W() != 0).sum())
print(f"channels used: {idx['used']}/48 (idle: {cm.PLACEMENT['idle']})")
print(f"SLM pixels: all 48x48 = 2304 programmed; {nz_target} carry MPS weights, the rest cancel the floor")
print(f"MPS parameters: {sum(A.numel() for A in cores)} (each gate block is shared by both source groups -> {nz_target} weight pixels)")
# ---- settling: iterate from both initial conditions, record readout trajectory
def settle(init_target):
    m.init_at_target = init_target
    with torch.no_grad():
        c, b, phi = m.injection(Xte, W)
        z = b.clone() if init_target else torch.clip(c, -1.3, 1.3)
        traj, dzs = [], []
        for t in range(200):
            zn = m.cell(z, x=c); dzs.append(float((zn - z).abs().max())); z = zn
            traj.append((phi[:, -1] * (z[:, last] - b[:, last])).sum(-1) / eps)
    traj = torch.stack(traj)                         # [T, B]
    final = traj[-1]
    rel = (traj - final).abs() / float(final.pow(2).mean().sqrt())   # tolerance relative to typical output size
    # first iteration after which readout stays within 1% of final
    within = (rel < 0.01)
    t_settle = torch.tensor([int(T) for T in [ (~within[:, i]).nonzero().max().item() + 1 if (~within[:, i]).any() else 0 for i in range(within.shape[1])]])
    return t_settle, dzs, final
for init in [True, False]:
    ts, dzs, final = settle(init)
    lbl = "pre-settled at operating point" if init else "started from injected input"
    print(f"\n[{lbl}] iterations until readout within 1% (of typical output) of final: median {int(ts.median())}, 90th pct {int(ts.float().quantile(0.9))}, max {int(ts.max())}")
    print(f"   max |dz| per iteration at t=10,20,40,60: {dzs[9]:.1e} {dzs[19]:.1e} {dzs[39]:.1e} {dzs[59]:.1e} V")
    if not init:
        _, _, final_ref = settle(True)
        print(f"   same fixed point as pre-settled? max |difference| in readout {float((final-final_ref).abs().max()):.2e} (readout rms {float(final_ref.pow(2).mean().sqrt()):.2e})")
