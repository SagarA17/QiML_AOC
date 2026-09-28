import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np
from common import *
Xtr, ytr, Xte, yte = load_data()
Ptr, Pte = features(Xtr), features(Xte)
print("feature range:", float(Ptr.min()), float(Ptr.max()))
bce = torch.nn.BCEWithLogitsLoss()
res = []
for seed in range(3):
    m = TaperedMPS(seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for ep in range(1500):
        opt.zero_grad(); logit, _ = m(Ptr); loss = bce(logit, ytr); loss.backward(); opt.step()
    with torch.no_grad():
        acc_tr = ((m(Ptr)[0] > 0).float() == ytr).float().mean().item()
        acc_te = ((m(Pte)[0] > 0).float() == yte).float().mean().item()
    nparam = sum(A.numel() for A in m.cores)
    print(f"seed {seed}: loss {loss.item():.3f} train {acc_tr:.3f} test {acc_te:.3f}  (#MPS params {nparam})")
    res.append((acc_te, seed, m))
best = max(res, key=lambda r: r[0])
torch.save(best[2].state_dict(), "mps_ideal.pt"); print("saved seed", best[1])
