import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch
from common import *
import compile_mps as cm
from compile_mps import AOCMPS
cm.PLACEMENT["idle"] = [8, 1, 46, 3]
torch.manual_seed(0)
# a toy target: a few intended weights, everything else zero
M = torch.zeros(48, 48); M[5, 10] = 3.0; M[6, 10] = -2.0; M[7, 20] = 1.0; M[12, 30] = -4.0
dummy = [torch.zeros(2, 1, 2)] + [torch.zeros(2, 2, 2)] * 7
for pdx in [False, True]:
    cell = make_cell(apply_pd_crosstalk=pdx, apply_slm_darkness=pdx)
    m = AOCMPS(dummy, torch.ones(8), 0.01, cell)
    for label, prog in [("old: relu split, floors untouched", lambda M: M),
                        ("new: every pixel programmed", m.program_full)]:
        m.floor_cancel = label.startswith("new")
        Wp = prog(M); cell.matrix = Wp
        # light channel i by dz around a bias point; response of each column relative to intended
        b0 = torch.full((1, 48), 0.2); dz = 1e-3
        resp = torch.zeros(48, 48)
        with torch.no_grad():
            base = cell(b0, x=torch.zeros(1, 48)) - ALPHA * b0
            for i in [5, 6, 7, 12, 20, 40]:
                z = b0.clone(); z[0, i] += dz
                resp[i] = ((cell(z, x=torch.zeros(1, 48)) - ALPHA * z) - base)[0] / (dz * float(cell.scaling.mean()))
        fp = float(torch.autograd.functional.jacobian(F_pos, torch.tensor(0.2)))
        intended = M * fp
        rows = [5, 6, 7, 12, 20, 40]
        err_used = [(float(resp[i, j]), float(intended[i, j])) for i, j in [(5, 10), (6, 10), (7, 20), (12, 30)]]
        mask = intended[rows] == 0
        leak = resp[rows][mask]
        print(f"[pd crosstalk {'on ' if pdx else 'off'}] {label:36s} intended pixels (got, want): "
              + ", ".join(f"({g:+.3f},{w:+.3f})" for g, w in err_used)
              + f" | leak into 'zero' pixels: rms {float(leak.pow(2).mean().sqrt()):.4f}, max {float(leak.abs().max()):.4f}")
