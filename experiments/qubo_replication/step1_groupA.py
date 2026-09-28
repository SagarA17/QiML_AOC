import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import math, torch, numpy as np
from ising import *
import qubo_instances as rq
from common import make_cell, ALPHA
insts = torch.load("replicate_instances.pt", weights_only=False)
sch = torch.load("replicate_schedules.pt", weights_only=False); KT = dict(a0=sch["twin"][2], beta0=sch["twin"][3], gamma=sch["twin"][4])
h0 = torch.zeros(16); torch.manual_seed(0)
def effective_problem(tw, k):
    """Probe the programmed full twin: field(s) = G(z(s)) with spins held saturated at +-2 V.
    Fit field = A s + f0 on the used channels; return symmetric J_eff and h_eff in units of J."""
    c = tw.program_k(k); S = 4000
    s = torch.where(torch.rand(S, tw.N) < 0.5, -1.0, 1.0)
    z = torch.zeros(S, 48); z[:, tw.ch] = 2.0 * s
    with torch.no_grad():
        field = (tw.cell(z, x=c.expand(S, -1)) - ALPHA * z)[:, tw.ch]       # loop drive per spin
    X = torch.cat([s, torch.ones(S, 1)], 1)
    coef = torch.linalg.lstsq(X, field).solution                           # [N+1, N]
    A, f0 = coef[:-1], coef[-1]
    scale = k * tw.amp
    Jeff = 0.5 * (A + A.T) / scale; Jeff.fill_diagonal_(0.0)
    return Jeff, f0 / scale
print("inst kind bits | true opt E | E_true of twin's found state | effective-problem optimum == true optimum? | max |J_eff - J| / max|J|")
for i in [45, 60, 63, 64, 66, 83, 91, 92, 97]:
    kind, bits, J, Eopt = insts[i]
    tw = TwinIsing(J, h0, programming="new")
    k = 3.0 * KT["beta0"] / (tw.lam * tw.amp)
    Jeff, heff = effective_problem(tw, k)
    s_found = twin_run_paper(tw, runs=200, **KT)
    E_found = energy(s_found, J, h0)
    # exact optimum of the effective problem the hardware sees (with its residual field)
    ids = torch.arange(1 << 16); bits_ = ((ids.unsqueeze(1) >> torch.arange(16)) & 1).float() * 2 - 1
    Eeff = energy(bits_, Jeff, heff); s_eff = bits_[int(Eeff.argmin())]
    same = float(energy(s_eff, J, h0)) <= Eopt + 1e-6
    print(f" {i:3d} {kind:6s} {bits}  | {Eopt:9.4f} | modal {float(E_found.mode().values):9.4f} (best {float(E_found.min()):9.4f}) | "
          f"{'YES' if same else 'NO  -> hardware sees a different optimum'} (its true E {float(energy(s_eff, J, h0)):.4f}) | "
          f"{float((Jeff - J).abs().max() / J.abs().max()):.3f}")
print("\ninst | |h_eff| max (units of max|J|) | optimum of J_eff alone == true? | optimum of J + h_eff == true?")
for i in [45, 60, 63, 64, 83, 91, 92, 97]:
    kind, bits, J, Eopt = insts[i]
    tw = TwinIsing(J, h0, programming="new"); k = 3.0 * KT["beta0"] / (tw.lam * tw.amp)
    Jeff, heff = effective_problem(tw, k)
    ids = torch.arange(1 << 16); b_ = ((ids.unsqueeze(1) >> torch.arange(16)) & 1).float() * 2 - 1
    sJ = b_[int(energy(b_, Jeff, h0).argmin())]; sh = b_[int(energy(b_, J, heff).argmin())]
    print(f" {i:3d} | {float(heff.abs().max()/J.abs().max()):.3f} | {'YES' if float(energy(sJ, J, h0)) <= Eopt+1e-6 else 'NO'} | "
          f"{'YES' if float(energy(sh, J, h0)) <= Eopt+1e-6 else 'NO'}")
