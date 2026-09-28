"""Tabu-Enhanced dSB (Tao, Zeng, ..., Okawa, Yung, Commun. Phys. 9, 100 (2026)) and its AOC mapping.

Penalty (c0*beta / 2|Mb|) sum_{s in Mb} |x + s|^2 expands to
  * a pump shift       a(t) -> a(t) - c0*beta          (AOC: annealing schedule alpha(t))
  * a linear field     T_i = -(c0*beta/|Mb|) sum_s s_i  (AOC: additive injection, no SLM change)
Variants:
  "paper"    : mini-batch Mb re-drawn every iteration, pump exactly as published (alpha' from -c0*beta)
  "hardware" : Mb drawn once per sample (injection fixed during a run), pump shifted so alpha' in [0, 1]
"""
import math
import torch
from ising import best_of, energy, calibrate_field
from common import F_pos, N_CH
from sb import c0_goto


def _dsb_runs(J, h, runs, T, c0, g, field=None, pump_shift=0.0, per_iter_field=None, window=40, final_only=False):
    """Idealised dSB (dt = 1) with optional tabu field and pump shift. Returns spins (best-of-window or final)."""
    N = J.shape[0]
    x = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    y = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    samples = []
    for t in range(T):
        a = t / T
        drive = c0 * (torch.sign(x) @ J + h)
        if per_iter_field is not None:
            drive = drive + per_iter_field()
        elif field is not None:
            drive = drive + field
        y = y + (-(1.0 - (a - pump_shift)) * x + drive)
        x = x + y
        wall = x.abs() > 1
        x = torch.where(wall, torch.sign(x), x); y = torch.where(wall, torch.zeros_like(y), y)
        if not final_only and t >= T - window:
            samples.append(torch.sign(x) + (x == 0).float())
    if final_only:
        return torch.sign(x) + (x == 0).float()
    return best_of(torch.stack(samples), J, h)


def tesb_ideal(J, h, variant="paper", n_warm=100, runs=1000, T=1000, alpha=0.9, beta=1.0, mb=2,
               c0_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    c0w = c0_goto(J); c0 = c0_scale * c0w
    Tw, Tc = int(round((1 - alpha) * T)), int(round(alpha * T))
    M = _dsb_runs(J, h, n_warm, Tw, c0w, g, final_only=True)             # tabu list (warm-up minima)
    N = J.shape[0]
    if variant == "paper":
        def per_iter():                                                   # fresh mini-batch per iteration
            idx = torch.randint(0, len(M), (runs, mb), generator=g)
            return -(c0 * beta / mb) * M[idx].sum(1)
        return _dsb_runs(J, h, runs, Tc, c0, g, per_iter_field=per_iter, pump_shift=c0 * beta)
    idx = torch.randint(0, len(M), (runs, mb), generator=g)              # one mini-batch per sample
    field = -(c0 * beta / mb) * M[idx].sum(1)
    return _dsb_runs(J, h, runs, Tc, c0, g, field=field, pump_shift=0.0)  # alpha' = a(t) in [0, 1]


def tesb_twin(tw, n_warm=100, runs=1000, T=1000, alpha=0.9, beta=1.0, mb=2, c0_scale=1.0, gamma=0.95,
              seed=0, wall=3.0, window=40):
    """Hardware-compatible TE-dSB on the twin: warm-up dSB runs give the tabu list; checking runs get a
    per-sample tabu field added to the (field-calibrated) static injection."""
    from aoc.aoc_cell import create_saturation
    sat = create_saturation(tw.cell.hardware_parameters.saturation)
    g = torch.Generator().manual_seed(seed)
    c0w = c0_goto(tw.J)
    Tw, Tc = int(round((1 - alpha) * T)), int(round(alpha * T))

    def run(nruns, iters, c0, extra=None, keep_window=True):
        k = wall * c0 / tw.amp
        c = calibrate_field(tw, k, tw.program_k(k))                     # calibrate BEFORE adding tabu field
        c = c.unsqueeze(0).expand(nruns, -1).clone()
        if extra is not None:
            c[:, tw.ch] = c[:, tw.ch] + wall * extra
        x0 = 0.2 * torch.rand(nruns, tw.N, generator=g) - 0.1
        y0 = 0.2 * torch.rand(nruns, tw.N, generator=g) - 0.1
        z = torch.zeros(nruns, N_CH); z[:, tw.ch] = wall * x0
        zp = z.clone(); zp[:, tw.ch] = wall * (x0 - y0); tw.cell.z_prev = zp
        samples = []
        with torch.no_grad():
            for t in range(iters):
                z = sat(tw.cell(z, x=c, alpha=t / iters, gamma=gamma))
                if keep_window and t >= iters - window:
                    sp = F_pos(z[:, tw.ch]) - tw.mid; samples.append(torch.sign(sp) + (sp == 0).float())
        if not keep_window:
            sp = F_pos(z[:, tw.ch]) - tw.mid; return torch.sign(sp) + (sp == 0).float()
        return best_of(torch.stack(samples), tw.J, tw.h)

    M = run(n_warm, Tw, c0w, keep_window=False)
    c0 = c0_scale * c0w
    idx = torch.randint(0, len(M), (runs, mb), generator=g)
    field = -(c0 * beta / mb) * M[idx].sum(1)
    return run(runs, Tc, c0, extra=field)
