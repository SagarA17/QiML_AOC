"""Simulated bifurcation (quantum-inspired; Goto et al., Sci. Adv. 2019/2021) and its AOC mapping.

dSB (Delta t = 1):  y += -(1 - a(t)) x + c0 J sign(x);  x += y;  |x| > 1 -> x = sign(x), y = 0.
Eliminating y:      x_{t+1} = a(t) x_t + c0 J sign(x_t) + 1 * (x_t - x_{t-1})     == AOC eq. (1)
                    with alpha(t) = a(t): 0 -> 1, beta = c0, f = sign (high-gain tanh), gamma = 1.
"""
import math
import torch
from ising import best_of, energy, calibrate_field
from common import F_pos, N_CH


def c0_goto(J, h=None):
    """c0 = 0.5 sqrt((N - 1) / (sum_ij J_ij^2 + 2 sum_i h_i^2)) (Goto et al. 2021; h term as in Tao et al. 2026)."""
    N = J.shape[0]
    s = float((J ** 2).sum()) + (2.0 * float((h ** 2).sum()) if h is not None else 0.0)
    return 0.5 * math.sqrt((N - 1) / s)


def sb_ideal(J, h, runs=1000, T=1000, variant="dSB", c0_scale=1.0, seed=0, window=40):
    """Goto's (b/d)SB, Delta t = 1, a0 = 1, linear pump, inelastic walls; best-of-window readout."""
    g = torch.Generator().manual_seed(seed)
    N = J.shape[0]; c0 = c0_scale * c0_goto(J, h)
    x = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    y = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    samples = []
    for t in range(T):
        a = t / T
        src = torch.sign(x) if variant == "dSB" else x
        y = y + (-(1.0 - a) * x + c0 * (src @ J + h))
        x = x + y
        wall = x.abs() > 1
        x = torch.where(wall, torch.sign(x), x); y = torch.where(wall, torch.zeros_like(y), y)
        if t >= T - window:
            samples.append(torch.sign(x) + (x == 0).float())
    return best_of(torch.stack(samples), J, h)


def sb_twin(tw, runs=1000, T=1000, gamma=1.0, c0_scale=1.0, seed=0, wall=3.0, window=40, field_cal=True):
    """dSB on the hardware twin: alpha(t) = t/T, beta = c0, momentum gamma, the twin's own saturation
    as the (inelastic) wall, tanh at high drive as sign. Same static programming + field calibration."""
    from aoc.aoc_cell import create_saturation
    sat = create_saturation(tw.cell.hardware_parameters.saturation)
    g = torch.Generator().manual_seed(seed)
    k = wall * c0_scale * c0_goto(tw.J, tw.h) / tw.amp
    c = tw.program_k(k)
    if field_cal:
        c = calibrate_field(tw, k, c)
    c = c.unsqueeze(0).expand(runs, -1)
    x0 = 0.2 * torch.rand(runs, tw.N, generator=g) - 0.1
    y0 = 0.2 * torch.rand(runs, tw.N, generator=g) - 0.1
    z = torch.zeros(runs, N_CH); z[:, tw.ch] = wall * x0
    zp = z.clone(); zp[:, tw.ch] = wall * (x0 - y0)            # initial momentum x0 - x_{-1} = y0
    tw.cell.z_prev = zp
    samples = []
    with torch.no_grad():
        for t in range(T):
            z = tw.cell(z, x=c, alpha=t / T, gamma=gamma)
            z = sat(z)
            if t >= T - window:
                spin = F_pos(z[:, tw.ch]) - tw.mid
                samples.append(torch.sign(spin) + (spin == 0).float())
    return best_of(torch.stack(samples), tw.J, tw.h)
