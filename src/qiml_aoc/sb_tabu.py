"""Tabu-Enhanced dSB (TESB; Tao, Zeng, ..., Okawa, Yung, Commun. Phys. 9, 100 (2026)) and its AOC mapping.

Warm-up:  m plain-dSB runs of (1 - alpha) T steps. Their final states, minus those reaching the best warm-up
          energy, form the tabu list M (the paper's "sub-optimal solutions"; the authors' code drops the best).
Checking: alpha T steps of dSB with the penalty (c0 beta / 2|Mb|) sum_{s in Mb} |x + s|^2, |Mb| = mb, which
          expands to
  * a pump shift   a(t) -> a(t) - c0 beta                    (paper eq. 8; the authors' released code omits it)
  * a linear field T_i = -(c0 beta / |Mb|) sum_{s in Mb} s_i  (AOC: additive injection, no SLM change)
Mini-batch draw:
  "iteration": one Mb per iteration, shared by all samples (paper Algorithm 1 and the authors' code)
  "sample":    one Mb per sample, fixed for the whole run (a static per-run injection on the AOC)
Readout: sign(x) at the last step (paper); window > 1 keeps the best of the last `window` steps.
Checking-phase c0: the paper grid-searches it; here c0 = c0_scale * (Goto's c0). Warm-up uses Goto's c0.
"""
import torch
from ising import best_of, energy, calibrate_field
from common import F_pos, N_CH
from sb import c0_goto


def tabu_list(states, J, h, tol=1e-4):
    """Warm-up final states minus those within tol of the best warm-up energy (may be empty)."""
    E = energy(states, J, h)
    return states[E > E.min() + tol]


def _tabu_field(M, runs, coef, mb, minibatch, g):
    """Tabu field T = -(coef / mb) * sum of mb random tabu states: [runs, N] tensor or callable t -> [N]."""
    if len(M) == 0:
        return None
    if minibatch == "sample":
        return -(coef / mb) * M[torch.randint(0, len(M), (runs, mb), generator=g)].sum(1)
    if minibatch == "iteration":
        return lambda t: -(coef / mb) * M[torch.randint(0, len(M), (mb,), generator=g)].sum(0)
    raise ValueError(f"minibatch must be 'iteration' or 'sample', not {minibatch!r}")


def _dsb(J, h, runs, T, c0, g, field=None, pump_shift=0.0, window=1):
    """Idealised dSB (dt = 1, a0 = 1, inelastic walls) with an optional tabu field and pump shift."""
    N = J.shape[0]
    x = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    y = 0.2 * torch.rand(runs, N, generator=g) - 0.1
    samples = []
    for t in range(T):
        drive = c0 * (torch.sign(x) @ J + h)
        if field is not None:
            drive = drive + (field(t) if callable(field) else field)
        y = y + (-(1.0 - (t / T - pump_shift)) * x + drive)
        x = x + y
        wall = x.abs() > 1
        x = torch.where(wall, torch.sign(x), x); y = torch.where(wall, torch.zeros_like(y), y)
        if t >= T - window:
            samples.append(torch.sign(x) + (x == 0).float())
    return best_of(torch.stack(samples), J, h)


def tesb_ideal(J, h, runs=1000, T=1000, alpha=0.9, beta=1.0, mb=2, m=100, c0_scale=1.0,
               minibatch="iteration", pump_shift=True, window=1, seed=0):
    """Idealised TE-dSB. Defaults follow the paper. Returns (checking-phase spins [runs, N], tabu list M)."""
    g = torch.Generator().manual_seed(seed)
    c0w = c0_goto(J, h); c0 = c0_scale * c0w
    Tw, Tc = int(round((1 - alpha) * T)), int(round(alpha * T))
    M = tabu_list(_dsb(J, h, m, Tw, c0w, g), J, h)
    field = _tabu_field(M, runs, c0 * beta, mb, minibatch, g)
    shift = c0 * beta if (pump_shift and field is not None) else 0.0
    return _dsb(J, h, runs, Tc, c0, g, field=field, pump_shift=shift, window=window), M


def tesb_twin(tw, runs=1000, T=1000, alpha=0.9, beta=1.0, mb=2, m=100, c0_scale=1.0, minibatch="sample",
              pump_shift=False, gamma=0.95, wall=3.0, window=1, seed=0):
    """TE-dSB on the twin. Warm-up dSB runs give the tabu list; the tabu field enters through the injection,
    added after field calibration. Defaults are the conservative hardware choices (static per-run injection,
    alpha(t) in [0, 1]); whether per-iteration injection updates and alpha < 0 are possible is a question for
    the hardware team. Returns (checking-phase spins [runs, N], tabu list M)."""
    from aoc.aoc_cell import create_saturation
    sat = create_saturation(tw.cell.hardware_parameters.saturation)
    g = torch.Generator().manual_seed(seed)
    c0w = c0_goto(tw.J, tw.h); c0 = c0_scale * c0w
    Tw, Tc = int(round((1 - alpha) * T)), int(round(alpha * T))

    def run(nruns, iters, c0, field=None, shift=0.0, window=1):
        k = wall * c0 / tw.amp
        c = calibrate_field(tw, k, tw.program_k(k))                     # calibrate BEFORE adding tabu field
        c = c.unsqueeze(0).expand(nruns, -1).clone()

        def with_field(f):
            out = c.clone(); out[:, tw.ch] = out[:, tw.ch] + wall * f
            return out
        if field is None:
            inj = lambda t: c
        elif callable(field):
            inj = lambda t: with_field(field(t))
        else:
            c_static = with_field(field); inj = lambda t: c_static
        x0 = 0.2 * torch.rand(nruns, tw.N, generator=g) - 0.1
        y0 = 0.2 * torch.rand(nruns, tw.N, generator=g) - 0.1
        z = torch.zeros(nruns, N_CH); z[:, tw.ch] = wall * x0
        zp = z.clone(); zp[:, tw.ch] = wall * (x0 - y0); tw.cell.z_prev = zp
        samples = []
        with torch.no_grad():
            for t in range(iters):
                z = sat(tw.cell(z, x=inj(t), alpha=t / iters - shift, gamma=gamma))
                if t >= iters - window:
                    sp = F_pos(z[:, tw.ch]) - tw.mid; samples.append(torch.sign(sp) + (sp == 0).float())
        return best_of(torch.stack(samples), tw.J, tw.h)

    M = tabu_list(run(m, Tw, c0w), tw.J, tw.h)
    field = _tabu_field(M, runs, c0 * beta, mb, minibatch, g)
    shift = c0 * beta if (pump_shift and field is not None) else 0.0
    return run(runs, Tc, c0, field=field, shift=shift, window=window), M
