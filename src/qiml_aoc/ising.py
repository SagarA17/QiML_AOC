"""Ising toy problems on the AOC: instances, references, SA baseline, idealized and twin solvers.

Convention (Ising): minimise E(s) = -1/2 s^T J s - h^T s,  s in {-1,+1}^N,  J symmetric, zero diagonal.
QUBO <-> Ising is an affine change of variables, so this covers QUBO.
"""
import math
import types
import numpy as np
import torch
from common import ALPHA, FP0, F_pos, N_CH, make_cell
import compile_mps as cm

torch.set_default_dtype(torch.float32)


# ------------------------------------------------------------------ instances
def wishart_planted(N, M_over_N=0.75, seed=0):
    """Wishart planted ensemble (Hamze et al.): J = -W^T W / N with W t = 0 -> t is a ground state."""
    g = torch.Generator().manual_seed(seed)
    t = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    M = max(1, int(round(M_over_N * N)))
    W = torch.randn(M, N, generator=g)
    W = W - (W @ t).unsqueeze(1) * t.unsqueeze(0) / N          # project out t: W t = 0
    J = -(W.T @ W) / N
    J.fill_diagonal_(0.0)
    J = J / J.abs().max()
    return J, torch.zeros(N), t


def three_regular(N, seed=0):
    """Random 3-regular graph with random +-1 couplings (sparse; NP-hard family used in the paper)."""
    rng = np.random.default_rng(seed)
    while True:
        stubs = np.repeat(np.arange(N), 3); rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        key = set(map(tuple, np.sort(pairs, 1)))
        if len(key) == len(pairs):
            break
    J = torch.zeros(N, N)
    for a, b in pairs:
        w = float(rng.choice([-1.0, 1.0])); J[a, b] = J[b, a] = w
    return J, torch.zeros(N), None


def energy(s, J, h):
    """s: [..., N] in {-1,+1}."""
    return -0.5 * torch.einsum("...i,ij,...j->...", s, J, s) - s @ h


def brute_force(J, h, chunk=1 << 20):
    """Exact ground-state energy (fix s_0 = +1 when h = 0 by symmetry). Feasible to N ~ 26."""
    N = J.shape[0]
    free = N - 1 if bool((h == 0).all()) else N
    best = math.inf
    for start in range(0, 1 << free, chunk):
        ids = torch.arange(start, min(start + chunk, 1 << free))
        bits = ((ids.unsqueeze(1) >> torch.arange(free)) & 1).float() * 2 - 1
        s = torch.cat([torch.ones(len(ids), 1), bits], 1) if free < N else bits
        best = min(best, float(energy(s, J, h).min()))
    return best


def simulated_annealing(J, h, runs=100, sweeps=1000, seed=0):
    """Single-spin Metropolis, geometric temperature schedule, vectorised over runs."""
    g = torch.Generator().manual_seed(seed)
    N = J.shape[0]
    s = torch.where(torch.rand(runs, N, generator=g) < 0.5, -1.0, 1.0)
    T0 = float((J.abs().sum(1) + h.abs()).max()); Tf = 0.02 * T0
    for k in range(sweeps):
        T = T0 * (Tf / T0) ** (k / max(1, sweeps - 1))
        for i in range(N):
            field = s @ J[:, i] + h[i]
            dE = 2 * s[:, i] * field
            acc = (dE <= 0) | (torch.rand(runs, generator=g) < torch.exp(-dE / T))
            s[:, i] = torch.where(acc, -s[:, i], s[:, i])
    return s


# ------------------------------------------------------------------ AOC dynamics
def norm_factor(J):
    """Normalisation used by AOCoptimizer.jl (_calculate_normalization_factor)."""
    ev = torch.linalg.eigvalsh(J); lmax, lmin = float(ev.max()), float(ev.min())
    if (lmax > 0) != (lmin > 0):
        lam = 1.0 if lmax <= 0.1 else lmax
    else:
        lam = (abs(lmax) + abs(lmin)) / 2
    return 1.0 if lam <= 0.1 else lam


def alpha_schedule(t, T, alpha0):
    """alpha(t) = 1 - alpha_hat(t), alpha_hat linear from alpha0 to 0 (paper, Methods)."""
    return 1.0 - alpha0 * (1.0 - t / max(1, T - 1))


def aoc_ideal(J, h, runs=100, T=1000, alpha0=1.0, beta0=0.5, gamma=0.0, seed=0, init_scale=0.01):
    """Idealised eq. (1): s <- alpha(t) s + beta J tanh(s) + gamma (s - s_prev) + beta h, beta = beta0/lambda_max."""
    g = torch.Generator().manual_seed(seed)
    lam = float(torch.linalg.eigvalsh(J).max())
    beta = beta0 / lam
    s = init_scale * torch.randn(runs, J.shape[0], generator=g); prev = s.clone()
    for t in range(T):
        a = alpha_schedule(t, T, alpha0)
        new = a * s + beta * (torch.tanh(s) @ J + h) + gamma * (s - prev)
        prev, s = s, new
    return torch.sign(s) + (s == 0).float()


class TwinIsing:
    """Map an Ising instance onto the 48-channel twin and run the annealed dynamics through AOCCell."""

    def __init__(self, J, h, idle=(8, 1, 46, 3), programming="new", field_comp=True, cell=None):
        self.cell = cell if cell is not None else make_cell()
        self.N = J.shape[0]
        phys = [i for i in range(N_CH) if i not in idle]
        assert self.N <= len(phys), f"{self.N} variables > {len(phys)} usable channels"
        self.ch = torch.tensor(phys[: self.N])
        self.J, self.h = J, h
        self.programming, self.field_comp = programming, field_comp
        hi, lo = float(F_pos(torch.tensor(2.0))), float(F_pos(torch.tensor(-2.0)))
        self.amp, self.mid = (hi - lo) / 2, (hi + lo) / 2             # 'spin' = mid + amp * s
        self.lam = norm_factor(J)

    def program(self, beta0):
        """Target effective matrix in physical channels, per-iteration gain beta0/(lambda_max F'(0))."""
        return self.program_k(beta0 / (self.lam * FP0))

    def program_k(self, k):
        """Program coupling matrix k*J (volts per unit 'spin' light) and return the static injection."""
        S = self.cell.scaling.detach()
        M = torch.zeros(N_CH, N_CH)
        M[self.ch.unsqueeze(1), self.ch.unsqueeze(0)] = k * self.J
        M = M / S.unsqueeze(0)                                        # undo closed-loop column scaling
        if self.programming == "new":
            dummy = types.SimpleNamespace(cell=self.cell, xtalk_comp=True)
            W = cm.AOCMPS.program_full(dummy, M)
        else:                                                         # plain signed matrix, relu split
            W = torch.where(M < 0, M * cm.NEG_RATIO, M)
        self.cell.matrix = W
        # static injection: linear field, minus known closed-loop offsets, minus 'spin' midpoint field
        off = float(self.cell.solution_to_adc_ratio) * (self.cell.dc_offset + W.abs().max() * self.cell.linear_offset)
        c = torch.zeros(N_CH)
        c[self.ch] = k * self.amp * self.h                            # h in the same units as J s
        c = c - S * off
        if self.field_comp:
            field = torch.zeros(N_CH)
            field[self.ch] = self.mid * (k * self.J).sum(0)           # J @ (mid * 1)
            c = c - S * field
        return c

    def run(self, runs=100, T=1000, alpha0=1.0, beta0=0.5, gamma=0.0, seed=0, init_scale=0.01):
        g = torch.Generator().manual_seed(seed)
        c = self.program(beta0).unsqueeze(0).expand(runs, -1)
        z = torch.zeros(runs, N_CH); z[:, self.ch] = init_scale * torch.randn(runs, self.N, generator=g)
        self.cell.z_prev = None
        with torch.no_grad():
            for t in range(T):
                z = self.cell(z, x=c, alpha=alpha_schedule(t, T, alpha0), gamma=gamma if gamma > 0 else None)
        spin = F_pos(z[:, self.ch]) - self.mid
        return torch.sign(spin) + (spin == 0).float()


# ------------------------------------------------------------------ paper-faithful dynamics (AOCoptimizer.jl port)
def best_of(samples, J, h):
    """samples: [W, runs, N] -> per run, the lowest-energy sample (digital readout over a window)."""
    E = energy(samples, J, h)                          # [W, runs]
    i = E.argmin(0)
    return samples[i, torch.arange(samples.shape[1])]


def aoc_paper(J, h, runs=100, T=1000, a0=0.3, beta0=0.1, gamma=0.0, seed=0, window=40):
    """Port of AOCoptimizer.jl `sample_mixed_ising!` (dt folded into a0, beta0):
         s = sign(x);  x += beta*(J s + h) - a(t) x + gamma (x - s_prev);  x <- clamp(x, -1, 1)
       a(t) linear from a0 to 0; x0 ~ U(-1/sqrt(N), 1/sqrt(N)); beta = beta0 / lambda_max."""
    g = torch.Generator().manual_seed(seed)
    N = J.shape[0]
    beta = beta0 / norm_factor(J)
    x = (2 * torch.rand(runs, N, generator=g) - 1) / math.sqrt(N)
    y = torch.zeros_like(x)
    samples = []
    for t in range(T):
        a = a0 * (1.0 - t / T)
        s = torch.sign(x) + (x == 0).float()
        x = x + beta * (s @ J + h) - a * x + gamma * (x - y)
        y = s
        x = torch.clamp(x, -1.0, 1.0)
        if t >= T - window:
            samples.append(torch.sign(x) + (x == 0).float())
    return best_of(torch.stack(samples), J, h)


def calibrate_field(tw, k, c, probes=4000, rounds=2, seed=123):
    """Static per-problem calibration: hold spins at random saturated patterns (open loop), fit the
    loop drive as A s + f0, and cancel the residual field f0 through the injection."""
    from common import ALPHA
    g = torch.Generator().manual_seed(seed)
    for _ in range(rounds):
        s = torch.where(torch.rand(probes, tw.N, generator=g) < 0.5, -1.0, 1.0)
        z = torch.zeros(probes, N_CH); z[:, tw.ch] = 2.0 * s
        with torch.no_grad():
            drive = (tw.cell(z, x=c.expand(probes, -1)) - ALPHA * z)[:, tw.ch]
        X = torch.cat([s, torch.ones(probes, 1)], 1)
        f0 = torch.linalg.lstsq(X, drive).solution[-1]
        c = c.clone(); c[tw.ch] = c[tw.ch] - f0
    return c


def twin_run_paper(tw, runs=100, T=1000, a0=0.3, beta0=0.1, gamma=0.0, seed=0, wall=3.0, window=40, init=None, field_cal=False):
    """Same algorithm on the hardware twin: tanh at high drive ~ sign, the twin's own physical
    saturation model inside the loop plays the wall. k maps beta0 (normalised x) to volts."""
    from aoc.aoc_cell import create_saturation
    sat = create_saturation(tw.cell.hardware_parameters.saturation)
    g = torch.Generator().manual_seed(seed)
    k = wall * beta0 / (tw.lam * tw.amp)
    c = tw.program_k(k)
    if field_cal:
        c = calibrate_field(tw, k, c)
    c = c.unsqueeze(0).expand(runs, -1)
    z = torch.zeros(runs, N_CH)
    amp0 = wall / math.sqrt(tw.N) if init is None else init
    z[:, tw.ch] = amp0 * (2 * torch.rand(runs, tw.N, generator=g) - 1)
    tw.cell.z_prev = None
    samples = []
    with torch.no_grad():
        for t in range(T):
            a = a0 * (1.0 - t / T)
            z = tw.cell(z, x=c, alpha=1.0 - a, gamma=gamma if gamma > 0 else None)
            z = sat(z)
            if t >= T - window:
                spin = F_pos(z[:, tw.ch]) - tw.mid
                samples.append(torch.sign(spin) + (spin == 0).float())
    return best_of(torch.stack(samples), tw.J, tw.h)
