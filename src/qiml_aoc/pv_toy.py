"""Toy 1D primary-vertex clustering as a QUBO on the AOC twin.

Window: 11 tracks, 2-4 true vertices within 2 mm (>= 0.15 mm apart), per-track z0 resolution 20-120 um.
QUBO over x_ik (track i -> slot k, 4 slots): E(x) = sum_k sum_{i<j} D_ij x_ik x_jk + lam sum_i (1 - sum_k x_ik)^2
D_ij = min(pull_ij, 6) - 2.5  (reward compatible pairs, penalise incompatible; empty slots are free).
"""
import math
import numpy as np
import torch

N_TRK, K_SLOT = 11, 4
PULL0, PULL_CAP = 2.5, 6.0


def make_window(rng, width=2.0, min_sep=0.15):
    Kt = int(rng.choice([2, 3, 4], p=[0.35, 0.4, 0.25]))
    while True:                                                # resolvable toy: minimum vertex separation
        v = np.sort(rng.uniform(0, width, Kt))
        if Kt == 1 or np.diff(v).min() >= min_sep:
            break
    counts = np.full(Kt, 2) + rng.multinomial(N_TRK - 2 * Kt, np.ones(Kt) / Kt)
    lab = np.repeat(np.arange(Kt), counts)
    sig = rng.uniform(0.02, 0.12, N_TRK)
    z = v[lab] + rng.normal(0, sig)
    perm = rng.permutation(N_TRK)
    return dict(z=z[perm], sig=sig[perm], lab=lab[perm], vz=v)


def pair_matrix(w):
    z, s = w["z"], w["sig"]
    pull = np.abs(z[:, None] - z[None, :]) / np.sqrt(s[:, None] ** 2 + s[None, :] ** 2)
    D = np.minimum(pull, PULL_CAP) - PULL0
    np.fill_diagonal(D, 0.0)
    return torch.tensor(D, dtype=torch.float32)


def idx(i, k):
    return i * K_SLOT + k


def build_qubo(D):
    """Return (Q, q, lam): E(x) = x^T Q x + q^T x + const, Q symmetric, zero diagonal."""
    n = N_TRK * K_SLOT
    reward = torch.clamp(-D, min=0).sum(1)                    # max a track can gain in one slot
    lam = 1.1 * float(reward.max()) + 0.5
    Q = torch.zeros(n, n); q = torch.zeros(n)
    for k in range(K_SLOT):
        for i in range(N_TRK):
            for j in range(N_TRK):
                if i != j:
                    Q[idx(i, k), idx(j, k)] += 0.5 * D[i, j]      # sum_{i<j} D x x  (symmetric split)
    # lam (1 - sum_k x_ik)^2 = lam [1 - 2 sum_k x + sum_k x + 2 sum_{k<l} x x]   (x^2 = x)
    for i in range(N_TRK):
        for k in range(K_SLOT):
            q[idx(i, k)] += -lam
            for l in range(K_SLOT):
                if k != l:
                    Q[idx(i, k), idx(i, l)] += lam
    return Q, q, lam


def qubo_to_ising(Q, q):
    """x = (1+s)/2.  x^T Q x + q^T x = 1/4 s^T Q s + (1/2 Q 1 + 1/2 q)^T s + const.
    Our Ising convention: E = -1/2 s^T J s - h^T s  ->  J = -Q/2, h = -(Q 1 + q)/2."""
    J = -0.5 * Q
    h = -0.5 * (Q.sum(1) + q)
    return J, h


def qubo_energy(x, Q, q):
    return torch.einsum("...i,ij,...j->...", x, Q, x) + x @ q


def exact_assignment(D):
    """Brute force over all K^N one-hot assignments: E = sum_{i<j} D_ij [a_i == a_j]."""
    best, arg = math.inf, None
    Dt = torch.triu(D, 1)
    total = K_SLOT ** N_TRK
    for start in range(0, total, 1 << 18):
        ids = torch.arange(start, min(start + (1 << 18), total))
        A = torch.stack([(ids // K_SLOT ** t) % K_SLOT for t in range(N_TRK)], 1)
        same = (A.unsqueeze(2) == A.unsqueeze(1)).float()
        E = (same * Dt).sum((1, 2))
        i = int(E.argmin())
        if float(E[i]) < best:
            best, arg = float(E[i]), A[i].clone()
    return best, arg


def decode(spins):
    """spins [R, N*K] -> assignments [R, N] (-1 where a track is not exactly one-hot)."""
    x = ((spins + 1) / 2).reshape(-1, N_TRK, K_SLOT)
    ok = x.sum(-1) == 1
    a = x.argmax(-1)
    return torch.where(ok, a, torch.full_like(a, -1))


def deterministic_annealing(w, T0=100.0, steps=60, iters=10, seed=0):
    """Simplified 1D DA clustering (CMS-style idea): soft assignments to K prototypes,
    temperature annealed to 1, hard assignment, merge prototypes closer than 3*median(sigma)."""
    rng = np.random.default_rng(seed)
    z, s = w["z"], w["sig"]
    v = np.median(z) + 1e-3 * rng.normal(size=K_SLOT)
    for T in np.geomspace(T0, 1.0, steps):
        for _ in range(iters):
            d2 = (z[:, None] - v[None, :]) ** 2 / (2 * s[:, None] ** 2 * T)
            p = np.exp(-(d2 - d2.min(1, keepdims=True))); p /= p.sum(1, keepdims=True)
            wgt = p / s[:, None] ** 2
            v = (wgt * z[:, None]).sum(0) / np.maximum(wgt.sum(0), 1e-12)
        v = v + 1e-4 * rng.normal(size=K_SLOT)                  # let degenerate prototypes split
    a = np.argmin((z[:, None] - v[None, :]) ** 2 / s[:, None] ** 2, 1)
    order = np.argsort(v); merged = {order[0]: 0}; c = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if v[cur] - v[prev] > 3 * np.median(s): c += 1
        merged[cur] = c
    return np.array([merged[k] for k in a])


def with_reference_spin(J, h):
    """Fold linear terms into the matrix (paper, Methods): extra spin s_r coupled with h.
    E' = -1/2 s'^T J' s' has the same minima as E with s_i -> s_i * s_r."""
    n = J.shape[0]
    Ja = torch.zeros(n + 1, n + 1); Ja[:n, :n] = J; Ja[:n, n] = h; Ja[n, :n] = h
    return Ja
