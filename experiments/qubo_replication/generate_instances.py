"""Replicate the paper's hardware QUBO benchmark protocol (16 variables, dense SK + sparse 3-regular,
3-8 bit weights, principal-eigenvector filter, samples-to-optimum) on idealised + twin solvers."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import time, math, torch, numpy as np
from ising import *
from common import make_cell

def quantise(J, bits):
    q = 2 ** (bits - 1) - 1
    return torch.round(J / J.abs().max() * q) / q

def make_instance(kind, N, bits, rng):
    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    if kind == "dense":
        J = torch.randn(N, N, generator=g); J = (J + J.T) / 2
    else:
        J3, _, _ = three_regular(N, seed=int(rng.integers(1 << 30)))
        J = (J3 != 0).float() * torch.randn(N, N, generator=g); J = torch.triu(J, 1); J = J + J.T
    J.fill_diagonal_(0.0)
    return quantise(J, bits)

def ground_state(J):
    N = J.shape[0]
    ids = torch.arange(1 << (N - 1))
    bits = ((ids.unsqueeze(1) >> torch.arange(N - 1)) & 1).float() * 2 - 1
    s = torch.cat([torch.ones(len(ids), 1), bits], 1)
    E = energy(s, J, torch.zeros(N)); i = int(E.argmin())
    return float(E[i]), s[i]

def eig_trivial(J, s_opt, Eopt):
    """Rejected if the sign pattern of the principal eigenvector is already optimal."""
    v = torch.linalg.eigh(J).eigenvectors[:, -1]
    sv = torch.sign(v) + (v == 0).float()
    return float(energy(sv, J, torch.zeros(len(sv)))) <= Eopt + 1e-6

rng = np.random.default_rng(2025)
N = 16
insts = []
for kind in ["dense", "sparse"]:
    made = rejected = 0
    while made < 50:
        bits = int(rng.integers(3, 9))
        J = make_instance(kind, N, bits, rng)
        Eopt, sopt = ground_state(J)
        if eig_trivial(J, sopt, Eopt): rejected += 1; continue
        insts.append((kind, bits, J, Eopt)); made += 1
    print(f"{kind}: 50 instances kept, {rejected} rejected by the principal-eigenvector filter")
torch.save(insts, "replicate_instances.pt")
