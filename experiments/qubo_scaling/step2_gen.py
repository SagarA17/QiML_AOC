"""Step 2: paper-family instances at N = 24, 32, 44 with exact (24, 32) or best-known (44) references."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))

import time, math, torch, numpy as np
from ising import energy, three_regular, simulated_annealing
import qubo_instances as rq

def exact_mitm(J):
    """Exact Ising ground state by meet-in-the-middle over two halves (N <= ~34). Fixes s_0 = +1."""
    N = J.shape[0]; nA = N // 2; nB = N - nA
    def allspins(n, fix_first=False):
        m = n - 1 if fix_first else n
        ids = torch.arange(1 << m); b = ((ids.unsqueeze(1) >> torch.arange(m)) & 1).float() * 2 - 1
        return torch.cat([torch.ones(len(ids), 1), b], 1) if fix_first else b
    A, B = allspins(nA, True), allspins(nB)
    JAA, JBB, JAB = J[:nA, :nA], J[nA:, nA:], J[:nA, nA:]
    EA = -0.5 * ((A @ JAA) * A).sum(1); EB = -0.5 * ((B @ JBB) * B).sum(1)
    best = math.inf
    for i in range(0, len(A), 256):
        a = A[i:i + 256]
        tot = EA[i:i + 256].unsqueeze(0) + EB.unsqueeze(1) - B @ (a @ JAB).T   # [|B|, chunk]
        best = min(best, float(tot.min()))
    return best

rng = np.random.default_rng(4242); insts = []; t0 = time.time()
for N in [24, 32, 44]:
    for kind in ["dense", "sparse"]:
        made = rej = 0
        while made < 30:
            bits = int(rng.integers(3, 9)); J = rq.make_instance(kind, N, bits, rng)
            if N <= 32:
                Eref = exact_mitm(J); exact = True
            else:   # best known from a long SA campaign (refined later with all solver samples)
                s = simulated_annealing(J, torch.zeros(N), runs=500, sweeps=4000, seed=int(rng.integers(1 << 30)))
                Eref = float(energy(s, J, torch.zeros(N)).min()); exact = False
            v = torch.linalg.eigh(J).eigenvectors[:, -1]; sv = torch.sign(v) + (v == 0).float()
            if float(energy(sv, J, torch.zeros(N))) <= Eref + 1e-6: rej += 1; continue
            insts.append(dict(N=N, kind=kind, bits=bits, J=J, Eref=Eref, exact=exact)); made += 1
        print(f"N={N} {kind}: 30 kept, {rej} rejected by eigenvector filter ({time.time()-t0:.0f}s)", flush=True)
        torch.save(insts, "step2_instances.pt")
print("done", flush=True)
