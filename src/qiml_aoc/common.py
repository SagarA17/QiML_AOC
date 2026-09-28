"""Shared pieces: twin nonlinearity, hardware-native feature map, data, tapered MPS."""
import warnings
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)

from aoc import AOCCell, MatrixConnectivityType  # noqa: E402


class PatchedAOCCell(AOCCell):
    """Fix: upstream `matrix` property ignores an externally set matrix inside
    get_distorted_weight_matrix() and closed_loop_correction(). Physics unchanged."""

    @property
    def matrix(self):
        if self._externally_set_matrix is not None:
            return self._externally_set_matrix
        return self.matrix_structure.build_aoc_matrix()

    @matrix.setter
    def matrix(self, m):
        self._externally_set_matrix = m

    @property
    def bias(self):
        # same issue: forward() reads the internal nn.Linear bias; we inject everything via x
        if self._externally_set_bias is not None:
            return self._externally_set_bias
        return self.matrix_structure.build_aoc_bias()

    @bias.setter
    def bias(self, b):
        self._externally_set_bias = b

N_CH = 48          # physical channels on the 2300-weight machine
ALPHA = 0.5        # paper / repo default
B_MAX = 0.45       # max gate bias [V] -> gain range ~[0.06, 1]
BONDS = [1, 2, 3, 4, 4, 4, 3, 2, 1]   # tapered, D_0..D_N  (N = 8 sites)
N_SITES = len(BONDS) - 1
D_LOC = 2


def make_cell(**flags) -> AOCCell:
    """48-channel AOCCell with externally set matrix; flags toggle non-idealities."""
    kw = dict(apply_weight_distortion=True, apply_input_efficiencies=True,
              apply_output_efficiencies=True, apply_pd_crosstalk=True,
              apply_pbs_crosstalk=False, apply_slm_darkness=True)
    kw.update(flags)
    cell = PatchedAOCCell.from_parameters([N_CH, N_CH], connectivity=MatrixConnectivityType.FEEDBACK,
                                          normalise_matrix=False, alpha=ALPHA, **kw)
    cell.bias = torch.zeros(N_CH)
    return cell


_REF = make_cell()


def F_pos(z):
    """Effective nonlinearity seen through positive-weight SLM path (LED+ o tanh)."""
    return _REF._uled_nonlinearity(_REF._aoc_tanh(z))[0]


def F_neg(z):
    return _REF._uled_nonlinearity(_REF._aoc_tanh(z))[1]


def _deriv(fn, z):
    z = z.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(fn(z).sum(), z, create_graph=False)
    return g


FP0 = float(_deriv(F_pos, torch.zeros(1)))          # F_pos'(0)
FN0 = float(_deriv(F_neg, torch.zeros(1)))
NEG_RATIO = FP0 / FN0                                # ~0.95, absorbed into W-


def gamma(b):
    """Normalised small-signal gain at operating point b: F_pos'(b)/F_pos'(0). Differentiable."""
    with torch.enable_grad():
        zb = b if b.requires_grad else b.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(F_pos(zb).sum(), zb, create_graph=True)
    return g / FP0


FEATURE_MODE = {"mode": "complement", "rho": None}   # rho[k] = E1/E2 group efficiency ratio per site      # "complement": b2 = B(1-x) ; "balanced": F(b1)+F(b2) = const
_ZS = torch.linspace(0.0, 0.7, 7001)
with torch.no_grad():
    _FS = F_pos(_ZS)
assert bool((_FS[1:] > _FS[:-1]).all()), "F_pos must be monotonic on the LUT range"


def F_pos_inv(t):
    """Inverse of F_pos on [0, 0.7] V by linear interpolation (lookup table)."""
    i = torch.clamp(torch.searchsorted(_FS, t.contiguous()), 1, len(_FS) - 1)
    f0, f1, z0, z1 = _FS[i - 1], _FS[i], _ZS[i - 1], _ZS[i]
    return z0 + (t - f0) * (z1 - z0) / (f1 - f0)


def set_feature_mode(mode, rho=None):
    FEATURE_MODE["mode"] = mode
    FEATURE_MODE["rho"] = rho


def biases(x):
    """Per-site gate biases [V] for the two local features. x in [0,1], shape [B, N]."""
    b1 = B_MAX * x
    if FEATURE_MODE["mode"] == "balanced":
        # seesaw: F(b1) + F(b2) = F(B_MAX) + F(0) for every x -> total light per site constant
        with torch.no_grad():
            rho = FEATURE_MODE["rho"] if FEATURE_MODE["rho"] is not None else torch.ones(b1.shape[-1])
            F0, FB = F_pos(torch.tensor(0.0)), F_pos(torch.tensor(B_MAX))
            # efficiency-aware seesaw E1 F(b1) + E2 F(b2) = const, both channels kept reachable:
            # E1 (F(b1)-F0) = t S, E2 (F(b2)-F0) = (1-t) S, S = min(E1,E2)(FB-F0), t = x
            e1 = torch.clamp(rho, max=1.0); e2 = torch.clamp(1.0 / rho, max=1.0)   # min(E1,E2)/E1, /E2
            b1 = F_pos_inv(F0 + x * e1 * (FB - F0))
            b2 = F_pos_inv(F0 + (1.0 - x) * e2 * (FB - F0))
        return torch.stack([b1, b2], dim=-1)
    return torch.stack([b1, B_MAX * (1.0 - x)], dim=-1)   # [B, N, 2]


def features(x):
    """Hardware-native local feature map phi_s(x) = gamma(b_s(x)). Shape [B, N, 2]."""
    return gamma(biases(x)).detach()


def load_data(seed=0, n_pc=N_SITES):
    X, y = load_digits(return_X_y=True)
    lab = (y >= 5).astype(np.float32)                 # 0-4 vs 5-9
    Xtr, Xte, ytr, yte = train_test_split(X, lab, test_size=0.3, random_state=seed, stratify=lab)
    p = PCA(n_pc, random_state=seed).fit(Xtr)
    Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    lo, hi = np.percentile(Ztr, 1, axis=0), np.percentile(Ztr, 99, axis=0)
    f = lambda Z: np.clip((Z - lo) / (hi - lo), 0, 1).astype(np.float32)
    return (torch.tensor(f(Ztr)), torch.tensor(ytr), torch.tensor(f(Zte)), torch.tensor(yte))


class TaperedMPS(torch.nn.Module):
    """Scalar-output MPS f(x) = A_1(x_1) ... A_N(x_N), A_k(x) = sum_s phi_s(x) A_k[s]."""

    def __init__(self, bonds=BONDS, d=D_LOC, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.bonds = bonds
        self.cores = torch.nn.ParameterList()
        for k in range(len(bonds) - 1):
            Dl, Dr = bonds[k], bonds[k + 1]
            eye = torch.zeros(d, Dl, Dr)
            for s in range(d):
                eye[s, :min(Dl, Dr), :min(Dl, Dr)] = torch.eye(min(Dl, Dr))
            self.cores.append(torch.nn.Parameter(eye + 0.3 * torch.randn(d, Dl, Dr, generator=g)))
        self.a = torch.nn.Parameter(torch.tensor(1.0))
        self.c = torch.nn.Parameter(torch.tensor(0.0))

    def envs(self, phi):
        """Left environments l_k (row vectors) for k=1..N. phi: [B, N, d]."""
        B = phi.shape[0]
        l = torch.ones(B, 1)
        out = []
        for k, A in enumerate(self.cores):
            Ak = torch.einsum("bs,slr->blr", phi[:, k], A)
            l = torch.einsum("bl,blr->br", l, Ak)
            out.append(l)
        return out

    def forward(self, phi):
        f = self.envs(phi)[-1][:, 0]
        return self.a * f + self.c, f
