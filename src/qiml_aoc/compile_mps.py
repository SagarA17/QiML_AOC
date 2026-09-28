"""Compile a tapered MPS onto the AOC digital twin (fused vertical+horizontal contraction).

Layout (48 physical channels):
  E1 : D_1 carrier channels, biased at 0, carry eps * l~_1 (injected; l_1 = A_1(x_1) is input x weight)
  G_k: for k = 2..N, 2*D_k gate channels (s, j), biased at b_s(x_k), driven optically by the
       previous block through the static matrix omega * A_k[s]  (fused vertical contraction).
Small-signal: F(b + u) - F(b) ~= F'(b) u = FP0 * phi_s(x) * u  -> the gate multiplies by the feature.
Readout: z* of the last block (2 channels), f^ = sum_s phi_s(x_N) (z*_s - b_s) / eps.
"""
import torch
from common import (ALPHA, BONDS, D_LOC, FP0, F_neg, F_pos, N_CH, N_SITES, NEG_RATIO,
                    biases, features, make_cell)


PLACEMENT = {"idle": []}      # physical channels left unused (e.g. dead/weak rows)


def layout(bonds=BONDS, d=D_LOC):
    """Return physical channel index tensors: e1 [D1], gates[k] -> [d, D_k] (0-based site k)."""
    phys = torch.tensor([i for i in range(N_CH) if i not in PLACEMENT["idle"]])
    idx, nxt = {}, 0
    idx["e1"] = phys[nxt:nxt + bonds[1]]; nxt += bonds[1]
    idx["gates"] = {}
    for k in range(1, len(bonds) - 1):           # 0-based site k = 1..N-1  (sites 2..N)
        Dk = bonds[k + 1]
        idx["gates"][k] = phys[nxt:nxt + d * Dk].reshape(d, Dk); nxt += d * Dk
    assert nxt <= len(phys), f"needs {nxt} channels > {len(phys)} available"
    idx["used"] = nxt
    idx["used_ch"] = phys[:nxt]
    return idx


class AOCMPS(torch.nn.Module):
    def __init__(self, cores, r, eps, cell, n_iter=100):
        """cores: list of [d, Dl, Dr] (gauge-normalised); r: per-site env scales; eps: signal [V]."""
        super().__init__()
        self.cores = torch.nn.ParameterList([torch.nn.Parameter(A.clone()) for A in cores])
        self.register_buffer("r", r.clone())
        self.eps = eps
        self.cell = cell
        self.n_iter = n_iter
        self.idx = layout()
        self.a = torch.nn.Parameter(torch.tensor(1.0))   # digital readout (refit)
        self.c = torch.nn.Parameter(torch.tensor(0.0))
        self.noise_std = 0.0                             # per-iteration Gaussian noise [V]
        self.signal_on = True
        self.exact_dc = False                            # use calibrated loop model G(b) for DC
        self.init_at_target = False                      # open-loop pre-settle to operating point
        self.n_avg = 0                                   # sampling window (iterations averaged)
        self.register_buffer("K", torch.zeros(N_CH, 2 * 2 * N_SITES + 1))   # DC calibration
        # calibrated closed-loop correction (per channel, known from calibration yaml)
        self.S = cell.scaling.detach().clone()
        self.dc = cell.dc_offset.detach().clone()
        self.lin = cell.linear_offset.detach().clone()
        self.ratio = float(cell.solution_to_adc_ratio)
        self.row_equalise = False                        # divide SLM rows by input efficiency
        self.eff_match = False                           # scale group rows to floor weight E_s (exact)
        self.static_dc = False                           # one fixed offset vector, no per-event DC work
        self.static_probe = None                         # probe inputs used to calibrate that vector
        self.gain_cal = False                            # invert per-column SLM response + output eff.
        self.floor_cancel = False                        # program EVERY pixel: exact weights, net-zero floor
        self.xtalk_comp = True                           # pre-compensate photodetector crosstalk

    # ---------------- matrix -----------------
    def build_W(self):
        W = torch.zeros(N_CH, N_CH)
        omega = 1.0 / (2.0 * self.S * FP0)                # per destination column
        e1, G = self.idx["e1"], self.idx["gates"]
        for k in range(1, N_SITES):
            A = self.cores[k]                              # [d, D_{k-1}, D_k] (0-based site k)
            src = e1.unsqueeze(0) if k == 1 else G[k - 1]  # [n_src_groups, D_{k-1}]
            for s in range(D_LOC):
                dst = G[k][s]                              # [D_k]
                blk = A[s] * omega[dst].unsqueeze(0)       # [D_{k-1}, D_k]
                if not self.floor_cancel:
                    blk = torch.where(blk < 0, blk * NEG_RATIO, blk)
                for sg in range(src.shape[0]):
                    b_sg = blk
                    if self.row_equalise and self.cell._apply_input_efficiencies:
                        ie = self.cell.hardware_parameters.input_efficiency       # [2, 48] pos/neg
                        iep, ien = ie[0][src[sg]].unsqueeze(1), ie[1][src[sg]].unsqueeze(1)
                        b_sg = torch.where(blk > 0, blk / iep, blk / ien)
                        if self.eff_match:
                            kap = self._kappa()
                            if k >= 2:                      # source is a gate group: scale by kappa
                                b_sg = b_sg * kap[k - 1, sg]
                            if k <= N_SITES - 2:            # dest group output is scaled downstream
                                b_sg = b_sg / kap[k, s]
                    rows = src[sg].unsqueeze(1).expand(-1, dst.numel())
                    cols = dst.unsqueeze(0).expand(src.shape[1], -1)
                    W = W.index_put((rows.reshape(-1), cols.reshape(-1)), b_sg.reshape(-1), accumulate=True)
        return W

    def _kappa(self):
        """kappa[k, s] = E_s / mean(E_1, E_2): group floor weight (pos-path input efficiency)."""
        ie = self.cell.hardware_parameters.input_efficiency[0]
        kap = torch.ones(N_SITES, D_LOC)
        for k, g in self.idx["gates"].items():
            E = torch.stack([ie[g[s]].sum() for s in range(D_LOC)])
            kap[k] = E / E.mean()
        return kap

    def program_full(self, M):
        """Choose the SLM pattern for ALL 48x48 pixels so that the effective net transmission of
        pixel (i, j) -- positive half minus negative half, including floors, per-column response,
        input efficiency of row i and output efficiency of column j -- equals the target M[i, j]
        (in positive-path units). Unused pixels get M = 0, i.e. their floors cancel."""
        hp, cl = self.cell.hardware_parameters, self.cell
        one = torch.ones(N_CH)
        iep = hp.input_efficiency[0] if cl._apply_input_efficiencies else one
        ien = hp.input_efficiency[1] if cl._apply_input_efficiencies else one
        oep = hp.output_efficiency_pos if cl._apply_output_efficiencies else one
        oen = hp.output_efficiency_neg if cl._apply_output_efficiencies else one
        P = iep.unsqueeze(1) * oep.unsqueeze(0)                   # gain of positive half
        Q = ien.unsqueeze(1) * oen.unsqueeze(0) / NEG_RATIO       # gain of negative half (pos units)
        P, Q = torch.clamp(P, min=1e-3), torch.clamp(Q, min=1e-3)  # dead rows emit no light anyway
        if cl._apply_weight_distortion:
            ap, bp, cp = (hp.weight_distortion_pos[:, i] for i in range(3))
            an, bn, cn = (hp.weight_distortion_neg[:, i] for i in range(3))
        else:
            ap = an = torch.zeros(N_CH); bp = bn = one; cp = cn = torch.zeros(N_CH)

        def xmat(pd):
            to_lower, to_upper = pd[:, 1], pd[:, 0]
            X = torch.eye(N_CH); i = torch.arange(N_CH - 1)
            X[i + 1, i] = to_lower[1:]            # column j receives from j+1
            X[i, i + 1] = to_upper[:-1]           # column j+1 receives from j
            return X
        if cl._apply_pd_crosstalk and self.xtalk_comp:
            Xp, Xn = xmat(hp.pd_cross_talk_pos), xmat(hp.pd_cross_talk_neg)
        else:
            Xp = Xn = torch.eye(N_CH)
        Xb = 0.5 * (Xp + Xn); Xb_inv = torch.linalg.inv(Xb)

        def inv(T, a, b, c):          # transmission a w^2 + b w + c = T  ->  w >= 0
            y = torch.clamp(T - c, min=0.0)
            lin = y / b
            quad = (-b + torch.sqrt(torch.clamp(b * b + 4 * a * y, min=1e-12))) / (2 * a.where(a != 0, torch.ones_like(a)))
            return torch.where(a != 0, quad, lin)

        def solve(amax):
            """Per row i, effective map after crosstalk:
               E_i = (T+_i * P_i) Xp - (T-_i * Q_i) Xn  with T = floor c + programmed increment t >= 0.
               Want E = M / amax, i.e. floors cancelled AFTER each path's own detector crosstalk."""
            m = M / amax
            R = m - (cp.unsqueeze(0) * P) @ Xp + (cn.unsqueeze(0) * Q) @ Xn   # target minus floor light
            d = R @ Xb_inv
            for _ in range(12):                                              # small path mismatch
                tp, tn = torch.relu(d) / P, torch.relu(-d) / Q
                corr = (tp * P) @ (Xp - Xb) - (tn * Q) @ (Xn - Xb)
                d = (R - corr) @ Xb_inv
            tp, tn = torch.relu(d) / P, torch.relu(-d) / Q
            wp = torch.where(d > 0, inv(cp.unsqueeze(0) + tp, ap, bp, cp), torch.zeros_like(m))
            wn = torch.where(d > 0, torch.zeros_like(m), inv(cn.unsqueeze(0) + tn, an, bn, cn))
            return wp - wn
        amax = M.abs().max().detach() / torch.clamp(torch.minimum(bp.min(), bn.min()), min=0.3)
        for _ in range(40):                                          # self-consistent: max |w| == 1
            amax = solve(amax).abs().max().detach() * amax
        return solve(amax) * amax

    def program_slm(self, Wt):
        """Static gain calibration: find SLM pattern Wp whose *effective* weight (after the twin's
        per-column response a w^2 + b w, and output efficiency) equals target Wt. Floor c is untouched."""
        hp = self.cell.hardware_parameters
        if not self.gain_cal:
            return Wt
        oe_p = hp.output_efficiency_pos if self.cell._apply_output_efficiencies else torch.ones(N_CH)
        oe_n = hp.output_efficiency_neg if self.cell._apply_output_efficiencies else torch.ones(N_CH)
        Wt = torch.where(Wt > 0, Wt / oe_p, Wt / oe_n)
        if not self.cell._apply_weight_distortion:
            return Wt
        ap, bp = hp.weight_distortion_pos[:, 0], hp.weight_distortion_pos[:, 1]
        an, bn = hp.weight_distortion_neg[:, 0], hp.weight_distortion_neg[:, 1]
        a = torch.where(Wt > 0, ap, an); b = torch.where(Wt > 0, bp, bn)
        def solve(amax):
            y = Wt.abs() / amax                              # desired  a w^2 + b w  (normalised)
            return (-b + torch.sqrt(torch.clamp(b * b + 4 * a * y, min=1e-12))) / (2 * a)
        amax = Wt.abs().max().detach() / b.min()
        for _ in range(40):                                  # fixed point: max programmed w == 1
            amax = (solve(amax).max().detach() * amax)
        w = solve(amax)
        Wp = torch.sign(Wt) * w * amax
        return Wp

    # ---------------- per-event injection -----------------
    def operating_points(self, x):
        """Per-channel bias b [B, 48] and features phi [B, N, d]."""
        B = x.shape[0]
        bb = biases(x)                                     # [B, N, 2]
        b = torch.zeros(B, N_CH)
        for k, g in self.idx["gates"].items():
            for s in range(D_LOC):
                b[:, g[s]] = bb[:, k, s].unsqueeze(1)
        return b, features(x)

    def injection(self, x, W):
        b, phi = self.operating_points(x)
        B = x.shape[0]
        # l~_1 = A_1(x_1) / r_1   (digital, input x weight)
        l1 = torch.einsum("bs,slr->blr", phi[:, 0], self.cores[0])[:, 0, :]     # [B, D1]
        target = b.clone()
        target[:, self.idx["e1"]] = self.eps * l1 if self.signal_on else 0.0
        # known DC contribution of source operating points through the matrix
        Wp, Wn = torch.relu(W), torch.relu(-W)
        const = F_pos(b) @ Wp - F_neg(b) @ Wn                                   # [B, 48]
        off = self.ratio * (self.dc + W.abs().max() * self.lin)
        if self.static_dc:
            # calibrated ONCE per programmed matrix: mean zero-signal loop response over probe events
            bp, _ = self.operating_points(self.static_probe)
            c_static = (self.cell(bp, x=torch.zeros_like(bp)) - ALPHA * bp).mean(0, keepdim=True)
            c = (1 - ALPHA) * target - c_static
        elif self.exact_dc:
            # G(b) = loop response at the zero-signal operating point (calibrated model = twin)
            zero = torch.zeros_like(b)
            G = self.cell(b, x=zero) - ALPHA * b
            c = (1 - ALPHA) * target - G
        else:
            c = (1 - ALPHA) * target - self.S * (const + off)
            c = c - (1 - ALPHA) * (self.dc_regressors(x) @ self.K.T)   # learned-by-probing DC fix
        return c, b, phi

    def dc_regressors(self, x):
        """[F_pos(b_ks), F_neg(b_ks) for all sites/features, 1] -- only per-site scalars."""
        bb = biases(x).reshape(x.shape[0], -1)
        return torch.cat([F_pos(bb), F_neg(bb), torch.ones(x.shape[0], 1)], dim=-1)

    @torch.no_grad()
    def calibrate_dc(self, x_probe, rounds=3, ridge=1e-8, verbose=True):
        """Zero the signal path, measure fixed-point offsets from intended bias, fit linear DC model."""
        self.signal_on = False
        for rd in range(rounds):
            W = self.program_slm(self.build_W()); self.cell.matrix = W
            c, b, _ = self.injection(x_probe, W)
            z = torch.clip(c, *self.cell.hardware_parameters.deq_input_min_max)
            for _ in range(self.n_iter):
                z = self.cell(z, x=c)
            err = z - b                                            # intended: z* = b (u = 0)
            mask = torch.zeros(N_CH, dtype=torch.bool); mask[self.idx["used_ch"]] = True
            err[:, ~mask] = 0
            R = self.dc_regressors(x_probe)
            dK = torch.linalg.solve(R.T @ R + ridge * torch.eye(R.shape[1]), R.T @ err).T
            self.K += dK
            if verbose:
                print(f"   DC-cal round {rd}: rms offset before fit {err[:, :self.idx['used']].pow(2).mean().sqrt()*1e3:.3f} mV")
        self.signal_on = True

    # ---------------- run the twin -----------------
    def run(self, x, return_state=False):
        W = self.program_full(self.build_W()) if self.floor_cancel else self.program_slm(self.build_W())
        self.cell.matrix = W
        c, b, phi = self.injection(x, W)
        lo, hi = self.cell.hardware_parameters.deq_input_min_max
        z = b.clone() if self.init_at_target else torch.clip(c, lo, hi)
        self.clip_frac = float(((c < lo) | (c > hi)).float().mean())
        for _ in range(self.n_iter):
            z = self.cell(z, x=c)
            if self.noise_std > 0:
                z = z + self.noise_std * torch.randn_like(z)
        last = self.idx["gates"][N_SITES - 1][:, 0]        # [d] channels of final block
        if self.n_avg > 0:                                  # sampling window after settling
            acc = torch.zeros_like(z[:, last])
            for _ in range(self.n_avg):
                z = self.cell(z, x=c)
                if self.noise_std > 0:
                    z = z + self.noise_std * torch.randn_like(z)
                acc = acc + z[:, last]
            u_last = acc / self.n_avg - b[:, last]
        else:
            u_last = z[:, last] - b[:, last]                # [B, d]
        fhat = (phi[:, -1] * u_last).sum(-1) / self.eps     # ~ l~_N
        if return_state:
            return fhat, z, b
        return fhat

    def forward(self, x):
        fhat = self.run(x)
        return self.a * fhat + self.c, fhat


def gauge_normalise(mps, Ptr):
    """Rescale cores so that the 99th percentile of |l_k| is 1 at every site."""
    with torch.no_grad():
        envs = mps.envs(Ptr)
        r = torch.stack([torch.quantile(l.abs().flatten(), 0.99) for l in envs])
        cores, prev = [], torch.tensor(1.0)
        for k, A in enumerate(mps.cores):
            cores.append(A.detach() * prev / r[k])
            prev = r[k]
    return cores, r


def choose_eps(cores, Ptr, u_max):
    """eps such that the 99th pct of the pre-gate drive |A_k[s]^T l~_{k-1}| maps to u_max [V]."""
    with torch.no_grad():
        l = torch.einsum("bs,slr->blr", Ptr[:, 0], cores[0])[:, 0, :]
        q = [torch.quantile(l.abs().flatten(), 0.99)]
        for k in range(1, len(cores)):
            drive = torch.einsum("bl,slr->bsr", l, cores[k])
            q.append(torch.quantile(drive.abs().flatten(), 0.99))
            l = torch.einsum("bs,bsr->br", Ptr[:, k], drive)
        return float(u_max / torch.stack(q).max())


def site_efficiency_ratios(cell):
    """rho[k] = E1/E2: summed input efficiency (pos path) of feature group 1 vs 2 at each site.
    Site 1 (carriers, no gates) gets 1."""
    idx = layout(); ie = cell.hardware_parameters.input_efficiency[0]
    rho = torch.ones(N_SITES)
    for k, g in idx["gates"].items():
        rho[k] = ie[g[0]].sum() / ie[g[1]].sum()
    return rho
