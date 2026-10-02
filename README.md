# QiML_AOC — quantum-inspired ML and optimisation on the Analog Optical Computer (digital twin)

Exploratory study of mapping **quantum-inspired, HEP-motivated workloads** onto Microsoft's Analog Optical
Computer (AOC; Kalinin et al., *Nature* 645, 354 (2025)) using its open-source digital twin
([microsoft/aoc](https://github.com/microsoft/aoc)). Two workloads, one device:

- **Inference:** tensor-network classifiers (tapered MPS) compiled onto the AOC loop.
- **Optimisation:** QUBO / Ising problems, including a toy primary-vertex (PV) finder, solved with
  **simulated bifurcation** (quantum-inspired; Goto et al.) mapped natively onto the AOC update rule.

Everything here runs on the **digital twin only** (no hardware). The twin has no noise model; all
simulations are noise-free unless explicitly stated. This is a proof-of-concept / co-design study, not a
performance claim against classical baselines.

---

## Setup

The digital twin is a git submodule at `external/aoc` (pinned to upstream commit `92d9014`).

```bash
git clone --recurse-submodules <this repo>      # or, in an existing clone: git submodule update --init
conda create -p <envs>/QiML_AOC python=3.11 pip && conda activate <envs>/QiML_AOC
pip install -r requirements.txt
pip install -e external/aoc
```

Note: `pip install aoc` from PyPI installs an **unrelated** package; always install the twin from the submodule.
The twin's `AOCCell.matrix` / `AOCCell.bias` setters are ignored by the forward pass upstream; we patch
this in `src/qiml_aoc/common.py` (`PatchedAOCCell`) without changing any physics.

## Layout

```
src/qiml_aoc/          reusable modules
  common.py            patched twin cell, measured nonlinearity F, hardware-native feature map, data, TaperedMPS
  compile_mps.py       MPS -> 48-channel AOC compiler (layout, bias gating, full-pixel SLM programming, DC calibration)
  ising.py             Ising/QUBO instances, SA baseline, AOC-optimiser dynamics, twin mapping, field calibration
  sb.py                simulated bifurcation (dSB/bSB): reference + AOC-twin mapping
  sb_tabu.py           tabu-enhanced dSB: paper-exact, hardware-compatible (per-sample field), twin
  pv_toy.py            toy 1D PV windows, QUBO formulation, exact solver, deterministic-annealing baseline
  qubo_instances.py    paper-protocol QUBO instance generator (16 variables)
experiments/           scripts, grouped by study (each writes/reads artefacts in results/)
  mps/                 MPS compile, diagnosis, fine-tuning, floor/crosstalk fixes, resources
  qubo_replication/    replication of the paper's 16-variable hardware QUBO benchmark
  qubo_scaling/        paper family at N = 24/32/44 (step 2) and rugged Wishart (step 2.5)
  simulated_bifurcation/  dSB and tabu-enhanced dSB vs previous solvers on rugged instances
  vertexing/           toy PV finding on the twin
  early_exploration/   first QUBO attempts (kept for the record; superseded)
results/               saved results (.pt) used by the report scripts
scripts/fetch_datasets.py  download + md5-verify Zenodo datasets into $QIML_AOC_DATA (default: eagle project dir)
```

Scripts can be run from anywhere: each adds `src/qiml_aoc` to the path and works inside `results/`.
Long runs (`*_run.py`) are resumable and save after each instance.

---

## 1. Tensor networks on the AOC (tapered MPS)

**Construction.** The AOC only computes static-weight x signal (optics) + elementwise nonlinearity. An MPS
needs hidden x input products. We fuse vertical and horizontal contractions into
`l_k = sum_s phi_s(x_k) W_{k,s}^T l_{k-1}` and implement the input-dependent gain via the **slope of the
nonlinearity at an input-set bias**: `F(b+u) - F(b) ~ F'(b) u`. Local feature map is hardware-native:
`phi_s(x) = F'(b_s(x))/F'(0)` using the twin's measured curve.

**Task.** sklearn digits "0-4 vs 5-9", 8 PCA features, tapered bonds `[1,2,3,4,4,4,3,2,1]` (144 params),
44/48 channels. Baselines: logistic regression 0.787, MLP(32) 0.950.

| Setting | Test acc. |
|---|---|
| Ideal tapered MPS (float) | ~0.92 |
| Twin, ideal loop (real nonlinearity + calibrated offsets) | 0.922 (exact to 2nd order) |
| Full twin, naive compile | 0.65 |
| Full twin, final static compile, **zero-shot** | 0.902–0.904 |
| Full twin, final compile + hardware-aware fine-tuning | 0.917 |

**What it took (all static, no per-event calibration):**
1. *Seesaw-balanced features* — choose the two biases so total brightness per site is input-independent.
2. *Channel placement* — leave dead/weak rows idle (channel 8 has zero efficiency).
3. *Full-pixel SLM programming* — program **every** pixel so "zero" pixels pass net-zero light
   (cancels the SLM darkness floor, which otherwise leaks ~-190 mV into every channel), with per-row input
   efficiency, per-column output efficiency, SLM response curve and pos/neg path ratio folded in.
4. *Floor cancellation after photodetector crosstalk* — balance pos/neg paths *after* each path's own
   neighbour crosstalk (the last residual).
5. One static offset vector calibrated once per programmed pattern.

Event-to-event DC wobble went 17 mV -> 0.04 mV. Per-event digital work: ~18 lookups, ~7 MACs, 44 DAC
writes, 2 reads (vs ~190 MACs for the MPS in software). Settling: <= 36 iterations (~0.7 us at 20 ns/iter,
paper figure; twin has no timing model). No energy numbers (twin has no power model).

## 2. QUBO on the AOC

**Replication of the paper's hardware benchmark** (16 variables, 3–8 bit, dense SK + sparse 3-regular,
principal-eigenvector filter, samples-to-optimum). Full twin, new programming, **residual-field
calibration** (static, per problem) + small per-instance schedule scan: **100/100 instances solved**,
worst case 152 samples (paper: < 1000). The residual field (asymmetric saturated LED levels) was the
dominant hardware error; couplings were accurate to ~0.5%.

**Scaling on the paper family (N = 24/32/44, 20+20 instances each).** Full twin tracks the no-hardware twin;
with the scan all 120 instances solved (one at 1065 samples). Misses concentrate on 8-bit instances.

**Rugged Wishart planted instances (step 2.5).** Moderate ruggedness (M/N = 0.75): all solved up to N = 44.
Hard (M/N = 0.5): the AOC-optimiser algorithm degrades at N = 44 (full twin 0 -> 4/15 with scan).

**Simulated bifurcation (dSB).** With Delta t = 1, dSB *is* the AOC update rule (pump = annealing, c0 = beta,
momentum gamma, saturation = inelastic walls). Untuned (Goto's defaults), gamma = 0.95 on the twin:

| M/N | N | AOC-optimiser, full twin | **dSB, full twin** | dSB ideal | SA |
|---|---|---|---|---|---|
| 0.5 | 24 | 13/15 | **15/15** | 15/15 | 15/15 |
| 0.5 | 32 | 10/15 | **15/15** | 15/15 | 15/15 |
| 0.5 | 44 | 0/15 | **6/15** | 10/15 | 14/15 |

(1000 samples each.) SA remains stronger per run; comparisons are per sample, not per unit compute.

**Tabu-enhanced dSB (TESB; Tao et al., Commun. Phys. 9, 100 (2026)).** The tabu penalty
(c0 beta / 2|Mb|) sum_s |x + s|^2 expands into a pump shift a -> a - c0 beta plus a *linear field*, so it maps onto
the AOC without SLM reprogramming: warm-up dSB runs build a tabu list (minus the best warm-up states), and checking
runs get the field through the injection (applied after field calibration). Implementation follows the paper's
equations; note that the authors' released code omits the pump shift. Variants: mini-batch re-drawn every iteration
and shared by all samples (paper), or drawn once per sample (static per-run injection). The paper's beta = 1 is
calibrated to Max-Cut coupling drives (~10-20 per spin); ours are ~1.3, so beta ~ 0.05-0.1 is used. Same ~1M loop
iterations per instance as dSB; readout window 40 for all. Cells: instances solved at least once (mean per-run
success probability):

| N | dSB ideal | TE paper eqs. | TE no shift | TE per-sample | dSB twin | TE per-sample, twin | TE per-iter., twin | SA |
|---|---|---|---|---|---|---|---|---|
| 32 | 15/15 (.0155) | 15/15 (.0157) | 15/15 (.0137) | 15/15 (.0127) | 15/15 (.0071) | 15/15 (.0091) | 15/15 (.0084) | 15/15 (.0338) |
| 44 | 10/15 (.0011) | 12/15 (.0014) | 11/15 (.0015) | 7/15 (.0013) | 6/15 (.0009) | 4/15 (.0005) | 6/15 (.0007) | 14/15 (.0035) |

The solved-instance count is dominated by sampling noise at ~0.1% per-run success: re-drawing the random numbers
of an otherwise identical prototype run moved the per-sample variant from 13/15 to 7/15 at an unchanged mean
success rate. On the per-run success rate, TESB gains at most ~1.3x over dSB on these small instances (idealised),
and the static per-sample field does not help on the twin. SA remains ~3x better per run. The paper's large gains
are on N = 800-2000 Max-Cut; the benefit on our problem sizes has to be measured per application.

## 3. Toy primary-vertex finding

1D windows: 11 tracks, 2–4 vertices in 2 mm (>= 0.15 mm apart), z0 resolution 20–120 um. QUBO over
x_ik (11 tracks x 4 slots = 44 vars): pull-based pair rewards/penalties + one-hot penalty; linear terms
folded into the matrix via a reference spin (45 channels). 100 windows:

| Method | Samples | QUBO optimum | Mean ARI | Perfect windows |
|---|---|---|---|---|
| Exact QUBO optimum (ceiling) | – | – | 0.858 | 63 |
| Deterministic annealing (classical) | – | – | 0.870 | 68 |
| dSB, full twin | 1 / 10 / 100 | 22 / 82 / 100 | 0.616 / 0.837 / 0.858 | 18 / 56 / 63 |
| dSB, idealised | 1 / 10 | 60 / 97 | 0.799 / 0.855 | 49 / 64 |
| SA | 1 / 10 | 56 / 99 | 0.826 / 0.857 | 45 / 62 |

Takeaways: deployment works (optimum always reached at 100 samples); the **formulation**, not the solver,
limits physics quality (DA beats the QUBO optimum); the twin needs ~10x more samples than idealised dSB
(9% of single samples violate one-hot). ARI is a placeholder — proper vertexing metrics are next (see AGENTS.md).

---

## Caveats

- Digital twin only; its fidelity is established by the AOC team for their operating regimes, not for
  deliberately saturated-tanh gating or our full-pixel programming. Hardware validation is required.
- No noise in simulation (twin has no noise model).
- Several calibrations assume open-loop probe measurements are possible on hardware (as used in the
  paper's MVM calibration).
- Toy datasets throughout (digits, synthetic QUBOs, toy PV windows).
- Throughput, not latency, is the likely constraint for event-filter-style use (see AGENTS.md).

## References

Kalinin et al., *Analog optical computer for AI inference and combinatorial optimization*, Nature 645 (2025).
Kalinin et al., *Analog Iterative Machine (AIM)*, arXiv:2304.12594. Goto et al., *Sci. Adv.* 5 (2019) & 7 (2021).
Tao et al., Tabu-enhanced simulated bifurcation, Commun. Phys. 9, 100 (2026).
Okawa et al., Quantum-annealing-inspired algorithms for track reconstruction, Comput. Softw. Big Sci. 8, 16 (2024).
Hamze et al., "Wishart planted ensemble: a tunably rugged pairwise Ising model". Das et al., track clustering with a quantum annealer (2019).
