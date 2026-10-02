# AGENTS.md — working notes and plan

Guidance for anyone (human or agent) continuing this project. Read README.md first for results.

## Goal

A paper exploring **HEP-native, quantum-inspired workloads on AOC-like analog optical hardware**, done
entirely on the digital twin first. Framing: feasibility, mappings and hardware *requirements* for future
collider readout / event-filter systems (e.g. FCC), not beating tuned classical baselines. Only after the
twin-based study is complete will the authors contact the AOC hardware team for validation.

Theme: quantum-/physics-inspired methods on physical analog hardware —
tensor networks (inference) and simulated bifurcation (optimisation), both mapped natively onto AOC eq. (1).

## Ground rules

1. **Use only available information.** No speculative hardware numbers. The twin has no noise model ->
   simulate without noise; if noise studies are added, label them clearly as stress tests.
2. **Calibration must not cost more than the logic.** Prefer static, one-time calibrations (SLM pattern,
   offset vectors, field calibration). Report any per-event digital work explicitly.
3. **Separate measured (twin) from projected numbers.** Never quote the paper's 500 TOPS/W as ours.
4. **Always include a classical reference** (SA, deterministic annealing, FPGA-style baselines) and the
   idealised-algorithm reference, so hardware effects and algorithmic effects can be separated.
5. **Diagnose by ablation.** Switch twin non-idealities on one at a time; compare full twin vs "twin, no
   hardware effects" vs idealised algorithm before drawing conclusions.
6. Avoid over-engineering: stop adding compensation layers once the remaining gap is understood.

## Practical notes / lessons learned

- The twin is the submodule `external/aoc` (PyPI `aoc` is unrelated); bump the pin deliberately and re-check
  results. Keep `PatchedAOCCell` (external matrix/bias fix).
- Use the dedicated `QiML_AOC` conda env; never install into other existing environments.
- Per-channel calibration is applied only for 48x48 matrices — always pad to 48 channels.
- Channel 8 is dead; channels 1, 46, 3 are weakest. Leave dead/weak rows idle where possible.
- The SLM darkness floor leaks through every pixel; sparse encodings need full-pixel programming.
- Saturated spin levels are asymmetric and differ between pos/neg LED paths -> residual field; calibrate it.
- Energies from different summation orders differ by ~1e-6: use tolerance = 1/4 of the energy quantum.
- Sandbox/CI note: single commands may be time-limited; long jobs are written resumable (`*_run.py`).
- Linear QUBO terms: fold into the matrix via a reference spin (injection range is only +-1.3 V).
- TESB's beta must be scaled to the typical coupling drive |J s| per spin (paper's beta = 1 assumes ~10-20).
- Field calibration subtracts *all* measured field: calibrate first, then add intended fields (e.g. tabu).
- "Instances solved at least once" is noise-dominated at ~0.1% per-run success (an RNG re-draw moved one TESB
  variant from 13/15 to 7/15). Compare mean per-run success probability, paired per instance.
- TESB as published (Tao et al.): the authors' released code omits the pump shift a -> a - c0*beta and drops the
  best warm-up states from the tabu list; our `sb_tabu.py` follows the paper's equations, with both as options.
- Compute: run jobs through PBS (`scripts/run.pbs`, queue by-gpu, 1 GPU, walltime as short as possible);
  the login node is for smoke tests only. Data lives on eagle (`$QIML_AOC_DATA`).

## Plan (next steps, in order)

### A. Vertexing (optimisation): secondary vertices in jets
Dataset: Shlomi et al. (arXiv:2008.02831; Zenodo 4044628): 14 TeV ttbar jets, Pythia8 + Delphes (ATLAS-like),
no pile-up (state as a limitation). The toy 1D PV study (`pv_toy.py`) is superseded.
1. **Benchmark metrics exactly as Shlomi et al., Table 2**, split by jet flavour (b, c, light): jet-level F1, RI and
   ARI (their *one-sided* ARI with Bell numbers, eq. 11; NOT sklearn's), the ARI categories perfect / intermediate /
   poor, vertex and vertex-pair edge accuracies, ARI vs n_tracks and n_vertices. Reference rows: AVR, Track Pair,
   RNN, Set2Graph.
2. **QUBO formulation** for track-to-vertex assignment; report the exact QUBO optimum as the ceiling.
3. **Solver**: TESB vs plain dSB (ablation) vs SA, idealised and on the twin; single-shot / few-sample quality.
4. **Throughput budget** for the HL-LHC HLT / event filter (b-jet triggers, e.g. HH -> 4b); jets per AOC unit
   (block-diagonal packing of small jets into 48 channels).

### B. Tensor networks (inference) — move to HEP data
1. Replace digits with a HEP task: jet tagging and/or anomaly detection (SMPO -> full MPS variant).
2. Keep the hardware-native feature map and seesaw balancing; report channel footprint vs 48 channels.
3. Study signal-amplitude vs small-signal-error trade-off and the differential-pair variant.
4. Quantify how much of the fine-tuned model is still an exact MPS (effective cores vs non-MPS residual).

### C. QUBO algorithm
1. TESB is the solver of record (justified by its exact AOC mapping, not as a general SOTA claim); plain dSB is
   the "tabu off" ablation. Its benefit at our sizes is small so far (~1.3x per-run success at N = 44,
   idealised; none on the twin with a static per-sample field): measure it on the jet QUBOs, don't assume it.
2. Rescale beta to the coupling drive of each QUBO family.
3. Cheap check: longer anneals / more samples on hard N = 44 instances (hardware makes these cheap).
4. Retune at the target size rather than at N = 16.
5. Related work to cite and position against: Okawa et al. 2024 (SB for tracking), Tao et al. 2026 (TESB).

### D. Questions for the hardware team (collect, do not guess)
Noise level per channel per iteration; fidelity of the SLM floor and PD crosstalk models; feasibility of
open-loop probe calibration (offsets, residual field); injection (DAC) update rate, i.e. can the
injection change every iteration (per-iteration tabu field); is alpha < 0 possible (TESB pump shift); momentum gain range (gamma ~ 0.95); timing per
iteration and sampling protocol; power per module.
