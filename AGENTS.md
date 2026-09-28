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

- Install the twin from GitHub (PyPI `aoc` is unrelated). Keep `PatchedAOCCell` (external matrix/bias fix).
- Per-channel calibration is applied only for 48x48 matrices — always pad to 48 channels.
- Channel 8 is dead; channels 1, 46, 3 are weakest. Leave dead/weak rows idle where possible.
- The SLM darkness floor leaks through every pixel; sparse encodings need full-pixel programming.
- Saturated spin levels are asymmetric and differ between pos/neg LED paths -> residual field; calibrate it.
- Energies from different summation orders differ by ~1e-6: use tolerance = 1/4 of the energy quantum.
- Sandbox/CI note: single commands may be time-limited; long jobs are written resumable (`*_run.py`).
- Linear QUBO terms: fold into the matrix via a reference spin (injection range is only +-1.3 V).

## Plan (next steps, in order)

### A. Vertexing (optimisation) — make the physics meaningful
1. **Proper metrics**: vertex reconstruction efficiency, purity, merge and split rates, z resolution of
   reconstructed vertices; per-window and per-event. Replace ARI as headline metric.
2. **Improve the formulation** until the QUBO optimum at least matches deterministic annealing
   (current: QUBO optimum 63/100 perfect vs DA 68/100). Options: better pair function, vertex-position
   variables (QUMO; note x*v^2 is cubic -> needs alternating/BCD or a different encoding), outlier handling.
3. **Single-shot quality on the twin** (currently 22% optimum at 1 sample, 9% invalid one-hot).
   Investigate penalty scaling / dynamic range and dSB parameters; target few samples per window.
4. **Throughput budget**: windows x samples x ~20 us per event vs event-filter rates (ATLAS Phase-II EF:
   1 MHz in, 10 kHz out). Report required number of AOC units.
5. Then secondary vertices in jets (FCC-ee flavour tagging) and pile-up-dense PV windows (FCC-hh).

### B. Tensor networks (inference) — move to HEP data
1. Replace digits with a HEP task: jet tagging and/or anomaly detection (SMPO -> full MPS variant).
2. Keep the hardware-native feature map and seesaw balancing; report channel footprint vs 48 channels.
3. Study signal-amplitude vs small-signal-error trade-off and the differential-pair variant.
4. Quantify how much of the fine-tuned model is still an exact MPS (effective cores vs non-MPS residual).

### C. QUBO algorithm
1. Keep dSB (quantum-inspired, exact mapping onto the AOC update) as the default optimiser.
2. Cheap check: longer anneals / more samples on hard N = 44 instances (hardware makes these cheap).
3. Retune at the target size rather than at N = 16.

### D. Questions for the hardware team (collect, do not guess)
Noise level per channel per iteration; fidelity of the SLM floor and PD crosstalk models; feasibility of
open-loop probe calibration (offsets, residual field); momentum gain range (gamma ~ 0.95); timing per
iteration and sampling protocol; power per module.
