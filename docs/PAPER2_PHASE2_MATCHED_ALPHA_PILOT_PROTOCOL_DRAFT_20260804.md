# Phase-2 Matched Alpha Pilot Protocol

Date: 2026-08-04

Status: `locked_before_training`. The Git commit containing this file and
`training/paper2_phase2_matched_alpha_preregistration.json` is the protocol lock.
No pilot optimizer step may precede that commit.

## Question

Among shared-basis transforms with alpha in `{0.0, 0.5, 1.0}`, which scaling
produces the best verified acceptance after the complete student module is
trained under a matched DEV-only budget, without violating identity,
frozen-lineage, tube, preservation, or upper-model-quality constraints?

Experiment 0A/0B established numerical validity and geometry differences. The
layer-mode bound selected `learned_mixture_rrr`. None selected alpha.

## Fixed Arms And Invariants

- Canonicalizer: `learned_mixture_rrr`, seed `20260814`, artifact SHA-256
  `53215130a75c929def4bcc4b81ba9187d90a7b6fe2b1502391518d45c09476e3`.
- Arms: alpha `{0.0, 0.5, 1.0}` and seeds `{0, 1}`; six runs.
- Shared across arms: DEV-C rows, canonical mean, learned layer mixture, RRR
  projector, PCA orientation, retained rank, raw/effective eigenvalues,
  architecture, initialization stream, batch order, budget, evaluation rows,
  and stopping rules.
- Sole alpha-arm difference: `lambda_eff ** (-alpha / 2)` in one frozen PCA
  basis. All non-alpha state is byte-identical and asserted.
- Four future slots are populated. Four reserved trace/span slots are masked
  from every loss and effective-rank calculation.
- Recurrent-gradient paths run in fp32. Loop cap is `K <= 4`.
- V1d constants are banked: `c = 0.15`, state-RMS cap
  `0.5508932316303252`, constants LF SHA-256
  `4e56a43a6692a4c88e60c17cd5e12076f1a2f0c3c65b3027dfc3f0800ef558fc`.
  The before-training line-ending correction is recorded in
  `docs/PAPER2_PHASE2_MATCHED_ALPHA_CONSTANTS_HASH_AMENDMENT_20260804.md`.
- Frozen probe parameters: `660,480`. Student module parameters: `1,184,917`.
  Probe evidence is therefore interpreted with the recorded anti-compensation
  caveat, but the frozen probe is not refit.

## Data And Verification Surface

- DEV-C data SHA-256:
  `05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d`.
- Sample-manifest SHA-256:
  `43edbb74c5edf84dc5e6512dbe4beb1bbbf0f4df31b2c6830714ce2c8fc7ba93`.
- Position-key SHA-256:
  `4b66fce9817fe499718506e9206005c5bf3876bc999f6bb5062b36a9fbb81490`.
- The fixed evaluation slice is selected by a stable document-isolated hash,
  identically for every arm and seed, and excluded from optimization batches.
- The frozen 0.5B and 14B cached hidden states plus their hashed LM heads
  reconstruct the zero-loop drafter and teacher distributions. Verified
  single-position acceptance is distributional overlap `sum min(p, q)`.
  Mean accepted length is the sum of prefix-survival products over horizons
  one through four on teacher-forced cached prefixes. This is DEV-only
  verification, not a serving-throughput result.

## Optimizer And Budget

- Phase-A losses only: final CE `1.0`, preserve KL `0.1`, flow `1.0`,
  functional-probe KL `0.5`, cumulative KL `1.0`, amended trust `0.01`.
- AdamW, learning rate `3e-4`, weight decay `0.01`, batch `128`, steps `1,000`.
- Linear warmup for steps `1..100`; constant learning rate afterward.
- Biases, RMSNorm gains, and all learnable scalars are excluded from weight
  decay, including gate biases, magnitude-head bias, `g_k`, `rho_k`, and
  `b_{k,j}`.
- Evaluate at step zero and every 100 steps on the fixed matched DEV slice.
- Per-module clipping: refiner `1.0`, bridge `0.5`, heads `1.0`.
- The run is resumable from a per-arm checkpoint containing model, optimizer,
  scheduler position, RNG, batch-stream position, and receipt history.

## Trust Amendment And Stopping

The active trust ratio is
`RMS(delta_z) / (max(RMS(z_k), RMS(stop_gradient(Z_T))) + 1e-6)`.
The original state-referenced ratio remains telemetry. The trust ceiling is
`0.5`. If the amended trust penalty is nonzero on more than 50 of any 100
post-warmup steps, the arm halts with a receipt.

An arm also halts on a non-finite loss/state, frozen-hash mutation, assertion
failure, or upper-model-quality non-inferiority failure at two consecutive
evaluations. At the diagnostic DEV scale, preservation requires baseline-
correct top-1 retention at least `0.997` and Wilson 95% lower bound at least
`0.990`. Aborts are reported and never silently restarted.

## Adequacy Before Selection

The equivalence rule is available only if at least four of six runs satisfy
both: flow validation loss improves at least 20% from step 100, and mean
gate-open rate exceeds initialization. Otherwise all six arms resume once to
2,000 total steps. If adequacy still fails, the result returns to strategy and
alpha is not selected.

## Selection Rule

1. Exclude assertion-invalid arms.
2. Exclude arms failing quality non-inferiority; quality does not rank arms.
3. Rank qualifiers by verified mean accepted length.
4. Tie-break by flow convergence, gradient balance, then clipping burden.
5. Treat arms as equivalent when the paired bootstrap 95% interval on the
   accepted-length difference is wholly inside a relative `+/-2%` band, or
   between-seed spread exceeds the between-arm difference. Alpha `0.5` wins
   equivalence.

If alpha `1.0` beats `0.5` outside the band, add alpha `0.75` with the same
seeds and budget. If alpha `0.0` beats `0.5` outside the band, add alpha `0.25`.
No denser grid is otherwise authorized. The preregistered prediction is that
alpha `0.0` underperforms on stability and decision alignment; an acceptance
win by alpha `0.0` counts against that prediction.

## Required Telemetry

- verified overlap, accepted length, paired row-level intervals;
- upper-model/final-token quality versus zero-loop;
- flow convergence and endpoint error;
- both trust ratios, trust rent, radial drift, and update ratios by loop;
- per-module gradient norms, conflict cosines, coefficient of variation,
  clipping fractions, and gate-open rates;
- scratch effective rank on populated slots only;
- probe KL, probe top-1, and verified-acceptance correlation table;
- trained-module monotonicity: accepted updates that worsen probe KL or
  final-token quality, stratified by gate state.

## Hard Assertions And Boundaries

Zero-loop identity is bit-exact. Teacher, canonicalizer, probe, LM heads, and
pretrained backbone remain frozen and hash-identical. Targets and sampled
tokens are gradient-isolated. Every receipt records the protocol-lock commit,
constants LF hash, data/manifest hashes, canonicalizer hash, and source-cache
hashes. Packed batches remain document isolated. Masked slots never enter a
loss or rank statistic.

This pilot selects a DEV configuration only. It is not E1 confirmation, does
not establish serving speed, and does not authorize later E phases.
