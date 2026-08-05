# Paper Two Phase 2 A2 Lock Amendment

Date: 2026-08-05

Status: `locked_before_a2_training`

This amendment binds the A2 state-use experiment authorized by
`STRATEGY_TO_CODING_AGENT_A2_LOCK_RESOLUTION_20260805.md`. The commit containing
this document and the matching machine-readable registration is the lock. No A2
optimizer update may occur before that commit.

## 1. Fixed inputs

- Canonicalizer alpha: 0.5.
- Seeds: 0 and 1.
- Banked A1 endpoints: seed 0 SHA-256
  `823c1865878a86079a6423fabf432b6f1d36d431ec4381800846019882afb136`;
  seed 1 SHA-256
  `a9c20510f6cf2561f6208fa8d1915626e2ec6e68a588228d3f0edd9cd0efde89`.
- Batch size: 128. Evaluation cadence: every 100 optimizer steps.
- Nominal budget: 1,000 steps. Maximum budget after the single registered
  extension: 2,000 steps.
- Optimizer: AdamW, learning rate 3e-4, weight decay 0.01, linear warmup over
  100 steps.
- Training-row schedule: generator seed 20260805. Each full/control pair uses
  the identical precomputed row-index sequence.

## 2. Four-run matrix

The run order is seed 0 full A2, seed 0 draft-only control, seed 1 full A2,
seed 1 draft-only control. Each pair receives identical rows, cadence, nominal
budget, and any registered extension.

Full A2 reconstructs the matching A1 module, loads and freezes its learned flow,
and trains bridge, control, and draft parameters. Its fixed loss weights are the
seed-specific values banked by the zero-update calibration:

| Seed | final CE | cumulative KL | local CE | preserve KL |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 2201.8315363546058 | 395.0116541409731 | 521.7796435629211 |
| 1 | 1.0 | 1789.411575181737 | 303.9643794779331 | 467.10657754438176 |

The draft-only control uses the same seed-specific cumulative-KL and local-CE
weights. It reads the frozen initializer state, but it executes neither the
learned flow nor bridge writeback. Only control and draft parameters train.
Final CE and preservation are evaluation-only for the control. The unchanged
executed hidden path is asserted exactly at startup and after training.

## 3. Directional audit

The full A2 directional contract is evaluated only on the registered matched
51 x 128 training estimator at steps 200, 400, 600, 800, and 1,000, and every
200 steps during an extension. Step-zero 35/35/10/20 shares remain an
initialization record and are not a contract event.

- Primary losses are cumulative KL and local CE. Each must carry at least 50%
  of its relevant parameter-group gradient share.
- Non-primary losses are final CE and preserve KL. Each must carry at most 25%
  of its relevant parameter-group gradient share.
- A marginal miss is a primary share in [40%, 50%) or a non-primary share in
  (25%, 35%]. It warns and records the per-batch distribution. Two consecutive
  marginal misses at the same bound stop the run with receipts.
- A gross miss is a primary share below 40% or any non-primary share above 35%.
  It stops immediately with receipts.

The draft/control and bridge parameter groups are audited separately because
the losses are structurally disjoint across those groups. Zero gradients outside
a loss's relevant group are expected and are not pooled into the share.

## 4. Tripwires and extension

The seed-specific weighted-gradient p99-times-10 values, 2.3620389630494447 and
2.648981693767161, are catastrophe tripwires, not clipping shapers. No gradient
clipping is applied. A non-finite loss or gradient, frozen-state mutation,
quality collapse under the registered two-observation rule, or weighted gradient
norm above the seed's catastrophe threshold stops before the update and writes
receipts.

At step 1,000, the single extension applies to both members of a seed pair if
the full A2 arm has recomputed quality-safe oracle headroom below 2% relative to
the common zero-loop accepted length or its final-100-step accepted-length slope
exceeds 0.002. No second extension is permitted.

## 5. Verdict

The positive reading requires all of the following:

1. recomputed quality-safe oracle headroom of at least +2% relative to the common
   zero-loop mean accepted length;
2. full-system mean accepted length greater than its matched draft-only control;
3. registered endpoint-quality non-inferiority retained.

The receipt also reports the probe-KL/probe-top-1/accepted-length correlation
table, the trained-module monotonicity analogue, paired row differences, and all
directional-audit distributions. The vocabulary is `budget-limited` whenever the
single extension is exhausted without a positive verdict. V1d is attached to
the result handoff.

## 6. Authority

- Strategy resolution Drive ID: `17Am977A9iCg0-QYfEc0wUZHwN4ZYAVe1`
- Strategy resolution bytes: 5,381
- Strategy resolution SHA-256:
  `ca530b0153761d143b2f1c4f518e39ceadda67da8142cf5ae6d5279912587818`
- Calibration public summary SHA-256:
  `78476476407becb39e3ec5402a7403457ebd189dfe5afa180040296305b686ff`
- Calibration result commit: `e4275e02`
- Amendment-preparation result commit: `d5c9bbbc`
