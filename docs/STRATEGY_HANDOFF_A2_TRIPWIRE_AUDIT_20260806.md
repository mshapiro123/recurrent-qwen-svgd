# Strategy Handoff: A2 Gradient-Tripwire Audit

Date: 2026-08-06

## 0. Executive verdict

The read-only audit landed and resolves the step-237 stop as
`STATIC_TRIPWIRE_STALE`, not as a numerical catastrophe and not as an isolated
malformed training batch. The static gradient threshold was calibrated at the
step-zero A2 state as ten times the initial p99. By step 237, the entire gradient
distribution had shifted upward enough that the old seed-0 threshold sat near the
current upper decile rather than at a catastrophe tail. Both stopping updates were
finite and small relative to trainable parameter norm.

This audit does not clear the stop. A separate strategy amendment is required.
The coding recommendation is to resume all four exact step-237 states under an
observe-only gradient-norm policy, retaining the non-finite, quality-trajectory,
directional-allocation, lineage, and endpoint gates unchanged. The previously
rejected batch should be applied to every arm rather than skipped.

## 1. Audit design

The audit used the two landed full-system step-237 checkpoints and reconstructed
the exact A2 training schedule from row seed `20260805`.

- Exact stopping attempt: 238, row hash
  `3fee92d729b86a2a354a259562343f5ba7cf084e2b01bae84012fdc5c4a3d716`.
- Local schedule window: attempts 228 through 248, 21 batches.
- Broader reference: the registered 51-batch matched directional estimator,
  re-evaluated at each step-237 checkpoint.
- Detailed stopping-batch decomposition by loss and module.
- Batch composition compared with all 41,969 training anchors.
- One exact AdamW update simulated in memory from each saved optimizer state,
  followed by full 8,031-anchor DEV evaluation and immediate state restoration.
- Zero persisted optimizer updates. All source hashes remained unchanged.

## 2. Gradient-distribution result

| Measurement | Seed 0 | Seed 1 |
|---|---:|---:|
| Initial calibration p99 | 0.2362 | 0.2649 |
| Static stop threshold | 2.3620 | 2.6490 |
| Step-237 reference median | 1.7002 | 1.7292 |
| Step-237 reference p95 | 2.4895 | 2.3890 |
| Step-237 reference p99 | 2.8911 | 2.6935 |
| Step-237 reference maximum | 3.2294 | 2.9447 |
| Stopping-batch norm | 2.3932 | 2.7742 |
| Stop / threshold | 1.013 | 1.047 |
| Reference batches above threshold | 7/51 | 1/51 |
| Window batches above threshold | 4/21 | 1/21 |
| Stop percentile in reference | 90.2% | 98.0% |

The stopping row was high, especially for seed 1, but seed 0 demonstrates that
the old threshold had ceased to represent a catastrophe tail at all. The current
seed-0 median was 7.2 times the initial calibration p99; seed 1 was 6.5 times.
This is optimization-trajectory scale drift under fixed loss weights, not a
non-finite event.

## 3. Stopping-batch composition

The row batch was ordinary on the observed covariates:

| Feature | Full training population | Batch 238 |
|---|---:|---:|
| Base/teacher top-1 agreement | 71.25% | 69.53% |
| Hidden RMS | 9.468 | 9.384 |
| Target RMS | 0.28471 | 0.28466 |
| Mean candidate count | 228.93 | 231.42 |

Its position-bucket mix was also consistent with the late-position-heavy training
population. Nothing in these measurements supports treating batch 238 as bad data.

## 4. Gradient mechanism

The stopping gradient remained strongly aligned with the intended primary
objectives:

| Measurement | Seed 0 | Seed 1 |
|---|---:|---:|
| Cumulative-KL share | 83.82% | 87.05% |
| Local-CE share | 14.79% | 12.03% |
| Aggregate primary share | 98.61% | 99.08% |
| Cumulative-KL/local-CE cosine | +0.855 | +0.655 |
| Final-CE share | 0.73% | 0.59% |
| Preserve-KL share | 0.65% | 0.33% |

The total norm was concentrated in the draft module: 2.369 of 2.393 for seed 0
and 2.756 of 2.774 for seed 1. Bridge norms were only 0.023 and 0.019. Thus the
event was not a bridge explosion or conflict inversion. It was growth in the
primary draft-learning signal as the draft gate opened.

## 5. One-update causal safety check

| Measurement | Seed 0 | Seed 1 |
|---|---:|---:|
| Parameters finite after simulated update | yes | yes |
| Total relative update norm | 0.2497% | 0.2467% |
| Bridge relative update | 0.0990% | 0.0874% |
| Control relative update | 0.4488% | 0.4253% |
| Draft relative update | 0.2908% | 0.2927% |
| DEV accepted-length change | -0.0000124 | -0.0000162 |
| DEV retained-correct change | -4 rows | -3 rows |
| DEV retention change | -0.0169 points | -0.0127 points |
| Oracle-headroom change | +0.00173 points | +0.00248 points |

The update was mechanically benign but not immediately beneficial. It produced
a tiny accepted-length and retention cost while slightly increasing oracle
headroom. Existing trajectory-quality checks are the appropriate instrument for
detecting accumulation of this cost; the stale raw-gradient cutoff was not.

## 6. What the audit does and does not establish

Established:

- The static raw-gradient threshold no longer represented catastrophe at step 237.
- Batch 238 was not malformed on the measured data covariates.
- The gradient remained primary-objective dominated and positively aligned.
- One exact update was finite, reversible, and small relative to parameter norm.
- The original A2 run remains protocol-blocked and has no endpoint verdict.

Not established:

- A2 will become useful with more training.
- The two-percent oracle-headroom gate is reachable.
- The slight immediate quality loss will reverse rather than accumulate.
- A relative-update threshold has been empirically calibrated.

At step 237, full-system accepted length remained slightly below the matched
control and quality-safe oracle headroom was only 0.040% and 0.056% against the
two-percent endpoint requirement. The prior on eventual success is therefore
lower, but the run stopped too early for the registered endpoint claim.

## 7. Recommended amendment

1. Resume all four exact step-237 checkpoints to the registered step-1,000
   endpoint, preserving seeds, rows, optimizer state, weights, and control pairing.
2. Reclassify raw gradient norm from a hard stop to observe-and-log. Record its
   distribution and old-threshold exceedances at each evaluation interval.
3. Retain hard stops for non-finite loss, gradients, parameters, lineage mutation,
   Wilson quality failure, and the registered two-evaluation quality trajectory.
4. Retain the directional-allocation contract and endpoint gates unchanged.
5. Apply attempt 238 to all four arms. Do not skip or clip it.
6. Reconstruct each batch generator to exactly 237 completed updates before
   continuation. The seed-1 checkpoint saved its generator after selecting the
   rejected attempt, so merely clearing the abort would silently advance it to
   attempt 239. This must be asserted and repaired before any update.
7. Log relative update norm descriptively. Do not convert it into a new hard
   threshold without calibration.

## 8. Strategy questions

1. Ratify `STATIC_TRIPWIRE_STALE` and authorize observe-only raw-gradient logging?
2. Authorize exact four-arm continuation from step 237 to step 1,000?
3. Keep the pair-matched extension rule unchanged if a full arm reaches step 1,000?
4. Require an intermediate strategy read at step 400, or let the existing quality
   and directional checks govern uninterrupted continuation?

Recommendation: authorize the narrow continuation without an additional step-400
decision gate. Step 400 already carries a directional audit and DEV evaluation;
write the receipt, but avoid creating another discretionary stop unless an existing
registered tripwire fires.

## 9. Plain-language summary

The alarm went off because the gradients naturally became much larger during
learning, while the alarm threshold stayed frozen at its starting value. The
rejected update was not explosive: it moved the trainable parameters by about one
quarter of one percent and left every value finite. It also did not produce an
instant win, which keeps the scientific question open. The disciplined next move
is to repair the alarm, not ignore safety and not declare the model dead: continue
the exact paired experiment while preserving the quality and numerical tripwires
that measure actual harm.

## 10. Canonical receipts

- Audit result commit: `42922c97`.
- Audit summary:
  `outputs/stage5/stage5_paper2_phase2_a2_tripwire_audit_20260806/summary.json`.
  SHA-256: `3007927b5da62454c2d12ddef4f99e15beb726cbf9d509aed3e5594a28070ee5`.
- Resumed-matrix result commit: `1217e02a`.
- Resumed-matrix summary:
  `outputs/stage5/stage5_paper2_phase2_a2_resume_20260805/summary.json`.
- Audit implementation/spec commit: `463dfd20`.
