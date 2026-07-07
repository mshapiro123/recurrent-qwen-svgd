# Stage 5 Release Gate Handoff

Prepared: 2026-07-07  
Repository: `mshapiro123/recurrent-qwen-svgd`  
Current tracked head reviewed: `0b6c551`  
Primary artifacts:

- `outputs/stage5/stage5_depth_support_ladder8_20260705_204923/summary.json`
- `outputs/stage5/stage5_support8_probe_readout_20260706_151627/summary.json`
- `outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json`
- `outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/summary.json`

## 1. Executive Status

The planned experiment batch is not fully complete.

Completed:

- Support-8 ladder, frozen depth 1-14 evaluation.
- Support-8 probe readout.
- Support-8 dose arm, plus 2,000 same-curriculum steps.
- Same-reader final-symbol release gate.

Not yet completed:

- `phase_a_surpass_prereg`
- `support6_seed_replication`
- `n24_support12_rung`

The most important result is positive: the same-reader final-symbol release gate exactly matches the active-label diagonal counts from the support-8 dose arm. The previously problematic option-text MCQ final-answer reader should remain retired as diagnostic only.

## 2. Main Findings

### Finding 1: The support-8 dose arm revived strong scaling.

The first support-8 run did not clear the strong scaling gate. It preserved trained-depth performance and extended to depth 9, but missed the depth-10/depth-11 strong bars.

After 2,000 additional same-curriculum steps, the support-8 dose arm crossed the locked strong scaling thresholds at depths 10 and 11.

| Depth | Support-8 original | Support-8 dose | Delta |
|---:|---:|---:|---:|
| 1 | 128 | 128 | 0 |
| 2 | 125 | 127 | +2 |
| 3 | 127 | 126 | -1 |
| 4 | 128 | 125 | -3 |
| 5 | 128 | 127 | -1 |
| 6 | 121 | 126 | +5 |
| 7 | 119 | 124 | +5 |
| 8 | 117 | 122 | +5 |
| 9 | 109 | 116 | +7 |
| 10 | 85 | 113 | +28 |
| 11 | 64 | 97 | +33 |
| 12 | 48 | 87 | +39 |
| 13 | 26 | 57 | +31 |
| 14 | 20 | 31 | +11 |

Locked gates from the support-8 dose run:

- Non-regression through depth 8: pass.
- Adjacent extension at depth 9: pass.
- Depth-10 asymptote rejection: pass.
- Strong scaling at depth 10 and depth 11: pass.
- Long-tail above chance through depth 14: pass.

Recorded verdicts:

- Original support-8 ladder: `asymptote_rejected_at_depth10`
- Support-8 dose arm: `strong_scaling`
- Dose read: `soft_scaling_revived`

Interpretation: the earlier depth-10 failure was at least partly dose-confounded. The system did not hit a hard support-8 wall at the previous checkpoint.

### Finding 2: The same-reader release gate cleared.

The same-reader final-symbol evaluator reads the model's final answer using the same full-symbol candidate space as the active-label chain evaluator. This was necessary because earlier option-text MCQ final-answer tables were confounded by reader/surface mismatch.

Same-reader result:

| Depth | Correct / 128 | Accuracy | Clears 0.71 |
|---:|---:|---:|:---:|
| 1 | 128 | 1.000 | yes |
| 2 | 127 | 0.992 | yes |
| 3 | 126 | 0.984 | yes |
| 4 | 125 | 0.977 | yes |
| 5 | 127 | 0.992 | yes |
| 6 | 126 | 0.984 | yes |
| 7 | 124 | 0.969 | yes |
| 8 | 122 | 0.953 | yes |
| 9 | 116 | 0.906 | yes |
| 10 | 113 | 0.883 | yes |
| 11 | 97 | 0.758 | yes |
| 12 | 87 | 0.680 | no |
| 13 | 57 | 0.445 | no |
| 14 | 31 | 0.242 | no |

Total:

- Same-reader final-symbol: `1506/1792 = 0.8404`
- Deterministic symbol-to-option mapped final: `1506/1792 = 0.8404`

The same-reader counts are identical to the support-8 dose active-label diagonal counts. This confirms that the support-8 dose result is not an artifact of active-label scoring alone. It also confirms the old option-text MCQ final matrices should stay suspended for this synthetic-depth line.

### Finding 3: Tail decay remains real.

The support-8 dose arm extends well beyond trained support but still decays in the far tail:

- Depth 10: `113/128`
- Depth 11: `97/128`
- Depth 12: `87/128`
- Depth 13: `57/128`
- Depth 14: `31/128`

This is not open-ended algorithmic recurrence yet. It is a positive scaling result with visible long-tail decay.

### Finding 4: Probe evidence still shows loop-clock and envelope drift.

The support-8 probe readout reported a very strong loop-index signal:

- Loop-index probe accuracy: `0.965`
- Permutation p95: `0.082`
- Lift over p95: `0.883`

State-envelope reconstruction error rises after the support region:

| Loop | Mean reconstruction MSE |
|---:|---:|
| 8 | 0.063 |
| 9 | 0.089 |
| 10 | 0.163 |
| 11 | 0.338 |
| 12 | 0.706 |
| 13 | 1.394 |
| 14 | 2.581 |

Interpretation: the model can push beyond the original support envelope with more training, but the hidden-state trajectory still becomes increasingly out-of-distribution in the tail.

## 3. What This Means

The synthetic-depth line is still alive.

The support-8 dose result plus same-reader confirmation show that the recurrent architecture can learn and preserve a structured iterative computation, then extend beyond its trained support after more same-curriculum training. This is stronger than the previous "horizon-bound finite chain" reading.

The strongest defensible claim is:

> The corrected recurrent Qwen-0.5B substrate learns a depth-indexed iterative symbolic task; additional support-depth training revives strong extrapolation at depths 10-11 and improves the tail through depth 14; same-reader final-symbol scoring confirms this is not a reader artifact.

The strongest claim we should not make yet:

> The system has learned open-ended algorithmic recurrence.

The remaining decay at depths 13-14 and the loop-clock/envelope-drift probe argue against that stronger claim for now.

## 4. Planned Experiment Status

| Planned item | Status | Purpose | Current read |
|---|---|---|---|
| Same-reader final-symbol gate | Complete | Retire MCQ reader confound | Cleared |
| Phase-A surpass preregistration | Pending | Lock comparison criteria before dense arms | Ready to run |
| Support-6 seed replication | Pending | Check support-route robustness across seeds | Ready to run |
| N=24 support-12 rung | Pending | Test whether scaling law survives larger symbol space and longer support | Ready to run, expensive |

## 5. Recommended Next Order

1. Run `phase_a_surpass_prereg`.
   - Cheap.
   - Locks dense-vs-recurrent comparison criteria before any baseline arms.

2. Run `support6_seed_replication`.
   - Moderate GPU cost.
   - Tests whether the support-route result is robust to seed variation.

3. Run `n24_support12_rung`.
   - Expensive.
   - This is the next decisive synthetic scaling test.
   - Prefer A100, H100, or high-memory G4. L4 can run it but may be slow.

4. Only after those: implement dense direct and dense scratchpad Phase-A arms.
   - The Phase-A preregistration already defines the primary gate:
     recurrent support-8 dose beats dense direct at three or more consecutive depths with one-sided Fisher p < 0.05 per depth.

## 6. Open Questions For Strategy Agent

1. Does the support-8 dose plus same-reader confirmation justify treating this as a live positive mechanism line?

2. Is the observed tail decay best understood as:
   - insufficient dose,
   - insufficient support depth,
   - state-envelope drift,
   - finite-horizon loop-clock learning,
   - or a mixture?

3. Should the N=24 support-12 rung be treated as the next decisive experiment, or should support-6 seed replication come first to protect against seed variance?

4. If N=24 support-12 clears depth 16/17, does that establish a scaling law strong enough to justify moving back to natural reasoning traces?

5. If N=24 fails, should we:
   - increase dose,
   - change curriculum,
   - add explicit state-envelope regularization,
   - or close the synthetic line as promising but bounded?

6. What is the right standard for "recurrence" here?
   - Same-task extrapolation beyond trained support?
   - Transfer to larger symbol spaces?
   - Hidden-state invariance across loop count?
   - Natural reasoning benchmark lift over dense baselines?

## 7. Bottom Line

The current batch is not fully finished, but the release gate that did land is important and positive.

The same-reader final-symbol gate confirms that the support-8 dose result is real under the corrected reader. The model clears depth 11 at `97/128`, preserves trained-depth non-regression, and improves every tail depth relative to the original support-8 ladder. The result does not yet prove open-ended recurrence, but it meaningfully strengthens the case that the corrected recurrent architecture can learn useful iterative latent computation.

The next decision should not be whether to abandon the line. The next decision is how much compute to spend on the final synthetic scaling rung before returning to dense baselines and natural reasoning traces.
