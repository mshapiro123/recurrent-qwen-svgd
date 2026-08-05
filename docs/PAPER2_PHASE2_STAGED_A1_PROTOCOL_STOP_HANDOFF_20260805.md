# Handoff: Phase-2 Staged A1 Protocol Stop

Date: 2026-08-05  
Run: `stage5_paper2_phase2_staged_a1_20260805`  
Result commit: `dd47e76e`  
Launcher commit: `bda826f8c786c2c1c9f97513406c7065fd4c87f4`  
Protocol lock: `c0ace7aa02ebea11fc0809298572736d29b24012`

## 0. Executive verdict

The run was operationally healthy but did not complete Stage A1. Both seeds
stopped at optimizer step 200 on the preregistered static-gradient-share
contract. The registered classification is therefore `protocol_bug` with
receipts, not `a1_negative` and not `a1_pass`.

The early learning signal was nevertheless favorable. By step 200, both seeds
already cleared the two numerical A1 state-construction gates by wide margins:
functional-probe KL improved by 0.686 and 0.800 nats against a required 0.10,
and flow MSE decreased in both seeds. Those numbers are liveness evidence only.
They are not a valid A1 verdict because the loss mixture had moved outside its
locked contract.

Do not launch A2. The next job should be a read-only matched-estimator gradient
share audit on the saved step-200 checkpoints. The hard-stop audit used one
fixed 16-row DEV batch, while calibration used 51 training batches of 128 rows.
That mismatch must be resolved before deciding whether the defect is the audit
estimator or the static-weight design itself.

## 1. What was tested

The staged repair separated state construction from state use.

Stage A1 trained only `module.flow` at alpha 0.5 for seeds 0 and 1. The
initializer, bridge, control state, draft head, student and teacher embeddings,
and cached targets were frozen. Executed bridge and draft gates remained
closed. The preservation term used a training-only counterfactual read through
the frozen initialized bridge and never entered the executed prediction.

Before any optimizer update, each seed ran 100 calibration batches. Batches 50
through 100 estimated independent gradient norms. Static loss weights were
solved to produce the registered 60/20/20 gradient-share target:

| Loss | Target share |
|---|---:|
| Flow construction | 60% |
| Functional-probe KL | 20% |
| Counterfactual preserve KL | 20% |

The Huber delta was fitted from the training partition's target-increment p75.
The clip ceiling was ten times the calibrated p99 total gradient norm. Trust
penalty weight was zero; only the endpoint-ratio catastrophe tripwire remained.
The run was nominally 1,000 steps with one possible extension to 2,000. A2 was
physically absent from the launcher and required strategy review after A1.

## 2. Calibration receipts

Calibration made zero optimizer updates and did not mutate trainable parameters.
It hit the requested shares numerically in both seeds.

| Metric | Seed 0 | Seed 1 |
|---|---:|---:|
| Huber delta | 0.136718 | 0.136914 |
| Raw flow gradient norm | 0.408782 | 0.410797 |
| Raw probe gradient norm | 67.379707 | 62.307116 |
| Raw preserve gradient norm | 0.0001907 | 0.0002467 |
| Flow weight | 1.000000 | 1.000000 |
| Probe weight | 0.002022 | 0.002198 |
| Preserve weight | 714.377315 | 555.161906 |
| Weighted-gradient p99 | 0.489871 | 0.513143 |
| Clip ceiling | 4.898712 | 5.131429 |

The extreme static weights are not by themselves an error. They reflect a raw
gradient scale range of more than five orders of magnitude. They do show why
the stability check was load-bearing: small changes in relative raw gradients
can materially change the realized objective.

## 3. The protocol stop

At step 200, every active loss had to remain within 10 percentage points of its
target share. Both seeds failed because flow had become too dominant.

| Seed | Step | Flow | Probe | Preserve | Maximum absolute miss | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 calibration | 60.0% | 20.0% | 20.0% | 0.0 pp | exact |
| 0 | 100 | 77.1% | 8.7% | 14.2% | 17.1 pp | warning |
| 0 | 200 | 75.9% | 7.9% | 16.2% | 15.9 pp | stop |
| 1 | 0 calibration | 60.0% | 20.0% | 20.0% | 0.0 pp | exact |
| 1 | 100 | 78.6% | 10.2% | 11.2% | 18.6 pp | observed |
| 1 | 200 | 79.7% | 10.2% | 10.2% | 19.7 pp | stop |

The ratio-based drift alarm and the absolute hard contract are different. Seed
1 remained just inside the broad `[target/2, 2*target]` alarm for both auxiliary
terms, but its flow share still missed the absolute target by 19.7 points and
therefore correctly triggered the hard stop.

## 4. Early learning signals

These results show that the state-construction path was learning before the
protocol stop. They do not authorize A2 or a positive A1 claim.

| Metric | Seed 0 step 0 | Seed 0 step 200 | Seed 1 step 0 | Seed 1 step 200 |
|---|---:|---:|---:|---:|
| Flow loss | 0.105874 | 0.073840 | 0.105831 | 0.075162 |
| Flow MSE | 0.041519 | 0.037094 | 0.041479 | 0.036520 |
| Functional-probe KL | 5.785360 | 5.099746 | 5.530631 | 4.731030 |
| Probe top-1 | 3.89% | 4.31% | 3.75% | 4.89% |
| Endpoint error | 0.403473 | 0.390402 | 0.403357 | 0.386801 |
| Mean endpoint ratio | 0.0458 | 0.1086 | 0.0458 | 0.1470 |

Derived changes at step 200:

| Seed | Probe-KL improvement | Required | Flow-MSE reduction | Flow-loss reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.6856 nats | 0.10 | 10.66% | 30.25% |
| 1 | 0.7996 nats | 0.10 | 11.96% | 28.98% |

The KL gate was already numerically green. Probe top-1 remained low, so the
result should be phrased as distributional improvement rather than useful token
prediction. The full 1,000-step A1 question remains unanswered.

## 5. Health and safety checks

The stop was not caused by numerical instability or hardware failure.

- Calibration optimizer updates: 0 for both seeds.
- Calibration parameter hashes: unchanged for both seeds.
- Zero-loop identity: bit-exact hidden states and logits for both seeds.
- Frozen-module hashes: unchanged for both seeds.
- Non-finite loss: none.
- Gradient clipping: never activated.
- Endpoint-ratio catastrophe threshold of 5: zero exceeding steps in 200 for
  both seeds.
- Quality retention under closed execution: 1.0, Wilson lower 0.999837.
- Accepted-length delta under closed execution: exactly 0.
- A2 launched: false.

The saved Drive checkpoints are intact:

| Seed | Drive checkpoint | SHA-256 |
|---:|---|---|
| 0 | `.../private/a1/alpha_0p5_seed_0/a1_resume.pt` | `9815592e5358fbde535bec27d102717f4f9fe4a0beb9f649f0d0879f88db2c58` |
| 1 | `.../private/a1/alpha_0p5_seed_1/a1_resume.pt` | `f3538465223c2f09f286bbb276631b3ce9e60a7c3ecd43bf677d4d4c4dfb6e4e` |

The omitted checkpoint prefix is
`/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/` followed by the
run ID.

## 6. Interpretation

### What is established

1. The staged A1 graph is connected and trainable. Both seeds improved the
   registered functional and geometric diagnostics quickly.
2. The old shaper failure was repaired enough to avoid trust saturation,
   clipping, quality loss, and frozen-lineage mutation through step 200.
3. Static weights fitted at initialization did not satisfy the registered
   step-200 share contract under the implemented audit.
4. The hard stop worked as intended. It prevented an off-contract objective
   from silently becoming the experiment.

### What is not established

1. A1 has not passed or failed at its registered budget.
2. The flow has not been shown to support downstream state use.
3. A2 is not authorized.
4. Alpha 0.5 is not selected.
5. The result does not establish that static weighting is intrinsically
   inadequate, because the calibration and hard-stop estimators were not
   matched in sample size or population.

### Plain-language reading

The model started learning the right kind of hidden state. The main state loss
fell, and the hidden state became substantially closer to the teacher's token
distribution. But the three training signals stopped pulling with the balance
we had promised before the run: the main flow loss grew to roughly 76-80% of
the gradient instead of 60%. The experiment stopped at step 200 rather than
letting that changed objective run to completion. That is a good safety outcome,
but it means we still do not have the A1 answer.

## 7. Important estimator mismatch

The calibration and hard-stop share estimates were not computed on comparable
samples:

- Calibration: 51 random training batches, each with 128 rows.
- Step-200 hard stop: one fixed DEV batch with 16 rows.

The step-100 and step-200 results are consistent across seeds, so genuine
gradient drift is plausible. Nevertheless, a hard protocol verdict should not
rest on a much noisier estimator than the one that defined the weights. The
saved checkpoints make this cheap to resolve without training.

## 8. Recommended next action

Authorize one read-only matched-estimator audit before amending the protocol.

For each seed at initialization and at the saved step-200 checkpoint:

1. Reuse the exact 51 calibration batches of 128 training rows.
2. Recompute independent gradient norms and realized weighted shares with the
   locked weights.
3. Also evaluate a fixed 51-batch DEV estimator as a population-shift check.
4. Report bootstrap confidence intervals across batches and the fraction of
   batches within the 10-point band.
5. Perform no optimizer update and mutate no parameter.

Decision rule proposed for strategy approval:

- If the matched training estimator is inside the 10-point band, classify the
  stop as an audit-estimator implementation defect, patch the hard-stop
  estimator, and resume the saved checkpoints from step 200.
- If the matched training estimator confirms the miss, bank the static-weight
  instability and amend before rerunning. The cleanest candidates are a
  preregistered piecewise-static recalibration cadence or conversion of the
  share contract from a hard shaper to observe-and-log mode. Strategy must pick
  the objective semantics before code changes.
- If training passes but DEV fails, report a population-dependent gradient mix
  and decide explicitly which population owns the shaper contract.

Do not simply loosen the 10-point threshold after seeing this result. Do not
resume into A2. Do not count this as one of the registered completed A1 attempts.

## 9. Questions for strategy

1. Is the matched-estimator read-only audit authorized as the next job?
2. Which population should define the hard share contract: training,
   document-isolated DEV, or both?
3. If real drift is confirmed, should the 60/20/20 target be enforced through
   piecewise-static recalibration, or should realized shares become descriptive
   after initialization?
4. If the stop is purely estimator-driven, may the two saved step-200
   checkpoints resume, or is a fresh rerun preferred for cleaner lineage?
5. Should the low probe top-1 despite large KL improvement remain descriptive,
   or should A1 gain a token-level adequacy criterion before A2 opens?

## 10. Canonical artifacts

- Summary: `outputs/stage5/stage5_paper2_phase2_staged_a1_20260805/summary.json`
- Seed 0: `outputs/stage5/stage5_paper2_phase2_staged_a1_20260805/alpha_0p5_seed_0.json`
- Seed 1: `outputs/stage5/stage5_paper2_phase2_staged_a1_20260805/alpha_0p5_seed_1.json`
- Locked protocol: `docs/PAPER2_PHASE2_STAGED_REPILOT_PROTOCOL_DRAFT_20260805.md`
- Machine-readable registration:
  `training/paper2_phase2_staged_repilot_preregistration.json`

