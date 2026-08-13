# Handoff: P3.4 Prerequisites Complete and Executed Lock Assembled

Date: 2026-08-13. Audience: strategy and research review. Experimental status: prerequisite batch complete; P3.4 training has not started.

## 0. Executive result

The P3.4 charter can now be reviewed as an executed lock with no open transcription fields. The task inference contract, A_r fork, DEV task panel, empirical guardrail calibration, collateral ceiling, initialization hashes, schedule, slot-arm design, and seed-specific loss weights are all bound and receipted. No optimizer was constructed and no P3.4 training step ran.

The prerequisite measurements support launch, with one important qualification. Tier-S is a properly calibrated catastrophe stop: at the conservative autocorrelation edge it has a familywise false-stop upper bound of `2.996e-5` and 99.822% power for a sustained 5.5-point drop. Tier-W is much less sensitive to its nominal 3-point class: only 30.694% power, rising to 98.601% at 5 points. Tier-W is therefore useful as a conservative demotion trigger but cannot support a claim that every 3-point degradation will be detected.

Summary figure: `docs/figures/p34_executed_lock_calibration_summary_20260813.svg` (PNG companion in the same directory).

The loss-share calibration also found substantial seed dependence. Both seeds can be brought to exactly the same registered shares, but the scalar weights needed to do so differ sharply. This supports the decision to bind weights per seed and warns against treating seed 0's gradient scale as universal.

## 1. Question and rationale

P3.4 asks whether the recurrent sidecar's token-level corrections convert into better task answers under a fully specified inference graph. It is the first campaign in this line whose primary curve is task-level gap closed against the 14B teacher rather than an oracle or token-only proxy.

The prerequisite batch exists to ensure that this question is not confounded by four prior failure modes:

1. an undefined task inference graph;
2. a fork chosen without pricing the readout bottleneck;
3. guardrails copied across estimators without recalibration;
4. aggregate loss buckets that hide starvation of individual losses.

The charter resolves the first two. This batch measured the latter two and converted them into executable constants.

## 2. Experimental design

### Task guardrail calibration

The fixed 1,024-row DEV panel contains equal floor and target halves. Six model conditions were scored under the exact P3.4 graph: initialization, P3.3 endpoint, and i1 endpoint for each seed. The calibration estimated paired discordance and adjacent-checkpoint autocorrelation, bootstrapped an upper-95 autocorrelation edge, and simulated 100,000 twenty-look campaigns at each sensitivity-band endpoint.

Tier-S searches for the smallest sustained true drop with at least 99% detection power while retaining familywise false-stop probability at or below `1e-4`. Tier-W keeps the charter's -3-point decision margin and two-consecutive-look rule, with demotion rather than termination as its consequence.

### Loss-share calibration

The B6 set is a fixed, hashed 256-row population: 128 code and 128 general, with a 1:3 positive-negative ratio within each stratum. Every selected source tensor and oracle direction was hash-verified. For each seed and arm, the evaluator measured independent unit-weight loss gradients over the registered trainable set, applied the registered combined group clipping factors, averaged norms across strata, and solved KL-normalized static weights against the registered per-loss target shares.

Seed 0 evaluated the main and slot arms. Seed 1 evaluated the campaign-of-record main arm. Slot seed 1 remains unauthorized unless the seed-0 paired arm read is positive and strategy confirms promotion.

## 3. Execution and engineering receipts

The initial Colab implementation exposed three infrastructure defects before any training:

- Dispersed source rows required hundreds of small Drive shard transactions. Colab reclaimed both GPU and CPU filesystems before staging finished.
- Raw file hashes differed across Windows and Linux because tracked JSON/JSONL line endings differed despite identical records.
- The pinned tied-head artifact used its actual `weight_bfloat16` schema, which the first loader did not recognize.

The repairs preserved the experiment:

- 346 source files were transferred in bulk and hash-verified on durable local storage, then compacted into a 9.7 MB immutable tensor batch.
- Portable identity uses canonical JSON/JSONL hashes; raw artifact hashes remain recorded where transport-specific.
- The evaluator reads the hash-pinned tied-head schema directly.
- Nested tensor telemetry is losslessly converted at the JSON receipt boundary.
- In-VM Drive publication was removed because the deliberately read-only credential cannot create files; receipts are downloaded via the Colab CLI and published through the governed Drive connector.

These are transport and receipt fixes only. No selected row, checkpoint, tensor, estimator, loss, clipping rule, target share, threshold, or scientific constant changed.

All Colab sessions were released after receipts were downloaded.

## 4. Results

### 4.1 Guardrail calibration

Measured paired discordance was 9.310%. Empirical adjacent-checkpoint autocorrelation was 0.8587; its bootstrap upper-95 edge was 0.8876 and is the lock value.

| Quantity | Empirical correlation | Conservative correlation |
|---|---:|---:|
| Tier-S delta_cat | 5.0 points | 5.5 points |
| Tier-S power at delta_cat | 99.145% | 99.822% |
| Tier-S null upper-95 | `2.996e-5` | `2.996e-5` |
| Tier-W power at -3 points | 32.585% | 30.694% |
| Tier-W power at -5 points | 99.146% | 98.601% |
| Tier-W null upper-95 | `4.744e-5` | `7.753e-5` |

The conservative edge is bound. Tier-S clears both its false-stop and power requirements. Tier-W's low 3-point power is not a contract failure because its consequence is demotion and review, not a capability claim or absolute safety stop, but it is a meaningful limitation.

### 4.2 Loss-share weights

| Configuration | KL | Aim | CE | Gate | Preserve | Slot |
|---|---:|---:|---:|---:|---:|---:|
| Seed 0 main | 1.0000 | 0.05415 | 0.03459 | 0.0002289 | 54.8435 | - |
| Seed 1 main | 1.0000 | 0.02110 | 0.03311 | 0.00006163 | 9.8908 | - |
| Seed 0 slot | 1.0000 | 0.05415 | 0.03459 | 0.0002289 | 63.5488 | 0.01728 |

All three solves converged in two iterations and matched target shares to numerical precision. Main-arm targets resolve to KL 41.667%, aim 17.857%, CE 11.905%, gate 3.571%, preserve 25%. Slot-arm targets resolve to KL 35.959%, aim 15.411%, CE 10.274%, gate 3.082%, slot 10.274%, preserve 25%.

The unit-weight geometry explains the unusual multipliers. Gate gradients dominate before weighting: mean post-clip gate norm was `0.49277` for seed 0 and `0.49924` for seed 1. Preservation gradients were only `1.72e-5` and `2.29e-5`, respectively. The gate therefore needs severe down-weighting and preservation needs substantial up-weighting. The magnitude differences across seeds are large enough that a shared scalar vector would violate the matched share contract.

The seed-0 slot loss has a unit-weight mean post-clip norm of `0.18633`, requiring a scalar of `0.01728` to occupy its registered 10.274% share. Its zero-initialized lift produced the expected step-zero chance behavior; this is an initialization check, not evidence for or against the arm's eventual value.

## 5. Interpretation

### What is supported

- The P3.4 task estimator now has a calibrated catastrophe stop under the exact 20-look schedule.
- The per-loss optimization budget can be made exact at step zero for both main seeds and the seed-0 slot arm.
- The slot arm adds a measurable gradient channel without perturbing the registered main-arm gradient computation.
- Seed-specific scalar weights are necessary to preserve the same causal comparison across independently initialized endpoints.
- The compact transport artifact is a reproducibility improvement: future runs load 9.7 MB instead of restaging 4.14 GiB from hundreds of files.

### What is not supported

- No task improvement, gap closed, or better-model claim exists yet. No P3.4 training has occurred.
- Tier-W does not reliably detect every 3-point drop at this panel size.
- Step-zero share matching does not guarantee trailing-window shares will remain in bounds. The registered warning and stop logic still has to operate during training.
- The slot arm is single-seed by design and cannot be generalized or promoted before its paired result.
- The calibration is DEV-only and does not spend or preview CONFIRM or EVAL-E.

## 6. Questions for strategy review

1. Does strategy accept Tier-W's measured 30.694% power at the nominal 3-point class given that its consequence is demotion and review, with 98.601% power at 5 points? The executed lock keeps the ratified rule unchanged and discloses the operating characteristic.
2. Does strategy affirm seed-specific static weights as the only matched implementation of B6? A shared vector would be a new, unsupported invariance assumption contradicted by the measured norms.
3. Should preservation landing exactly at its 25% ceiling be accepted as intended, or should the later runner treat equality as a watch condition? The current lock follows B6 literally: at or under 25%.
4. After Mark's approval, should the three sessions launch concurrently or should the slot seed-0 arm trail the two main lines by one look to reduce review concurrency? This is an operations choice only; the experimental constants are already fixed.

## 7. Next steps

1. Strategy reviews this handoff and `PAPER2_PHASE3_P34_EXECUTED_LOCK_20260813.md` against the charter and bindings.
2. Mark either approves the executed lock or identifies a specific amendment. No training occurs before explicit approval.
3. On approval, the coding lane adds a runner that can only read the locked JSON and asserts all hashes and constants before optimizer construction.
4. Launch main seed 0, main seed 1, and slot seed 0. Report all 20 windows as curves, with the controller state, loss shares, task guardrails, chi, gate statistics, and registered diagnostics.
5. Strategy reads the campaign against the prewritten failure-signature table and decides whether P3.6 confirmation, a P3.5 lever, or a boundary result is warranted.

## 8. Plain-language summary

The experiment is ready to be signed, but it has not started. We now know exactly how sensitive the damage alarms are and exactly how much weight each training signal needs on each seed. The strongest safety alarm is credible: it almost never fires by accident and catches the large sustained drop it is meant to stop. The gentler three-point warning is deliberately conservative and will miss many true three-point drops, so it should be read as an occasional demotion signal, not a complete detector. The optimization signals differ more across seeds than their identical architecture suggests, which is why each seed gets its own calibrated weights. With those facts written down, the next run can answer the scientific question rather than rediscovering a hidden loss imbalance or an uncalibrated stop rule.

## 9. Receipt map

- Executed lock: `docs/PAPER2_PHASE3_P34_EXECUTED_LOCK_20260813.md` and `training/paper2_phase3_p34_preregistration.json`.
- Task calibration: repository SHA `2efad887a5beade3bd21fc4d10caecae3675d6f1e25129b23cb0d9596d191476`; Drive-byte SHA `eb50fcacf1197e2436c34cb2d40e0a8cafbdd0cb6ba8dea7211e738209bcc17a`, Drive `1r0jUt6S8Eibe4jwm2-IebYk7qnz3zPaF`.
- Chi source: SHA `4980e986ba67a4dcdcd712f512ea57ac5abd9a41f19e0e5eb97efd234c5a7d2b`.
- Compact batch: SHA `f0207e6242424bcc44659d4c020af1cdc15a8e7f764cb3853a11f3086159b1df`, Drive `1NvWNpSie9lYnNjpmoJrWldyFMMmKpiK6`.
- Compaction receipt: repository SHA `09c49d6d710e8c9d545b1e8b98e4c1e3af168b8455b3f8fd7601f33df181fe87`; Drive-byte SHA `13764a81040944a0a9b19ca4d79c05b1fbbc6ecfb50060eb2a88f8e559ce2435`, Drive `1egPmflIyT-TSQ3zt5GUwXPbR9jr3X87m`.
- Seed-0 calibration: SHA `bdcc62eecb5272fee4e4cf3245733e55253b41cd8be7a2df2a19c17caf8f7942`, Drive `1SqneiMuzpB5HqDnhfaKUbPYv3__HZV-r`.
- Seed-1 calibration: SHA `072bdf418e32b7b65b1002cec0505f402efb5ff9f66a7ed26c9d32849a5c46f6`, Drive `1SwrOp4-h9w0wTppHxpph65f2RtYBkLRv`.
