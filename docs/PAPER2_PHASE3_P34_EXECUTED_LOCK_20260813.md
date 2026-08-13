# P3.4 Executed Lock - Awaiting Mark Approval

Date: 2026-08-13. Status: assembled and locked before training. Training remains structurally disabled.

## 1. Authority and effect

This executed lock combines:

- `STRATEGY_P34_CHARTER_20260812.md`, Drive `1lh2Vf3VG8yPXUjLPfr7IJkPCjnf2mKE6`, SHA-256 `80cb1b13eb48ffff064ff7cc6c0d02de773dfec80924c1c50736115821c97ce4`.
- `STRATEGY_P34_BINDINGS_20260812.md`, Drive `1mLkbVZYhKyoiOuiOwdbjT9_nwA3Z8eYI`, SHA-256 `a650580f63e9556e6b5d77732d810e61052e8d3f041f27c79eabbf12f1106294`.
- `STRATEGY_P34_BINDINGS_RATIFICATION_20260812.md`, Drive `1em-sujz1D9I-GtSpFrmzz7WDxANTlY5R`, SHA-256 `6311ae91478cd9cd306b9a61bc8788877e510fdc43c28abe5cd75ca80403bc0a`.
- Machine-readable lock: `training/paper2_phase3_p34_preregistration.json`.

All prerequisite values are now bound. `locked_before_training = true`, `training_authorized = false`, the unresolved-field list is empty, no P3.4 training runner exists, no optimizer was constructed by a prerequisite job, and zero P3.4 training steps have occurred. Mark's approval of this assembled lock is still required before any training launcher may be added or run.

## 2. Campaign configuration

The campaign of record is the no-arm configuration on seeds 0 and 1. The slot-supervision arm runs on seed 0 only. The capacity arm remains shelved to P3.5.

Initialization is hash-pinned to the i1 endpoints:

| Seed | Endpoint SHA-256 |
|---|---|
| 0 | `01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88` |
| 1 | `2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a` |

Training, once separately approved: AdamW, learning rate `3e-4`, weight decay `0.01`, betas `(0.9, 0.999)`, 100 warmup steps, batch size 128, 4,000 steps, 20 looks every 200 steps, and Tier-2 audits at looks 5, 10, 15, and 20. The run is resumable. The task graph uses fresh scratch per emitted token, four flow loops, greedy decoding, position zero closed, no cross-token persistence, and no draft-head scoring.

## 3. Task guardrails

The fixed 1,024-row DEV panel is 512 floor rows and 512 target rows, seed 20260812, panel SHA `3e6c62ac4ef36a22eeba961e5d4d84c3403fd55aa837d32746eb8f35d8fe3163`. The calibration uses six conditions: initialization, P3.3, and i1 for each seed. The estimand is paired augmented-minus-base task correctness.

Measured noise:

- Paired discordance: `0.0930989583`.
- Empirical adjacent-checkpoint autocorrelation: `0.8587331565`.
- Bootstrap upper-95 autocorrelation used as the conservative edge: `0.8875501005`.

The conservative operating contract is:

| Guardrail | Bound value | False-action estimate | Detection power |
|---|---:|---:|---:|
| Tier-S absolute stop | sustained 5.5-point drop; one-sided alpha 0.10; two consecutive looks | 0 observed in 100,000 simulations; upper-95 `2.996e-5` | `0.99822` at -5.5 points |
| Tier-W demotion | sustained 3-point drop; same alpha; two consecutive looks | `3e-5`; upper-95 `7.753e-5` | `0.30694` at -3 points; `0.98601` at -5 points |

Tier-S satisfies the registered familywise false-stop ceiling of `1e-4` and power floor of 0.99. Tier-W is a demote-and-review signal, not a high-power 3-point detector on this panel. Its limited 3-point sensitivity is disclosed rather than upgraded by narrative.

Task calibration receipt: `outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/receipts/p34_task_guardrail_calibration.json`, repository SHA-256 `2efad887a5beade3bd21fc4d10caecae3675d6f1e25129b23cb0d9596d191476`; Drive `1r0jUt6S8Eibe4jwm2-IebYk7qnz3zPaF`, Drive-byte SHA-256 `eb50fcacf1197e2436c34cb2d40e0a8cafbdd0cb6ba8dea7211e738209bcc17a`.

## 4. Collateral controller

The chi source is the banked adversarial-direction curve on the held-out negative audit population:

| Radius | Flips / 24,576 | Rate |
|---:|---:|---:|
| 0.30 | 0 | 0 |
| 0.60 | 23 | `0.0009358724` |
| 1.00 | 1,107 | `0.0450439453` |

The bound `chi_max` vector is `[0.0005, 0.0005, 0.0005, 0.001]`; the estimator is held-out negative-audit flips per window, pooled across seeds. It is revisit-labeled if the radius curve is remeasured on a trained-forward checkpoint. Source receipt: `outputs/stage5/stage5_paper2_phase3_p33_verification_20260812/summary.json`, SHA-256 `4980e986ba67a4dcdcd712f512ea57ac5abd9a41f19e0e5eb97efd234c5a7d2b`.

## 5. Loss-share calibration

The fixed B6 population contains 256 rows: 128 code and 128 general, each comprising 32 positives and 96 negatives. Selection seed is 20260812. The source population, audit exclusions, and rank rule are unchanged.

To prevent Colab filesystem reclamation during dispersed-shard staging, 346 source files were hash-verified and compacted locally into an immutable 9,699,014-byte tensor batch. This changed transport only, not row selection, tensors, estimator, or model computation. The portable v2 batch uses canonical JSON and JSONL hashes so Windows and Linux line endings cannot alter identity.

- Compact batch: Drive `1NvWNpSie9lYnNjpmoJrWldyFMMmKpiK6`, SHA-256 `f0207e6242424bcc44659d4c020af1cdc15a8e7f764cb3853a11f3086159b1df`.
- Compaction receipt: repository SHA-256 `09c49d6d710e8c9d545b1e8b98e4c1e3af168b8455b3f8fd7601f33df181fe87`; Drive `1egPmflIyT-TSQ3zt5GUwXPbR9jr3X87m`, Drive-byte SHA-256 `13764a81040944a0a9b19ca4d79c05b1fbbc6ecfb50060eb2a88f8e559ce2435`.
- Direction cache SHA-256: `611be787dea0438761d279aa035d5bfe2aa37e74710d880be1066d7ae80a45a2`.

The estimator is the mean over the fixed code and general strata of independent, unit-weight loss-gradient norms after the registered combined bridge/head clipping factors. Static weights are KL-normalized and seed-specific:

| Configuration | KL | Aim | CE | Gate | Preserve | Slot |
|---|---:|---:|---:|---:|---:|---:|
| Seed 0 main | 1 | 0.0541522 | 0.0345881 | 0.000228855 | 54.843469 | - |
| Seed 1 main | 1 | 0.0211030 | 0.0331089 | 0.0000616263 | 9.890760 | - |
| Seed 0 slot | 1 | 0.0541522 | 0.0345881 | 0.000228855 | 63.548782 | 0.0172772 |

Every solver converged in two iterations. Maximum target-share error was zero for seed 0 main, `2.78e-17` for seed 1 main, and `5.55e-17` for seed 0 slot. Preservation landed at the 25% ceiling in every configuration.

The resulting main-arm shares are KL 41.667%, aim 17.857%, CE 11.905%, gate 3.571%, and preservation 25%. The slot-arm shares are KL 35.959%, aim 15.411%, CE 10.274%, gate 3.082%, slot 10.274%, and preservation 25%. All registered per-loss floors are therefore met at step zero.

The raw gradient geometry is seed-dependent. Unit-weight gate gradients dominate both seeds, while preservation is tiny before weighting. Seed-specific weights are consequently binding experimental constants, not interchangeable formatting.

Receipts:

- Seed 0: `outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/receipts/p34_share_calibration_seed_0.json`, SHA-256 `bdcc62eecb5272fee4e4cf3245733e55253b41cd8be7a2df2a19c17caf8f7942`, Drive `1SqneiMuzpB5HqDnhfaKUbPYv3__HZV-r`.
- Seed 1: `outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/receipts/p34_share_calibration_seed_1.json`, SHA-256 `072bdf418e32b7b65b1002cec0505f402efb5ff9f66a7ed26c9d32849a5c46f6`, Drive `1SwrOp4-h9w0wTppHxpph65f2RtYBkLRv`.

## 6. Approval boundary

This document is an assembled lock for review, not authorization to train. Before Mark's approval:

- no P3.4 runner may be created;
- no optimizer may be constructed;
- no training step may execute;
- CONFIRM and EVAL-E remain untouched.

On explicit approval, implementation may add the runner for three sessions: main seed 0, main seed 1, and slot seed 0, all using only the values in this lock. Any change to the task graph, initialization, population, estimator, weights, schedule, thresholds, or arm design is an amendment.

**Mark approval:** pending.
