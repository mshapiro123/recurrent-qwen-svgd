# Paper Two Phase 2 A2 Step-200 Resume Amendment

Date: 2026-08-05

Status: `locked_before_a2_resumed_training`

This narrow amendment changes A2 stopping semantics only. The optimizer, loss
weights, training rows, directional contract, endpoint gates, extension rule,
and verdict remain exactly as locked in
`PAPER2_PHASE2_A2_LOCK_AMENDMENT_20260805.md`. The commit containing this
document and the matching machine-readable registration is the lock. No resumed
A2 optimizer update may occur before that commit.

## 1. Source classification

The first A2 matrix is classified as `protocol_stop_at_step_200`, not
`budget-limited`. Both full arms were below the endpoint point-retention threshold
at step zero, improved through step 200, remained above the absolute Wilson
floor, and passed every directional audit. No A2 inference is drawn from the
step-200 values.

## 2. Exact resume sources

All four saved step-200 states resume. They are copied into a new run directory
before any update, and each copy is asserted byte-identical to its registered
source SHA-256.

| Arm | Source SHA-256 |
|---|---|
| seed 0 full A2 | `75e335fe9771c2dce13d48f9cd68275509dded8f6ca320b086def1f45b6376b4` |
| seed 0 draft-only control | `3e85af090ac6e19833358dfa2f5d1891be5e898856318e07aa01fb4880aa7c29` |
| seed 1 full A2 | `1cf8b3d3cdc80ba3f58aa30b7b3619aebfa2c3af08e881fba7222e721b0690d4` |
| seed 1 draft-only control | `78c817bea3909ad5d13028fa917ce07540dfe5d20dd46190ad8821b33132b3a1` |

The two full-arm resume payloads may clear only the exact historical abort
reason `quality_noninferiority_two_consecutive_evaluations`, at exactly step
200, after their registered source hashes pass. Their obsolete consecutive
quality-failure counters reset to zero. No other abort reason is resumable.

## 3. Quality semantics

The endpoint gate is unchanged: point retention at least 0.997 and Wilson 95%
lower bound at least 0.990. It is evaluated for qualification at step 1,000 and
at the endpoint of any registered extension.

During training, each arm is grounded to its own saved step-zero retention:

- two consecutive evaluations below `step_zero_retention - 0.003` stop before
  further training and write receipts;
- a Wilson 95% lower bound below 0.990 at any evaluation stops immediately;
- a negative least-squares retention slope over the latest three consecutive
  evaluations writes a warning and does not stop.

The endpoint threshold is not used as a during-training tripwire.

## 4. Continuation and authority

The four arms continue to step 1,000. The registered pair-matched extension
decision, endpoint gates, oracle-headroom recomputation, control-superiority
test, correlation table, and monotonicity analogue then apply unchanged. V1d is
attached to the completion handoff.

- Strategy resolution Drive ID: `16aIjqnCOT0LzzXYoI7rihUVK2ydpsn1G`
- Strategy resolution bytes: 6,269
- Strategy resolution SHA-256:
  `62355265df2055df6480a9618c5f83694678f5e146d056f07bc6073bb963cff7`
- Source result commit: `10f3a3df34672b02ab82cf0c63d378be32bd489f`
- Source public summary:
  `outputs/stage5/stage5_paper2_phase2_a2_20260805/summary.json`
