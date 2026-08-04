# Phase-2 Oracle-Selector Headroom Handoff

Date: 2026-08-05  
Status: `complete_descriptive_cpu_only`

## 1. Purpose and design

This receipt prices the maximum benefit that perfect row-level hindsight could
extract from the six banked matched-alpha terminal checkpoints. It chooses the
sidecar on a row exactly when that row's accepted length exceeds the zero-loop
baseline. A second oracle applies the same rule but rejects any selection that
turns a baseline-correct horizon wrong.

The calculation uses 8,031 fixed DEV rows per arm. It performs no model
inference, optimizer steps, or parameter mutation and touches no frozen E1
partition. It is an upper ceiling, not a deployable routing result.

## 2. Results

The common zero-loop mean accepted length was `2.1195719242`.

| Alpha | Mean always-on delta | Mean hindsight-oracle delta | Mean quality-safe delta | Rows selected |
|---:|---:|---:|---:|---:|
| 0.0 | -0.001808 | +0.004805 | +0.004766 | 46.97% |
| 0.5 | -0.002481 | +0.004618 | +0.004582 | 47.06% |
| 1.0 | -0.002543 | +0.005072 | +0.005053 | 46.91% |

For the staged amendment's alpha-0.5 arm, perfect selection changes the mean
from `-0.002481` to `+0.004618`. Relative to the zero-loop mean, the oracle
result is approximately `+0.218%`; the full oracle swing over always-on is
approximately `0.335%`.

Across alpha-0.5 seeds, the oracle selected 3,711/8,031 and 3,847/8,031 rows.
The quality-safe rule removed only 23 and 20 selected rows. It reduced the
oracle delta by only `0.000041` and `0.000030`, respectively, while raising
retention of baseline-correct horizon decisions to exactly 1.0.

## 3. Interpretation

The row-level sign heterogeneity is real: roughly 47% of rows improve even in
these failed, early-stopped checkpoints. However, the effect magnitudes are
small and the harms on the remaining rows are larger in aggregate. Positive
to negative accepted-length mass ratios are below one in every arm
(`0.64` to `0.76`). This explains why always-on use is negative.

The selector is not the next primary bottleneck. Even an impossible
perfect-hindsight selector can recover only about two-tenths of one percent of
the baseline accepted-length scale at alpha 0.5. A practical router would
recover less. The present checkpoints therefore do not contain a large latent
benefit pool that routing alone can unlock.

This strengthens the staged amendment:

1. A1 must first create a materially useful state and increase the magnitude
   of positive row-level effects under a balanced objective.
2. A2 then tests whether the model can use that frozen state.
3. Routing becomes a serious workstream only if the repaired recipe creates
   meaningful oracle headroom and observable inference-time predictors.

The result does not select alpha. Differences among alpha ceilings are small,
and all six source checkpoints were trained under the confounded objective and
trust rail documented by the audit.

## 4. Limitations and do-not-claim boundaries

- The selection rule uses outcome hindsight unavailable at inference time.
- Accepted length is cached, teacher-forced DEV instrumentation, not serving
  throughput.
- The source checkpoints stopped at different steps and do not identify an
  alpha comparison.
- A large selected fraction does not imply a large available gain.
- This receipt neither authorizes a router nor demonstrates that available
  features can predict the oracle decisions.

## 5. Program status and next decision

The read-only selector request is complete. V1d is also already complete and
must be acknowledged rather than rerun. Training remains blocked pending the
exact A1/A2 budgets, calibration-update policy, static-weight solver,
extension allocation, sustained-trust arithmetic, quality-collapse tripwire,
and terminal-slope trigger in the staged preregistration.

## 6. Canonical artifacts

- Result commit: `37625b9f`
- Summary: `outputs/stage5/stage5_paper2_phase2_oracle_selector_headroom_20260805/summary.json`
- Receipt: `outputs/stage5/stage5_paper2_phase2_oracle_selector_headroom_20260805/receipt.md`
- Source exact-row tensors: Drive audit private directory, hashes recorded per
  arm in the summary
