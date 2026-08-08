# Phase-2 Option B Document-Bootstrap Audit Method

Date: 2026-08-08. Status: fixed before the audit runs. This document completes
an estimator specified in the locked Option B protocol and changes no training
result, threshold, source row, or model lineage.

## Reason

The locked protocol requires the positive second-half exposure slope used for
E1 support to have a document-bootstrap 95 percent interval excluding zero. It
also requires document-bootstrap intervals for the pre-splice dose and
post-splice fresh-data slopes. The landed matrix preserved the required row
receipts but its scripted reading used only a positive slope point estimate.

The source receipt remains immutable. This audit produces a separate corrected
receipt and reports the original and corrected readings side by side.

## Inputs and boundaries

- The canonical Option B four-arm matrix at public commit `ce79e913`.
- Saved `rows_fixed_evaluation_step_*.pt` receipts for all four arms.
- The Stage 0A sample manifest, used only to reconstruct the locked document
  blocks and the 8,031-row fixed evaluation order.
- No model load, optimizer, backward pass, training row, or confirmatory
  partition access.

Every consumed row file and the public source summary are SHA-256 hashed into
the output receipt. Row means must reproduce the public checkpoint means within
`2e-6` before any interval is interpreted.

## Fixed estimator

- Bootstrap unit: document cluster, preserving every anchor from a sampled
  document.
- Pairing: the same document multiplicities are used across checkpoints, full
  versus control arms, and both registered seeds.
- Replicates: 10,000.
- Random seed: 20260808.
- Interval: percentile 95 percent interval.
- Dose slope: EAL change per 1,000 updates from step 2,000 to step 4,000.
- Fresh-data slope: EAL change per 1,000 updates from step 4,000 to step 6,000.
- Second-half slope: ordinary least-squares EAL slope over all checkpoints from
  step 10,000 through step 20,000, expressed per 1,000 updates.
- Writeback growth: change from step zero to step 20,000 in the paired
  full-minus-control EAL increment.
- Late writeback slope: OLS slope of the paired full-minus-control increment
  from step 10,000 through step 20,000.

The two-seed aggregate is a descriptive mean conditional on the two registered
seeds. Its bootstrap resamples documents and does not supply inference over a
population of training seeds.

## Reading

The one-percent endpoint alternative remains the point-estimate threshold in
the locked protocol. Otherwise, E1 support for a seed requires the lower bound
of that seed's second-half full-system slope interval to be strictly above
zero. E1 support in both seeds requires that condition in both independently.

Fresh-data and dose intervals are called separated only when the lower bound of
the fresh-data interval exceeds the upper bound of the dose interval. The
fresh-minus-dose contrast interval is also reported, but it does not replace
the locked separated-interval wording.

Growing full-minus-control increment in both seeds retains writeback for E1 as
specified. Its interval is reported to quantify stability, not to add a new
gate after observing the matrix.
