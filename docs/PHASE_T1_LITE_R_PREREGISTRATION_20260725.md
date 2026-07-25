# Phase T1-lite-R Replication Preregistration

**Locked 2026-07-25 before training.** This is an amendment to the locked
T1-lite preregistration. The seed-0 registered negative remains final. No
training step may precede this document and its machine-readable contract.

## Basis

The seed-0 EMA primary failed. The raw final-step secondary passed allocation,
exact selection, and all 5,632 causal interventions, but missed the preservation
floor by eight rows. This invokes the locked seed-1 replication rule. The
read-only EMA audit localized the endpoint divergence to the recurrent block
and found a sharp transition along the raw-to-EMA interpolation.

## Registered amendment

Exactly one policy factor changes: raw final-step weights are primary.
Continuous EMA at 0.999 and a stage-reset EMA at 0.999 are passive descriptive
shadows. The stage-reset shadow is copied from raw at the start of each support
stage. Neither shadow affects gradients, optimizer updates, stopping, or the
registered verdict.

The training seed is 1 because this is the registered replication. All other
T1-lite constants, data hashes, curriculum stages, learning rates, optimizer,
losses, liveness rule, four gates, forced-versus-self-halted evaluation,
descriptive baselines, and extrapolation analyses are unchanged.

## Stage artifact policy

At steps 500, 2,500, 6,500, 8,500, and 10,500, atomically save and hash raw,
continuous-EMA, and stage-reset-EMA trainable states. Copy each artifact to
Drive. The run is incomplete and cannot be scored unless an end-of-run manifest
verifies all fifteen states. This policy applies to future staged runs.

## Expected readings

- A seed-1 raw pass supports token-pathway halting on trained depths, reported
  with both raw seeds and the complete seed-0 EMA failure history.
- Another small preservation miss with exact selection supports exact learned
  depth control at a small reproducible preservation cost.
- Failure of seed-1 selection makes the seed-0 acquisition seed-dependent.
- EMA-shadow behavior is descriptive only and cannot alter any gate.

## Authorized read-only seed-0 extensions

Layer-group swaps within the recurrent block and per-depth interpolation
breakdowns are authorized. They cannot change the seed-0 verdict or seed-1
gates.

## Do not claim

Do not claim the EMA implementation was mathematically wrong, identify a
seed-0 stage where lag began, claim extrapolation beyond trained support, or
use a shadow endpoint for a registered conclusion.

## Lock record

The original machine-readable lock is
`outputs/stage5/stage5_paper2_t1_lite_preregistration_20260724/preregistration.json`
with SHA-256
`4e55e946a8019d2c0c278bfaff2e76cd97b3efb7822b954b2cb74a539c037cba`.
The replication contract is
`outputs/stage5/stage5_paper2_t1_lite_r_preregistration_20260725/preregistration.json`.
