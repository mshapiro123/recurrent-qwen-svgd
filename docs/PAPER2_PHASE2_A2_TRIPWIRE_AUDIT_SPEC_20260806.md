# Phase 2 A2 Gradient-Tripwire Audit

Date: 2026-08-06

Status: authorized descriptive audit before any further A2 update

## Purpose

Both A2 full-system seeds stopped before update 238 on the same scheduled batch
because the finite total gradient norm exceeded the static catastrophe threshold.
This read-only audit determines whether that event is an isolated data-specific
spike, a shifted gradient distribution, or evidence of a genuinely unsafe update.
It cannot clear the stop or authorize training.

## Frozen inputs

- The two full-system step-237 checkpoints from the landed resumed matrix.
- The exact registered A2 training partition and row seed `20260805`.
- The alpha-0.5 canonicalizer and the frozen Stage 0A DEV partition.
- The registered seed-specific static loss weights and catastrophe thresholds.

Every checkpoint is asserted against the landed public receipt before use. Source
checkpoint hashes are checked again after the audit.

## Measurements

1. Reconstruct scheduled attempts 228 through 248. Attempt 238 must reproduce the
   rejected row hash stored by both seeds.
2. Measure total gradient norms throughout that window at each seed's step-237
   state.
3. Measure the same norm on the existing 51-batch matched directional estimator
   and report its full distribution and old-threshold exceedance rate.
4. On attempt 238, decompose raw and weighted gradients by loss and module group,
   including pairwise loss-gradient cosines.
5. Report batch composition relative to the training population: position buckets,
   base/teacher top-token agreement, hidden RMS, target RMS, and candidate count.
6. Simulate exactly one AdamW update from the saved optimizer state in memory.
   Report total and group-relative parameter displacement, finite-state checks,
   and pre/post DEV acceptance and retention. Restore the module immediately and
   assert its trainable hash is unchanged.

## Interpretation boundary

This audit is descriptive. It records whether the static threshold tracked current
optimization geometry and whether the rejected update appears mechanically benign
or harmful. It does not label an A2 endpoint, change a gate, clip a gradient, skip
a row, or persist an optimizer update. Any continuation requires a separate
strategy amendment written after this receipt lands.

