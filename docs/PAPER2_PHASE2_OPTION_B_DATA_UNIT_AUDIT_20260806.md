# Phase-2 Option B Data-Unit Audit

Date: 2026-08-06. Status: implementation audit before protocol lock. No training.

## Finding

The Option B charter names an existing full lattice of approximately 190,000
anchors. The banked Stage 0A receipt does not contain that population.

The canonical Stage 0A summary records:

- `anchor_count = 50,000`;
- four horizons per anchor;
- `boundary_sample_count = 200,000`;
- the document-isolated A2 split: 41,969 training anchors and 8,031 evaluation
  anchors.

The 200,000 quantity is a horizon-sample count, not an anchor count. A2 batches
anchors and supervises all four horizons. Reinterpreting the existing 200,000
horizon samples as approximately 200,000 anchors would silently change the
experimental unit, epoch arithmetic, and data-scale claim.

## Consequence

At batch size 128, 20,000 optimizer steps contain 2,560,000 anchor
presentations. This is approximately 61 epochs over the existing 41,969-anchor
training split, not approximately 13 epochs. Obtaining approximately 190,000
training anchors while preserving the fixed 8,031-anchor evaluation slice
requires a Stage 0A extension and new teacher/cache work. The exact target and
partition hashes must be amended before the protocol can lock.

The within-run x-axis also requires accurate naming. Twenty checkpoints from a
single fixed training population measure performance against cumulative anchor
presentations, or training dose. They do not independently identify a
unique-data scaling law. A true unique-data curve requires nested unique-anchor
populations or an equivalent design that varies unique data separately from
optimization exposure.

## Required strategy resolution

1. Authorize a new teacher/cache pass to create the intended expanded anchor
   population, or redefine Option B as a dose-only continuation on the existing
   41,969 training anchors.
2. Name the primary curve either `EAL versus cumulative anchor presentations`
   or authorize a nested-population design for a unique-data scaling curve.
3. Supply the resolved training-population count, data hash, document-partition
   hash, and exclusion proof before `locked_before_training`.

No substitution is permitted in the launcher. The protocol draft carries this
as a lock blocker.
