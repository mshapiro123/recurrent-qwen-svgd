# Phase-2 Option B Data-Unit Audit

Date: 2026-08-06. Status: resolved by strategy before protocol lock. No training.

Resolution: `STRATEGY_TO_CODING_AGENT_OPTION_B_DATA_RESOLUTION_20260806.md`,
Drive `1BkxgDfdLzDAKTiresWTbomY8LOzsx89I`.

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

## Strategy resolution

Strategy authorized a staged dose-then-data intervention:

1. Segment 1 continues the four A2 arms on the existing 41,969-anchor training
   population and measures exposure, train loss, and fixed-train-subset EAL.
2. A new teacher/cache pass targets 140,000 fresh training anchors from new
   documents, with 100,000 as the minimum admissible expansion.
3. Segment 2 adds the fresh anchors at a recorded 1,000-step checkpoint
   boundary. The target is step 4,000, but the actual durable boundary governs.
4. Learning rate is constant at `3e-4` after a 200-step warmup through step
   18,000, followed by a linear cooldown to 10 percent over the last 2,000
   steps. This prevents the splice comparison from being confounded by a
   changing learning rate.
5. The primary causal contrast is the EAL slope over the 2,000 updates before
   versus the 2,000 updates after the fresh-data splice. Curves also report
   cumulative anchor presentations and cumulative distinct anchors observed.

This is one data intervention inside a dose trajectory, not a general
unique-data scaling law. A flat dose segment alone cannot license a bounded
reading. That reading requires a flat fresh-data segment and a small train-eval
gap as specified in the protocol.

The strategy resolution removes the unit and axis blockers. Exact existing-data
hashes are required at the protocol lock. Expanded-data hashes are necessarily
created by the authorized teacher pass and enter through a hash-only amendment
before the splice. No launcher may silently substitute data, models, or a splice
boundary.
