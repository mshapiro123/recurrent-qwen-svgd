# Phase G A0 Margin Lock

This document records the strategy decision made before corrected Phase G
training. The machine-readable authority is
`STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json`.

## Surface And Primary Test

- Held-out posterior-control surface: 32 repeated-prompt base problems and
  106 target variants, with every variant sharing a prompt/table/start/depth
  only with its sibling target chains.
- Primary statistic: per-variant K=1 exact selected-target fidelity of the
  posterior teacher minus the prior on identical rows.
- Required absolute teacher fidelity: at least `0.60`.
- Required paired lift: at least `+0.15`, with a one-sided exact paired sign
  test at `alpha=0.05`.
- Required target switching: at least 24 of 32 groups must have two or more
  distinct posterior-teacher first predictions across their target variants.
- Validity is reported as a sanity check only. A posterior K=1 validity drop
  larger than `0.05` is not itself an A0 gate.

The group-level fidelity result remains reported to describe correlation within
prompt groups, but it is not an additional pass criterion.

## Locked Training Contingency

The primary arm uses KL coefficient `1e-3`, original EMA settings, frozen
deterministic keeper, base-problem-uniform sampling, and the unbounded
injection-scale parameterization. The recurrent block is frozen and training
asserts zero block gradients on every step.

Only if the primary arm blocks, the runner launches exactly one confirmation:
same seed and all other settings, with KL coefficient `1e-4`. No further KL,
optimizer, or seed sweep is authorized. If both arms block, corrected guided
width closes without a coverage rerun. If either passes, its coefficient is
the only coefficient inherited by A1.

## Launch Preconditions

- Train/control `base_problem_id` sets must be disjoint.
- The 32-group/106-variant control manifest must equal the committed lock.
- Original A1 coverage test/calibration manifests and deterministic test-row
  IDs are regenerated/loaded and verified before A0 consumes GPU time.
- The original entropy target `0.1432` and its plus/minus 10 percent matching
  band are recorded, though A0 does not use temperature sampling.

## Scope

This remains the corrected branching curriculum. It does not switch to the
non-injective abduction family. A1 and any later coverage claim stay closed
until A0 passes. Depth-4 K=1 is recorded rather than capped in A0; a later A1
breach makes a cap or penalty mandatory before any G-beta proposal.
