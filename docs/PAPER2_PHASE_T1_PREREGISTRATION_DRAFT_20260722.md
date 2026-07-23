# Paper Two Phase T1 Preregistration Draft

**Status:** `draft_not_locked`. No training is authorized.  
**Purpose:** test whether an explicit internal token pathway can causally select
recurrent depth on the controlled synthetic transition family.

## Lineages

Two fresh-base Qwen2.5-0.5B surgery lineages run independently:

1. Full recurrent block plus repaired split bridge and the three new control-token rows.
2. Rank-16 recurrent-block LoRA plus repaired split bridge and the three new control-token rows, with pretrained Qwen weights frozen.

No keeper checkpoint initializes either lineage.

## Explicit non-halting references

The chain-accuracy gate is not defined against an implicit baseline.

- Full-block reference: Phase A Arm A on the immutable Phase A rows,
  `1005/1024` over trained depths 1-8. Canonical receipt:
  `outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json`;
  checkpoint SHA `dc00f7b6...4f71b`.
- Adapter reference: Arm E on the same rows, `1021/1024` over trained depths
  1-8. Canonical receipt:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json`.

Both use full-symbol, question-only, first-completed-response scoring and forced
loop count equal to row depth. The T1 forced-depth and self-halted evaluations
use the same reader and row IDs (row-ID SHA
`14482ca4d1b539172e4ccced6d870818c8658314b7f9680d0fb6e685b0317500`).

## Targets and curriculum

At the reserved control position after each recurrent transition, the target is
`continue` before the row's exact depth and `stop` at that depth. The decision
is intercepted before visible decoding. All three internal symbols remain
masked from user-visible generation.

The training stream contains 70% control-target rows and 30% original
per-loop mechanism rehearsal. Depths 1-8 are trained.

Proposed, not yet locked, per-lineage budget mirrors the established mechanism
curriculum: 500 steps at depth 1, 2,000 at support 1-2, 4,000 at support 1-4,
and two 2,000-step support-1-8 stages, for 10,500 steps. Learning rates are
`2e-5` for the primitive stage and `1e-5` thereafter. Seed 0 is proposed, with
the resulting single-seed limitation stated explicitly.

## Four locked-form gates

A positive reading requires all four:

1. Forced-depth chain accuracy is within three percentage points of the matched lineage's non-halting reference.
2. Self-halted accuracy is within three percentage points of paired forced depth.
3. Continue/stop selection accuracy is at least `0.90` at every trained depth.
4. Causal override is demonstrated in both directions: forced stop at a model-continue transition terminates there, and forced continue at a model-stop transition executes at least one additional loop within the registered maximum.

An answer change alone does not satisfy gate 4. Executed loop count must change
as commanded. A miss on any gate is a registered negative.

## Integrity and boundaries

T0 must pass first. Each run retains the one-loop identity threshold of `1e-3`,
base-hash assertions, a Tier-1 canary hard stop, visible-generation masking,
and requested/executed/selected loop logging.

This experiment does not test natural content-determined depth, broad adaptive
computation, stochastic width, or multi-seed robustness. T2 and width remain
closed until the post-T1 decision point.

## Items requiring Mark's lock

1. Accept or revise the proposed 10,500-step curriculum.
2. Accept seed 0 as a single-seed design or authorize additional seeds.
3. Lock the exact causal-override row count and sampling seed.
4. Change this document and the machine-readable spec to `locked_before_training`; only then may a T1 launcher exist.
