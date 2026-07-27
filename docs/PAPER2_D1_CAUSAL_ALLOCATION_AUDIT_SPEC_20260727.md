# Paper Two D1 Causal Allocation Audit Specification

**Date:** 2026-07-27  
**Status:** frozen before the read-only audit  
**Parent result:** Paper Two D0, interpretation `not_recoverable_at_pilot_scale`

## Purpose

This audit measures the mismatch between D0's binary teacher-disagreement label and the deployment objective. It also constructs the empirical utility labels needed to draft, but not lock, D1. It is post-hoc and cannot change the registered D0 verdict.

## Frozen analysis choices

- Primary checkpoint: post-D0 final-step EMA, SHA-256 `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`.
- Runtime: A100-class, matching the registered D0 final evaluation. Replayed loop-1, loop-4, adaptive answer, and selected depth must exactly equal the banked private D0 rows before the new loop-2/loop-3 labels are admissible.
- Forced depths: 1 through 4.
- Transition label at loop `d`: `helps` only when loop `d` is wrong and loop `d+1` matches the cached 7B teacher; `hurts` only when loop `d` matches and loop `d+1` is wrong; all other transitions are `neutral`.
- D1 prototype action: continue only for `helps`; stop for `hurts` or `neutral`.
- Oracle objective: teacher-match indicator minus `penalty * (loops - 1)`, with shallower depth winning exact utility ties.
- Penalty grid: `0, 0.01, 0.02, 0.05, 0.10, 0.20, 1/3, 0.50, 1.00`.
- Deployable probe: cheap post-loop scalar features only. Teacher-derived features, projected hidden states, and evaluation labels are excluded from model inputs.
- Cross-fitting: five source-row folds, seed `20260727`. For each outer test fold, the following fold is validation and the remaining three folds are fit data. Thresholds are selected only on the validation fold.
- D1 dry run: deterministic source-row sample from label-train, seed `20260727`, capped at exactly 100,000 next-token positions, forced through depth 4.
- Private per-position records and tensor caches remain in Drive. Git receives aggregate receipts and figures only.

## Teacher-confidence limitation

The locked D0 cache contains teacher entropy, the drafter token's log-probability and rank under the teacher, and teacher-to-drafter KL. It does not contain the teacher top-1/top-2 margin on calibration or evaluation rows. The audit therefore stratifies rescue events by the available entropy, rank, and log-probability signals and records `teacher_top1_top2_margin_available=false`. Reloading either teacher to manufacture the omitted margin is prohibited by the single-pass cache contract.

## Non-training contract

The launcher and evaluator perform no backward pass, create no optimizer, mutate no model parameter, and write no checkpoint. The frozen checkpoint is fingerprinted before and after evaluation. Evaluation rows may be read because D0's registered evaluation is complete; all resulting decompositions are labeled post-hoc.

## Interpretation boundaries

- D0 remains `not_recoverable_at_pilot_scale`, scoped to binary teacher-disagreement targets, 4,000 steps, and one seed.
- The audit measures D1 headroom and label balance; it does not authorize D1 training.
- A D1 preregistration must add a `Label-to-objective alignment` section and lock the compute penalty, class weights, policy-level accepted-position guardrail, endpoint, and comparators before training.
