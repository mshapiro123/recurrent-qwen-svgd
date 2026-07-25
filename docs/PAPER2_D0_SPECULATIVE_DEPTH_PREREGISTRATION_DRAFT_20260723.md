# Phase D0 Preregistration - Speculative-Decoding Depth Supervision Pilot

**Draft 1, updated 2026-07-25. Status: build-only. No labeling or training is
authorized.** This draft becomes binding only after markup, a complete
machine-readable lock, and a separate launch authorization.

## Registered question

On natural text, what fraction of a larger same-family teacher's corrections
to a small recurrent drafter is recoverable by one to four adaptive vertical
loops controlled through the trained internal token pathway?

This measures teacher agreement, not reasoning or truth. The secondary systems
question is whether adaptive depth raises speculative-decoding acceptance at
an acceptable expected-loop cost.

## Models

The drafter will be a reviewed raw T1 endpoint. Continuous and stage-reset EMA
states are excluded. The exact choice between seed 0 and T1-lite-R seed 1 is a
post-replication decision; there is no silent substitution.

The teacher remains unresolved at markup: Qwen2.5-7B-Instruct,
Qwen2.5-14B-Instruct, or a calibration-only dual-teacher comparison. Qwen3 is
distribution-compatible only after masking its four added token IDs and
renormalizing. Trace-level Qwen3 compatibility requires retokenization and is
outside D0.

## Data and labeling

Two document-disjoint strata are proposed:

- post-Qwen2.5-cutoff FineWeb-Edu for general text;
- permissively licensed Stack v2 code with retained license metadata.

Label-train, calibration, and evaluation partitions are document-disjoint.
The proposed pooled budget is about 2 million tokens, plus a descriptive
100,000-token in-era contrast. A pre-lock forward-only density probe chooses
the final mix. No trace or think-token corpus enters D0.

At each true-prefix position, accepted means the drafter's greedy token equals
the teacher's greedy token. Record the drafter-token rank and probability under
the teacher, teacher-to-drafter KL, teacher entropy, and rejection-run length.
Only exact match drives labels.

## Depth calibration

Before training, evaluate rejected calibration positions at forced depths one
through four, binned by disagreement KL quartile.

- A graded floor requires depth-4 agreement to exceed depth-1 by at least two
  points in at least two bins. Each graded bin targets the smallest depth
  within one point of its depth-4 agreement. Non-graded bins target depth one.
- Otherwise, use the dynamic target: the first loop matching the teacher,
  capped at four; never-matched positions target four with distillation.

The primary mapping fit is monotone isotonic regression. Linear run-length,
linear log-KL, and saturating forms are descriptive comparisons. Rejection
runs longer than eight tokens are excluded from mapping calibration and
reported as a tail. These thresholds remain markup items, not locked values.

## Proposed training and readouts

The proposed objective combines retained corpus language-model CE,
teacher-distribution KL at the final executed loop on rejected positions, and
control-token CE. Candidate constants are lambda 0.5, equal class weights,
25% synthetic mechanism rehearsal, AdamW, batch one, seed zero, and 4,000
steps. None is locked or executable yet.

Primary readouts:

1. Depth-recoverable fraction: self-halted teacher-match rate minus loop-1
   teacher-match rate among rejected positions.
2. Simulated greedy speculative acceptance uplift, expected loops per token,
   and compute per accepted token.
3. Depth-response and unrecovered-at-four fractions by severity.
4. Loop allocation versus disagreement signals.
5. Every result pooled and split between general text and code.
6. Descriptive ARC depth allocation only, with no QA claim.

Interpretation language is pre-binned: below two points is minimal, two to ten
points partial, and at least ten points strong. These are descriptive bands,
not a pass/fail gate.

## Proposed hard guardrails

- On accepted positions, loop-1 teacher match may not fall more than one point
  below the pretraining drafter.
- The synthetic mechanism check must remain within three points of the chosen
  T1 raw endpoint.

These are draft values and cannot become operational until lock.

## Current build authorization

Permitted now: schemas and manifests, exact-match and signal scorers, added-ID
probability masking, calibration-branch logic, depth-recoverable-fraction
calculation, synthetic fixtures, tests, and a CPU dry-run receipt.

Forbidden now: corpus selection based on observed density, teacher forwards,
GPU labeling, optimizer construction, training, and checkpoint writes. The
machine-readable guard is `training/speculative_depth_d0_spec.py`; it has no
labeling or training launcher.

## Unresolved before lock

1. Teacher size and exact checkpoint hashes.
2. Post-cutoff corpus snapshot, total token budget, and partition hashes.
3. Prelude/coda freeze policy.
4. Direct transfer of P0 constants versus a locked three-cell check.
5. Rehearsal fraction.
6. Calibration thresholds and run-length cap.
7. Standard draft-window sweep for systems comparability.
8. The chosen T1 raw endpoint after replication review.

## Do not claim

Do not call recovered agreement reasoning; claim real speculative efficiency
without deployed wall-clock measurement; generalize beyond teacher-forced
labels; infer knowledge limitation from failure at four loops; make a QA claim
from ARC allocation; or connect D0 results to GRAM and the closed Arm G route.
