# Handoff: E1 Confirmation Draft and EVAL-D Readiness Finding

Date: 2026-08-08. Status: draft complete; E1 not locked; no frozen scores exposed.

## 0. Bottom line

The ratified E1 charter has been translated into a human preregistration draft, machine draft, evaluation-only rule inventory, and executable readiness checker. All four final Option B endpoint hashes match the landed matrix receipt exactly.

E1 cannot be locked yet. No EVAL-D freeze receipt has landed, and the older pre-window EVAL-D implementation is not compatible with the Option B scorer. It was built for a 7B token cache plus own-base boundary features. E1 requires the same four-horizon 14B sparse lattice, probe targets, student states, and learned-mixture canonicalizer coordinates used by Option B. Treating the old format as ready would silently violate the ratified byte-identical-evaluator requirement.

This is a cache-schema precondition, not a negative scientific result. The read-once score remains unspent.

## 1. What was built

- Human preregistration draft: `docs/PAPER2_PHASE2_E1_CONFIRMATION_PREREGISTRATION_DRAFT_20260808.md`.
- Machine draft: `training/paper2_phase2_e1_confirmation_preregistration.draft.json`.
- Rule inventory: `training/paper2_phase2_e1_confirmation_rule_inventory.json`.
- Pure readiness contracts: `training/paper2_phase2_e1_confirmation.py`.
- Readiness CLI: `eval/check_paper2_phase2_e1_readiness.py`.
- Current receipt: `outputs/stage5/stage5_paper2_phase2_e1_preregistration_20260808/readiness.json`.
- Strategy charter and ratification mirrored byte-for-byte in `docs/`.

Focused validation: eight tests pass, including rejection of the legacy 7B-only schema and acceptance of a synthetic complete, unscored Option-B-compatible receipt.

## 2. Locked scientific content already translated

The draft records:

1. The four final Option B checkpoints, frozen, with no training in E1.
2. Primary endpoint: paired full-minus-control EAL, two-sided 95% document-bootstrap interval strictly above zero in each seed.
3. Quality: retention at least 99.5% and Wilson 95% lower bound at least 99.0% in each full-system seed.
4. Ten thousand paired document-bootstrap replicates at seed 20260808, with identical document multiplicities across arms and seeds.
5. The 0.35% to 0.50% DEV effect as descriptive expectation only.
6. The 1% target retired as a gate and reported as an unmet exploratory aspiration.
7. Alpha 0.5 as an unselected design prior; alpha comparison deferred.
8. Four pre-written joint effect/quality readings.
9. Read-once discipline and integrity-only tripwires. Endpoint thresholds do not stop execution.

## 3. Endpoint lineage verification

| Seed | Arm | Final checkpoint SHA-256 |
|---:|---|---|
| 0 | full | `c1f5a6f217342ad721267a08d16c1bca75c8308d03f471db9d28ff3f319c777f` |
| 0 | control | `8c9a7f6573bd268d67592b271a1b10a37c1f882681dc20efdbd8e9a5232bd681` |
| 1 | full | `ccebda5c0b4bb1832194f690075b0be9ac1a96c557e63978ebb97a8632d278f7` |
| 1 | control | `b26ca18e76fc60a622d6056b2957d31ee37e0c6c26dde88a3250b9bbd54a2424` |

These are the final Option B endpoint hashes, not the earlier A2 source hashes.

## 4. Why the old EVAL-D format cannot be used

The Option B scorer consumes these fields:

- `student_hidden`;
- `target_centered_raw` plus the frozen whitening buffers;
- candidate IDs and mask;
- student and 14B teacher sparse log probabilities plus tail masses;
- 14B teacher top-k IDs and log probabilities;
- document, stratum, position, and canonicalizer decoder fields.

The old EVAL-D generator instead emits a D0-style 7B next-token cache and hidden states after layers 6, 18, and 24 of the post-D0 model. Those artifacts answer a different diagnostic question. They cannot be converted into the Option B payload without the pinned teacher/canonicalizer pass.

## 5. Proposed score-blind cache amendment

Recommended design, to be ratified before the GPU pass:

1. Freeze the already specified 200,000-token EVAL-D document partition, 50/50 general and code, excluding every training, DEV, and prior evaluation document.
2. Select 8,000 deterministic non-overlapping anchors, 4,000 per stratum, with horizons 1 through 4. This closely matches the 8,031-anchor DEV comparison while keeping the confirmation population independently frozen.
3. Run the pinned Stage 0A/Option B cache path without loading any endpoint: student 0.5B, teachers 7B and 14B, the registered 32B cascade, sparse union scoring, and all-admitted-anchor 14B states.
4. Apply the already frozen learned-mixture RRR canonicalizer at alpha 0.5 to materialize the exact evaluator payload.
5. Publish hashes, counts, model revisions, document exclusions, cache field inventory, and an unspent atomic read-once state. Publish no EAL, retention, or endpoint score.
6. Run the readiness checker. Only `ready_to_lock=true` permits filling and committing the final preregistration.

Expected hardware: A100 80GB for the pinned teacher path, high system RAM, and local scratch. E1 scoring itself is expected to fit A100 40GB after a score-blind one-batch memory preflight.

## 6. One strategy decision before the generator is locked

Ratify or replace the proposed EVAL-D anchor count of 8,000, balanced 4,000/4,000 across general and code. The old freeze specification fixed token count but not the Option B anchor population because that evaluator did not yet exist. Anchor count affects confirmation precision and therefore belongs in the score-blind cache amendment before generation, not in a post-generation judgment.

The recommendation is 8,000: it is matched to the established DEV scale, should provide comparable document-bootstrap precision, and does not spend compute on an unnecessarily larger teacher cache. The actual document count and baseline-correct `n` remain measured properties of the frozen data and are transcribed before the lock.

## 7. Next sequence

1. Strategy ratifies the cache amendment and anchor count.
2. Coding builds and launches the score-blind EVAL-D Option B cache target.
3. Readiness lands green; no endpoint score has been exposed.
4. Fill hashes, counts, scorer SHA, and resource telemetry in the machine preregistration.
5. Commit the completed preregistration as the lock.
6. Run E1 exactly once and return the scripted verdict.

No additional Option B training, alpha sweep, selector work, quality repair, or EVAL-E contact is authorized in this sequence.
