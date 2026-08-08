# Phase 2 E1 Confirmation Preregistration

Date: 2026-08-08. Status: **locked before E1 scoring; evaluation authorized only under this registration**.

This registration implements the ratified E1 charter without spending EVAL-D. The commit containing this completed preregistration is the lock. The Option-B-compatible EVAL-D cache, sparse-support QC, readiness receipt, and endpoint semantic identities all landed before this lock; each records `read_once_scoring_spent=false`.

## 1. Question

Does the small full-system advantage measured on DEV replicate on a document-disjoint frozen partition when the four final Option B endpoint checkpoints are evaluated without any additional training?

E1 is a one-shot confirmation pass. It is not another optimization stage, a selector experiment, an alpha sweep, or a quality-repair leg.

## 2. Frozen systems

The following final Option B checkpoints are the only admissible systems:

| Seed | Arm | File SHA-256 | Semantic trainable-state SHA-256 |
|---:|---|---|---|
| 0 | full system with writeback | `c1f5a6f217342ad721267a08d16c1bca75c8308d03f471db9d28ff3f319c777f` | `d47bd4094846c4ac540cbeb418fe24e05734557d9f9e3fa226e7f0c2b7f066f5` |
| 0 | drafter-only control | `8c9a7f6573bd268d67592b271a1b10a37c1f882681dc20efdbd8e9a5232bd681` | `e190080c98529fcb8e8adafad6c6439daa26c6959b799b94d23d198daf276a48` |
| 1 | full system with writeback | `ccebda5c0b4bb1832194f690075b0be9ac1a96c557e63978ebb97a8632d278f7` | `3ad55c17fe82afa0cf3b66533ba6ffec7864b8e13db68fb9dfc287b7a513ed31` |
| 1 | drafter-only control | `b26ca18e76fc60a622d6056b2957d31ee37e0c6c26dde88a3250b9bbd54a2424` | `bc2a0034687dcc64b8bebb7a9ca24bd351152d638a3b5c898ba7b5cb1e130181` |

All model parameters and buffers are frozen. The runner must hash each file and the sorted trainable-state tensor payload before evaluation, then repeat both checks after evaluation. No optimizer may be constructed. No parameter may have `requires_grad=True`.

## 3. Evaluation population and cache contract

The population is EVAL-D only. It must be frozen, document-disjoint from every development and training population in the lineage, generated without endpoint scoring, and hashed before this preregistration is locked. EVAL-E remains untouched.

The scorer and row schedule are byte-identical to the final Option B endpoint evaluator except for the partition swap. In particular, EVAL-D must provide, for four horizons per anchor:

- student hidden states;
- the same sparse candidate union and mask;
- student and 14B teacher candidate log probabilities plus tail masses;
- 14B top-128 IDs and log probabilities for the probe readout;
- canonical targets produced with the frozen learned-mixture RRR canonicalizer;
- document IDs, strata, positions, and stable anchor keys.

The existing pre-window EVAL-D design is **not sufficient for this confirmation**. It specifies a 7B token cache and own-base boundary features, whereas the Option B evaluator consumes the 14B four-horizon lattice and canonicalizer payload above. A score-blind EVAL-D cache amendment must land before lock. This is a schema correction, not a scoring pass.

The pre-window EVAL-D files were never materialized at the registered durable path. The cache target therefore first materializes EVAL-D data only under the original pre-window recipe: the pinned corpus revisions, 200,000-token budget split equally across general and code, and partition seed `20260731`. It additionally quarantines every subsequently created training and DEV document set before drawing rows. This step loads only the pinned tokenizer, computes no model output, leaves EVAL-E untouched, and freezes the resulting data and document hashes before teacher caching. Because no EVAL-D membership artifact previously existed, this is the first and only materialization of the registered partition recipe, not a replacement selected after observing scores.

The frozen population contains exactly 8,000 anchors, 4,000 general and 4,000 code. Selection uses the existing Stage 0A non-overlapping rule: enumerate every eligible `(row_id, prediction_position)` whose four-horizon span stays within one row; rank within stratum by SHA-256 of `20260808:row_id:prediction_position`; greedily retain non-overlapping four-position spans; then take the first 4,000 per stratum. The row-major reorder used for efficient teacher passes occurs only after selection and cannot change membership. Membership is asserted against the frozen EVAL-D data and document-partition hashes.

The readiness chain records 8,000 anchors across 132 documents, 4,000 anchors per stratum, all required data and cache hashes, source-model revisions, the canonicalizer identity, an 18.35% cascade fraction, the admission-ledger identity, and zero overlap with every quarantined document set. The sparse-support audit reports finite integrity metrics and explicit support-mismatch rates; it does not expose an E1 outcome.

Cache generation is strictly score-blind. The base 0.5B forward may materialize only the hidden states and sparse log-probability tensors required by the frozen Option B cache schema. No Option B endpoint checkpoint is loaded, and no correctness, EAL, retention, acceptance, arm comparison, or student-teacher quality aggregate is computed or emitted. Public products are limited to hashes, counts, pinned model revisions, cascade fraction, and integrity telemetry. The private tensor cache and admission ledger are infrastructure, not an evaluation result.

## 4. Fixed evaluation procedure

1. Verify the preregistration lock, rule inventory, cache readiness receipt, endpoint hashes, endpoint semantic digests, and EVAL-D hashes.
2. Verify `read_once_scoring_spent=false` and atomically create a one-pass execution lease before loading an endpoint.
3. Load the frozen 0.5B embeddings and the frozen 14B embedding table used by the Option B probe path.
4. Evaluate all EVAL-D anchors in the frozen row order for seed 0 full, seed 0 control, seed 1 full, and seed 1 control. The same cache and row order are used for every arm.
5. Persist row-level metrics before aggregation. The row receipt must include document ID, anchor key, EAL, zero-loop EAL, retention indicators, probe KL, probe top-1, and gate statistics.
6. Reproduce the public arm summaries from the row receipts within `2e-6`.
7. Run the fixed document bootstrap and scripted verdict once.
8. Mark the read-once lease spent even if a post-start process failure occurs. A failed pass is not silently rerun; recovery requires strategy review of whether any score-bearing data were exposed.

## 5. Primary endpoint

For each seed separately, compute the paired full-minus-control difference in mean expected accepted length (EAL). Construct a two-sided 95% paired percentile cluster-bootstrap interval by resampling documents with replacement and applying identical document multiplicities to the full and control arms. Use 10,000 replicates and seed `20260808`.

The primary pooled estimate gives general and code equal weight by construction: 4,000 anchors from each stratum. The primary endpoint passes only if the lower interval bound is strictly greater than zero in **both** seeds. Two training seeds are treated as two required replications, not as a sample from a seed population.

The DEV effect size of 0.35% to 0.50% is an expectation stated descriptively, not a confirmation margin. The former 1% target is retired as a gate and must be reported as an unmet exploratory aspiration.

## 6. Quality endpoint

For each full-system seed, define the baseline-correct population using the zero-loop prediction against the 14B teacher target. Retention is the fraction of those predictions that remain teacher-correct after writeback.

Quality passes only if, in each seed:

- point retention is at least 99.5%; and
- the Wilson 95% lower bound is at least 99.0%.

The earlier 99.7% diagnostic aspiration is reported transparently and is not a gate. The expected Pareto point from DEV, approximately +0.35% to +0.50% EAL against -0.35 to -0.45 retention percentage points, is descriptive.

## 7. Scripted readings

| Primary effect | Quality | Reading |
|---|---|---|
| passes in both seeds | passes in both seeds | `CONFIRMED_WITH_MEASURED_PARETO`: the causal writeback effect replicates on EVAL-D at the registered quality floor. |
| passes in both seeds | fails in either seed | `EFFECT_REPLICATES_QUALITY_BOUNDARY_FAILS`: the effect is real but the endpoint is outside the ratified preservation region. |
| fails or is mixed | passes in both seeds | `MECHANISM_NOT_CONFIRMED`: quality is retained but the full-over-control advantage does not replicate in both seeds. |
| fails or is mixed | fails in either seed | `BOTH_CONFIRMATION_REQUIREMENTS_FAIL`: neither the joint effect nor quality claim is supported. |

A positive point estimate with an interval touching zero is a primary failure. A pass in only one seed is mixed and does not support the positive confirmation sentence.

## 8. Registered secondary analyses

All are descriptive and cannot rescue the primary endpoint:

- full-system EAL relative to zero-loop;
- absolute and relative writeback increment, full minus matched control;
- probe KL versus EAL, probe top-1 versus EAL, and probe KL versus probe top-1 correlations;
- per-stratum and per-position-bucket effects;
- a DEV-mixture-reweighted full-minus-control estimate, using the immutable DEV general/code anchor proportions defined by the Stage 0A manifest and document-partition seed `20260804`, for direct comparison with the Option B 0.351% and 0.496% results;
- bridge-gate use;
- quality-safe oracle headroom;
- the 1% exploratory target and 99.7% diagnostic aspiration;
- the alpha=0.5 design prior and the deferred alpha matrix.

## 9. Read-once and rule doctrine

E1 has no continuous training shapers. The only armed rules are integrity cliffs: wrong lineage, wrong partition, overlap, cache/hash mismatch, missing rows, scorer drift, model mutation, non-finite output, or evidence that EVAL-D was previously scored. Endpoint thresholds are readings, not process aborts.

No population edits, threshold changes, additional arms, or reruns are allowed after first score exposure. EVAL-E is not a replacement partition.

## 10. Resource note

Cache generation is separate from E1 scoring. The Option-B-compatible EVAL-D cache requires the pinned 14B and cascade teacher path and an A100 80GB class runtime plus local scratch. The final E1 scorer is evaluation-only. No endpoint-bearing score-blind memory preflight was run, so this lock conservatively requires an A100 80GB runtime.

## 11. Lock closure

All five lock blockers are closed. The machine registration carries the EVAL-D hashes, DEV-mixture weights, exact evaluator and rule-inventory hashes, four endpoint file and semantic identities, and the conservative A100 80GB requirement. The final score-blind readiness checker returned `ready_to_lock=true` with no blockers. At lock: `locked_before_e1_scoring=true`, `e1_evaluation_authorized=true`, and `read_once_scoring_spent=false`.

## 12. Do-not-claim boundaries

- Do not call two seeds a population-level replication over seeds.
- Do not call alpha=0.5 selected or optimal.
- Do not claim a 1% effect.
- Do not claim quality neutrality; report the measured Pareto point.
- Do not generalize beyond EVAL-D, this model family, this teacher target, or batch/evaluation protocol.
- Do not interpret a secondary analysis as rescuing a failed primary endpoint.
