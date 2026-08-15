# Paper Two P3.5 Prerequisites: Exact-Reader Repair and Persistence Probe

Date: 2026-08-15
Status: complete, no training; assembled P3.5 lock awaits strategy ratification
Implementation commit: `87bac2a4364284dccb08976aa9b048521cde1469`

## 0. Direct answer

Both authorized no-training prerequisites landed.

1. The suspended causal estimator is repaired. All `4,096/4,096` registered positive-audit rows now reproduce their source token under the exact BF16 serving reader. No row was dropped.
2. Untrained cross-token carry is behaviorally active but did not improve pooled task accuracy. Fresh scratch scored `76/195`; carried scratch scored `75/195`. Carry changed later tokens on `35/195` rows, but only five rows crossed the correctness boundary: two fixes and three regressions.
3. The bounded persistence reading is `NO_FREE_PERSISTENCE_GAIN_ON_THIS_SEED0_DEV_PROBE`. This does not reject trained persistence. It keeps persistent training gated while leaving the nonpersistent P3.5 landing and reader A/B fully viable.
4. No optimizer was constructed, optimizer steps were zero, and CONFIRM and EVAL-E remained sealed.

The next scientific action is strategy ratification of the assembled P3.5 lock. Only after ratification may the two Arm S seeds and the seed-0 Arm R comparison train. The L4 prerequisite session was released after all receipts were copied to Drive.

## 1. Authority and rationale

The governing response is `STRATEGY_P35_CHARTER_RESPONSE_20260815.md`, Drive `1ZzWO3MzkFW5Ph0wAuF5r-ZpdYCzAEi6F`, SHA-256 `3bf476f1db8ebe451d798c941aeef3110129b8d946515125488b7871e0cf7c82`.

P3.5 was opened to answer two questions on the completed P3.4 mechanism:

- Can a controlled landing protocol turn an oscillating late-training endpoint into a stable one?
- Does a detached four-probe control reader improve selection over the current mean reader?

Two prerequisites had to land first. The prior oracle cache disagreed with the serving reader on 153 of 4,096 audit rows, invalidating registered `pi_dir`. Separately, frozen-source direction reuse had already been shown invalid after deployed writes, so any persistent path needed a current-source re-anchoring probe before training.

## 2. Experimental designs

### 2.1 Exact serving-reader repair

Population: the fixed 4,096-row positive audit, selected exactly from the registered 43,204-row strict oracle cache. Every row ID had to be present exactly once. Selection retained audit order and subset every row-aligned cache field together.

Reader: the same BF16 matrix multiplication and argmax used by the deployed audit path. Source tokens and analytic teacher-minus-source LM-head directions were recomputed under that reader.

Gate: source-token identity exactly `1.0`. Dropping mismatches was prohibited. Any optimizer construction, sealed-data contact, or row-population change was prohibited.

### 2.2 No-training persistence probe

Population: deterministic SHA-256 rank under seed `20260815`, up to 128 DEV rows per battery. The panel supplied 128 GSM8K rows and all 67 available MBPP rows, for 195 total.

Checkpoint: seed-0 P3.4 step 4,000, SHA-256 `381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7`, reconstructed through the hash-asserted migrated, P3.3, i1, and P3.4 state chain.

Paired conditions:

- `fresh`: a new scratchpad for every emitted token, the registered P3.4 inference path.
- `carried`: controlled cross-token scratch carry.

Both conditions used greedy decoding, four flow loops, gate ceiling `0.02`, and the same rows. Direction telemetry was re-anchored from the current source token at every generated step. The repaired oracle cache was not used in this probe.

This was a free-mechanism test, not persistence training. It asks whether carry helps without adaptation. It cannot determine whether trained persistence would help.

## 3. Implementation and quality control

Two pre-analysis defects were found and repaired without scientific spend.

1. The nested Colab launcher invoked the runner as a file, leaving the repository root off `sys.path`. It now invokes `python -m colab.run_stage5_paper2_phase3_p35_prerequisites`, with a regression assertion.
2. The repair code incorrectly required the 4,096-row audit to equal the complete 43,204-row oracle cache. Diagnostics showed `4,096/4,096` audit IDs were present exactly once and the source cache contained 39,108 additional eligible rows. The code now subsets all row-aligned fields in audit order and records both source and selected counts.

The exact P3.4 endpoints were also restored from the project durable store to the canonical Pharma Initiatives Drive tree before execution. Both remote MD5 values matched the local files, and the runner rechecked the registered SHA-256 values.

Tests after repair: 13 focused P3.5 and launcher tests passed locally; the same focused suite passed on Colab before analysis.

## 4. Results

### 4.1 Serving-reader estimator

| Measure | Result |
|---|---:|
| Source cache rows | 43,204 |
| Selected audit rows | 4,096 |
| Matched source tokens | 4,096 |
| Mismatched source tokens | 0 |
| Identity rate | 100.0% |

The repaired cache SHA-256 is `294358a7dacc746b733e9f08296494c6f461443a92c093f8019a1dda56422294`. The summary SHA-256 is `ab584b6ba008b0ade9247bee099f9bee4cce02ed1c58de94be68a8bb5c4197e6`.

This closes the estimator blocker. Registered P3.5 `pi_dir` and `pi_dep` may use this cache after the lock is ratified.

### 4.2 Persistence

| Battery | Rows | Fresh | Carried | Net rows | Delta, points | Fixes | Regressions | Changed continuation | Exact paired p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 128 | 43 (33.59%) | 41 (32.03%) | -2 | -1.56 | 1 | 3 | 29 (22.66%) | 0.625 |
| MBPP | 67 | 33 (49.25%) | 34 (50.75%) | +1 | +1.49 | 1 | 0 | 6 (8.96%) | 1.000 |
| Pooled | 195 | 76 (38.97%) | 75 (38.46%) | -1 | -0.51 | 2 | 3 | 35 (17.95%) | 1.000 |

Descriptive paired bootstrap 95% intervals for carried-minus-fresh accuracy were `[-4.69, +1.56]` points for GSM8K, `[0.00, +4.48]` for MBPP, and `[-2.56, +1.54]` pooled. These intervals are post-run descriptive uncertainty, not registered gates.

Carry was not inert: it changed later tokens on 35 rows and triggered 4,023 nontrivial current-source re-anchors. But only five changed rows crossed the correctness boundary. Thirty rows changed their continuation while retaining the same correct/incorrect status.

Figure: `docs/figures/p35_prerequisite_probe_20260815.svg`.

## 5. Interpretation

### 5.1 What is established

- The serving-reader mismatch is fully repaired on the complete registered audit population.
- Controlled carry reaches the generation path and changes outputs.
- Carry provides no free pooled accuracy gain on this seed-0 DEV sample.
- The current-source re-anchoring implementation is active and avoids the previously invalid frozen-source reuse.
- The registered nonpersistent P3.5 landing and reader comparison can proceed after ratification.

### 5.2 What is not established

- The probe does not show that persistence is generally harmful or useless.
- The probe does not test a persistence-trained model.
- The opposite small battery effects are not resolved and should not be narrated as task-specific benefits or harms.
- The larger GSM8K continuation-change rate may reflect longer or more fragile generated chains, but this probe does not causally identify that explanation.
- The result is one seed and DEV only. It makes no CONFIRM, EVAL-E, or generalization claim.

### 5.3 Program consequence

The free-carry hypothesis does not earn a training arm. Persistence remains a future, explicitly trained intervention if strategy later authorizes it. P3.5 should retain its current nonpersistent scope:

- Arm S: stabilized mean reader, seeds 0 and 1.
- Arm R: detached four-probe reader, seed 0.

This is the cleanest next experiment because it isolates landing and reader selection without adding a persistence mechanism that has not demonstrated free benefit.

## 6. Safety and sealed-data accounting

| Contract | Result |
|---|---|
| Optimizer constructed | No |
| Optimizer steps | 0 |
| CONFIRM scored | No |
| EVAL-E scored | No |
| Exact endpoint SHA checks | Passed |
| Exact 4,096-row reader identity | Passed |
| Frozen-source direction reuse | Disabled |

## 7. Receipts

Drive run root: `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_p35_prerequisites_20260815`

- Bindings: `receipts/bindings.json`, SHA-256 `d3235d09a85e2d2bb5a774acd7baa3ecbfa6da727e8373c792ad4f7d906566f6`.
- Repaired cache: `private/serving_oracle/agreement_oracle_directions_v2.pt`, SHA-256 `294358a7dacc746b733e9f08296494c6f461443a92c093f8019a1dda56422294`.
- Repair summary: `private/serving_oracle/agreement_oracle_directions_v2.summary.json`, SHA-256 `ab584b6ba008b0ade9247bee099f9bee4cce02ed1c58de94be68a8bb5c4197e6`.
- Persistence summary: `private/persistence_seed_0/summary.json`, SHA-256 `ca917aaac213451cf4c38c18bd5e25c642e13177b3dc2f8ee584bd056fe90116`.
- Persistence rows: `private/persistence_seed_0/rows.jsonl`, SHA-256 `7dd6882cbdcda62a8be4cc045fcaf0f8fd79cf25dc0f344327a42be2e1550c29`.
- Public aggregate: `outputs/stage5/stage5_paper2_phase3_p35_prerequisites_20260815/summary.json`.

## 8. Questions for strategy

1. Ratify the exact-reader repair and permit the machine lock to promote the v2 cache as the sole P3.5 causal-audit cache.
2. Bank the persistence reading as `NO_FREE_PERSISTENCE_GAIN_ON_THIS_SEED0_DEV_PROBE`, with trained persistence still untested and unauthorized.
3. Confirm that persistence remains outside P3.5 and that the registered Arm S/Arm R matrix is unchanged.
4. If those rulings stand, authorize Mark to set the four lock fields: `mark_ratified`, `locked_before_training`, `training_authorized`, and `status=approved_for_training`.

## 9. Plain-language summary

We fixed the measurement problem completely: every one of the 4,096 audit examples now uses the same token reader as the deployed model, so the next causal measurements are trustworthy.

We also tested whether simply carrying the model's internal scratch state from one generated token to the next helps without training. It changed some answers, especially on GSM8K, but did not make the model more accurate overall. That is useful: persistence is not a free shortcut, so it should not complicate the next experiment. The next clean test remains the planned landing protocol and the mean-reader versus probe-reader comparison.
