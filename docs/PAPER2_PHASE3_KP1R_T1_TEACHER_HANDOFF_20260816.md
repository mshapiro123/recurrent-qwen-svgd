# Paper Two Handoff: KP-1R Knowledge Readout and Teacher Fingerprints

**Date:** 2026-08-16
**Status:** complete, score-only, DEV only
**Authority:** `STRATEGY_KP1_T1_RESPONSE_20260816.md`, Drive `1yh0Y6x_2IV6AXCYKgOh0AM5R5-GylHFF`, 11,039 bytes, SHA-256 `2d72b9c59be3f091bcbb9592c7292f9760d598870905bd3b43ae73f141ba6f4f`
**Registered lock:** `training/paper2_phase3_kp1r_t1_teacher_lock.json`
**Seal status:** CONFIRM untouched; EVAL-E untouched; zero optimizer steps

## 0. Executive verdict

This wave returned two different answers to two different questions.

1. **The registered answer-content readout did not pass.** At the strongest predeclared substrate surface, the repaired linear probe exceeded the train-derived frequency control by 7.16 percentage points pooled, but its 95% interval crossed zero (`-3.08` to `+17.39`) and the matched permutation control was not significant (`p=0.172`). At the predeclared loop-4 recurrent surface, the pooled margin was negative (`-2.81` points; 95% interval `-12.77` to `+7.07`). The required pooled-and-macro gate failed at both surfaces. This is **not evidence that knowledge is absent**. It is evidence that the registered linear readout did not establish that the answers were present but unread.

2. **The teacher-fingerprint result was strongly positive.** Student and 14B teacher states share reproducible, basis-invariant item geometry. A split-fit map from the seed-1 student layer-6 state to teacher layer 12 retrieved the matching held-out item at top 1 on 42.20% of rows and in the top 10 on 88.54%, versus chance rates of 0.244% and 2.439%. Seed 0 replicated at 40.73% and 85.61%. The result survives a post-hoc within-battery row-permutation sanity check. This supports a teacher-fingerprint retrieval spine, but it does not show that the small model contains the teacher's answer.

The combined reading shifts the Stage 2A prior toward **content or lookup augmentation** while retaining expert-access as a secondary possibility. The data say that states can be translated and matched across models much more clearly than they say that the missing answer can be decoded from the small model alone.

## 1. Questions and rationale

### KP-1R

The original KP-1 probe was invalid because its first-token target was degenerate on generative tasks. KP-1R repaired the estimand:

- MCQ batteries retained the answer-choice token.
- GSM8K and Tier-1 used the first answer-bearing token for the cached rung.
- The strong rung scored the normalized answer token sequence under teacher forcing.
- MBPP was excluded from the primary decision and retained only as exploratory sequence log likelihood.
- Every reported probe was compared with a train-derived, battery-specific frequency control.
- The registered decision was the probe-minus-frequency margin, with pooled and battery-macro bootstrap intervals.

The positive gate required both pooled and battery-macro margins to be above zero with both 95% intervals excluding zero. The strict joint gate was chosen to prevent benchmark composition or frequent labels from being mistaken for recoverable answer content.

### Teacher fingerprints

Raw coordinates cannot be compared across independently trained state spaces because the scratch representation has an orthogonal gauge freedom. The registered analysis therefore used:

- linear CKA and principal angles for basis-invariant geometry;
- a 60/40 alignment-fit/evaluation split;
- teacher PCA to 128 dimensions;
- frozen orthogonal Procrustes fitted only on the alignment split;
- matching-item retrieval on 410 held-out rows.

This asks whether student states preserve item identity and teacher-like geometry after a split-fit translation. It is deliberately separate from whether they encode the correct answer.

## 2. Locked design and lineage

| Item | Locked value |
|---|---|
| Student | `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| Teacher | `Qwen/Qwen2.5-14B-Instruct` revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Student dtype | bfloat16 |
| DEV panel | 1,024 rows, SHA-256 `c0e15a890b598544059ac337cc475123f97c05e3c1626febcdee1c6d8fe02615` |
| KP-1R population | 329 rows; 304 primary after excluding MBPP |
| Probe split | 212 train, 92 evaluation |
| Primary surfaces | substrate layer 24; seed-0 loop-4 recurrent cell set |
| Resampling | 10,000 bootstrap draws; 10,000 matched permutations |
| Fingerprint split | 614 fit, 410 evaluation; seed 20260819 |
| Teacher taps | layers 12, 24, 36, 48 |
| Student schema | frozen 44-cell schema from T1 |
| Training | none; optimizer never constructed |

Target-entropy auditing passed before scoring. GSM8K had nine unique first answer-bearing tokens and a 35.25% dominant share. MBPP remained highly concentrated (88% dominant token), which is one reason it stayed outside the primary estimand.

## 3. Execution and gates

The work ran in two bounded jobs:

1. **CPU cached rung.** Reused the registered cached state surfaces, repaired targets, retained row predictions, and computed controls and confidence intervals.
2. **A100 score-only rung.** Loaded the pinned 0.5B student and 14B teacher, extracted teacher-forced student readouts and teacher fingerprints, and wrote row-level predictions, geometry comparisons, transport results, and state caches. It performed no parameter updates.

All pre-model contracts passed. The run status recorded `complete`, `confirm_scored=false`, `eval_e_scored=false`, and `optimizer_steps=0`. The A100 was shut down after the durable artifacts were verified.

## 4. KP-1R results

### 4.1 Cached rung

| Surface | Probe accuracy | Frequency control | Pooled margin | 95% interval | Verdict |
|---|---:|---:|---:|---:|---|
| Cached projected layer-24 proxy | 19.57% | 30.43% | -10.87 points | -22.83 to +1.09 | gate failed |
| Seed-0 loop-4 recurrent set | 20.65% | 30.43% | -9.78 points | -20.65 to +1.09 | gate failed |

The cached substrate surface is explicitly a projected proxy, not the raw substrate state. It was useful as a cheap screen but cannot replace the strong rung.

### 4.2 Strong teacher-forced rung

| Primary surface | Probe row-mean token accuracy | Frequency control | Pooled margin (95% interval) | Macro margin (95% interval) | Pooled permutation p | Gate |
|---|---:|---:|---:|---:|---:|---|
| Substrate layer 24 | 26.90% | 19.75% | +7.16 (-3.08, +17.39) | +5.28 (-6.29, +16.65) | 0.172 | fail |
| Seed-0 loop-4 recurrent set | 16.94% | 19.75% | -2.81 (-12.77, +7.07) | +6.91 (-3.96, +17.57) | 0.925 | fail |

The layer-24 point estimate is directionally interesting but unresolved. Neither its pooled nor macro interval excludes zero. The loop-4 result is weaker and changes sign between pooled and macro summaries, which reinforces the need to keep both accountings visible.

All secondary surfaces were Benjamini-Hochberg corrected. None passed. The best unresolved secondary hint was also substrate-side, not recurrent-loop-side.

### 4.3 MBPP exploratory read

The recovered MBPP sequence read covered 25 DEV rows. Mean native-base teacher-forced log probability was `-2.0229`; one row hit the registered 128-token truncation. This is descriptive only and is not part of the knowledge-presence gate.

### 4.4 Important log-probability limitation

The strong probe's classification logits are cosine scores without a separately calibrated temperature. Their softmax log probabilities are therefore not on the same identified scale as the native LM-head log probabilities. Probe-versus-native log-probability differences must not be interpreted scientifically from this run. The registered accuracy-margin decision remains valid because it depends on ranking, not logit calibration. A confirmatory sequence-log-probability claim would require a predeclared calibration procedure or an LM-head-compatible probe.

## 5. Teacher-fingerprint results

### 5.1 Split-fit transport to teacher layer 12

| Student surface | Seed | Top-1 | Top-10 | Median rank | Relative error |
|---|---:|---:|---:|---:|---:|
| Layer 6 | 0 | 40.73% | 85.61% | 2 | 0.774 |
| Layer 6 | 1 | 42.20% | 88.54% | 2 | 0.780 |
| Prelude pool | 0 | 29.27% | 70.00% | 4 | 0.999 |
| Prelude pool | 1 | 28.78% | 74.88% | 4 | 0.999 |
| Layer 24 | 0 | 19.76% | 60.00% | 7 | 0.683 |
| Layer 24 | 1 | 21.95% | 60.98% | 6 | 0.680 |
| Loop-4 pool | 0 | 11.95% | 47.56% | 11 | 0.987 |
| Loop-4 pool | 1 | 13.90% | 44.88% | 14 | 0.976 |
| Chance | - | 0.244% | 2.439% | - | - |

The best operational fingerprint is the early substrate state, especially layer 6 translated to teacher layer 12. The recurrent loop pool remains far above chance but is materially weaker than the base-model substrate and prelude surfaces. This argues for retrieving by stable input/item geometry before asking the recurrent pathway to transform or consume the retrieved content.

### 5.2 Basis-invariant geometry

The maximum cell-level linear CKA was `0.9927` for seed 1, student layer 6 to teacher layer 12. Across the main substrate layers, CKA was generally about `0.94-0.99`; prelude groups were about `0.46-0.56`; recurrent loop groups were about `0.18-0.27`.

Because benchmark clusters can inflate absolute CKA, a post-hoc sanity check permuted teacher rows within battery 200 times. All four tested matched pairs exceeded every permuted draw (`p=1/201`):

| Student surface | Seed | Matched CKA | Null mean | Null maximum |
|---|---:|---:|---:|---:|
| Layer 24 | 0 | 0.98610 | 0.98543 | 0.98596 |
| Layer 24 | 1 | 0.99049 | 0.98858 | 0.98914 |
| Loop-4 pool | 0 | 0.24597 | 0.22907 | 0.23068 |
| Loop-4 pool | 1 | 0.22818 | 0.20752 | 0.20995 |

This check is descriptive and post hoc. It establishes that the matched-row increment is not explained solely by battery labels in these draws. It does not convert CKA into an answer-content result. The split-fit retrieval rates provide the clearer practical evidence.

## 6. Interpretation

### Finding A: no registered positive answer-readout result

The correct verdict is **NO POSITIVE KNOWLEDGE-PRESENCE GATE**. Do not write "knowledge absent." A linear probe can fail because information is absent, distributed nonlinearly, represented at another position, or too weak for this sample. The layer-24 estimate leaves substantial uncertainty in both directions.

### Finding B: strong, replicated shared item geometry

The 0.5B and 14B states are not coordinate-aligned, but their item geometry is sufficiently shared that a map fitted on 614 rows retrieves held-out matching items at roughly 40-42% top 1 from layer 6. This is a large increase over 0.244% chance and replicates across both P3.5 seeds.

### Finding C: recurrence transforms away from the strongest retrieval surface

Teacher resemblance and transport retrieval are strongest before or early in the substrate. Recurrent-loop states remain structured and above chance but lose much of the direct item-fingerprint signal. That is consistent with the prior T1 observation that loops move through an ordered internal trajectory without necessarily changing the task answer.

### Combined implication for Stage 2A

Lead with a **teacher-fingerprint lookup/content path**, using early substrate states as keys and teacher layer 12 as the initial target space. Keep an expert-access arm available, but do not justify it by saying the current model already contains linearly readable answers. If expert access is tested, treat it as an intervention that may supply missing content rather than merely expose content proved to be latent.

## 7. Limitations and deviations

1. Only 92 rows were in the held-out KP-1R probe evaluation. The layer-24 interval is wide enough that a moderate positive effect remains possible.
2. The primary test is linear. A negative result does not exclude nonlinear or sequence-position-specific information.
3. Tier-1 has only three population rows and should not drive the macro interpretation by itself.
4. MBPP is exploratory and includes one truncated row.
5. Probe softmax log probabilities are uncalibrated and not comparable to native LM-head log probabilities.
6. The original strong-run `summary.json` and pre-model audit did not flush through the Drive FUSE mount before teardown. All row-level and geometry artifacts were durable. The publication summary was reconstructed deterministically; the reconstructed pre-model audit matches the original expected SHA exactly. This is a transport-layer recovery, not a scientific rerun.
7. The original implementation's permutation statistic operated at token level. The final publication receipt recomputes it at the registered row estimator, stratified by battery and token position. This stricter matched-estimator result is the number of record and is disclosed as post-run estimator hygiene.
8. The within-battery CKA null was added after inspection as a sanity check. It does not alter registered metrics or gates.
9. Strategy authorized T3a independently, but the repository and Drive authority available in this wave did not contain a sufficiently exact T3a implementation contract. No T3a code or result was invented.

## 8. Receipt and artifact map

### Drive publication set

- Handoff: Drive `1X85ANBbuM5xsdGY1iBzvALokIAaqPpFX`
- Figure: Drive `1LGdD74K0Q66aJHsDd26V8I78YVstIyPf`
- Recovery receipt: Drive `1d7lO5reTTGTFrTF47ennX4ddy9M3RSTt`
- Recovered machine summary: Drive `1mF2OeaF_TIVnaiWWrxSWocqytvpk5jV7`
- Exact recovered pre-model audit: Drive `1ysvSRK80Z0EEtx6oh_-tG02gomXoieQo`
- CKA null receipt: Drive `1KoPHFK34fcLET5BG7NVQ7EAOuvJcjOfJ`
- MBPP recovery summary: Drive `1kKDD8ei1RZGSNXNx0gN9gWEgBcY10HCM`

These raw files are in research folder `1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr`. The original Colab artifact folder remains the source for the large private tensors and row tables. The recovered publication files are mirrored in the research folder because the local read-only artifact credential could not create replacement children after runtime teardown.

### Canonical local artifacts

- Recovered machine summary: `artifacts/kp1r_teacher_20260816/analysis/recovered_summary.json`, SHA-256 `84177a9d561d6ed0384413a8898d5b3b06070a87a94d3865e41745c10e71cf69`
- Publication recovery receipt: `artifacts/kp1r_teacher_20260816/analysis/publication_recovery_receipt.json`
- Exact recovered pre-model audit: `artifacts/kp1r_teacher_20260816/analysis/pre_model_target_audit_recovered.json`, SHA-256 `edc750dccc768932566f075a7c34d058f9e45ae2e15dadae4f715a3fdf719e9a`
- KP-1R row predictions: `artifacts/kp1r_teacher_20260816/strong/private/kp1r_teacher_forced_row_predictions.jsonl`, SHA-256 `b05d8f966cbf9f9576b9d0583ad93e25b88e3f76268530c71cb11a75e5c6fed0`
- Fingerprint comparisons: `artifacts/kp1r_teacher_20260816/strong/private/teacher_fingerprint_comparisons.jsonl`, SHA-256 `3fabfd731d2374a7ccddc8ca866ff7c23385560e46f751d52b5f5b4a1cc58e21`
- Fingerprint transport: `artifacts/kp1r_teacher_20260816/strong/private/teacher_fingerprint_transport.jsonl`, SHA-256 `b6bf6adceec59167ec002e6aad614ac776152a3acc2c712d43d200a3439fcb2f`
- Teacher states: `artifacts/kp1r_teacher_20260816/strong/private/teacher_fingerprint_states.pt`, SHA-256 `1c5add3ec8b781744804342eb18f1d9e9747b230775bd29b8d40226adcddaf33`
- CKA null receipt: `artifacts/kp1r_teacher_20260816/analysis/teacher_fingerprint_null.json`, SHA-256 `8f6e0dd87ce62e7d24cb5635cee53d96822a737192831ede9eb36be388970987`
- MBPP recovery rows: `artifacts/kp1r_teacher_20260816/analysis/mbpp_recovery/mbpp_teacher_forced_sequence_log_likelihood.jsonl`, SHA-256 `bed39f61ed591eee961987c4f074b913910bfea0c5ee2b5510d92e97801a565b`
- Figure: `artifacts/kp1r_teacher_20260816/analysis/paper2_kp1r_t1_teacher_wave_20260816.svg`, SHA-256 `c11d463171ed60bf0903341a5c96c8dd56c3c4a3a769ed5998e387c38407645a`

### Durable Drive run folders

- Cached rung: `recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_kp1r_cached_20260816`
- Strong rung: `recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_kp1r_t1_teacher_20260816`

## 9. Decisions requested from strategy

1. Ratify `NO POSITIVE KNOWLEDGE-PRESENCE GATE` as the KP-1R verdict, with the explicit prohibition on translating it to "knowledge absent."
2. Ratify the teacher-fingerprint spine as positive and use early substrate layer 6 to teacher layer 12 as the initial retrieval pairing.
3. Decide whether Stage 2A leads with lookup/content augmentation and keeps expert access as a secondary intervention, as the present evidence recommends.
4. Decide whether the unresolved layer-24 `+7.16` point estimate merits a larger DEV-only linear/nonlinear probe later, or whether the present uncertainty is sufficient to proceed architecturally.
5. If sequence log probability is needed as a claim, authorize a separate calibrated reader contract. Do not reuse the uncalibrated cosine-softmax values.
6. Supply or identify the exact T3a intervention specification before implementation. Authorization alone is not sufficient to infer its data, intervention, controls, and decision rules.

## 10. Recommended next steps

1. Bank this wave and update the tracker with the two separate findings: no positive answer-readout gate; positive teacher fingerprints.
2. Draft Stage 2A around early-state fingerprint retrieval, with teacher layer 12 as the first target and split-fit retrieval as the baseline contract.
3. Preserve retrieval and answer-content metrics as separate gates. High item retrieval must never be presented as answer recovery.
4. Run T3a only after its exact specification is located or issued.
5. Do not spend CONFIRM or EVAL-E on this diagnostic branch.

## 11. Plain-language summary

We asked two questions. First: does the small model already know the missing answer, with its output head simply failing to read it? The repaired test did not establish that. One internal layer showed a hopeful but noisy advantage, while the recurrent state did not. The evidence is too uncertain to say the answer is present, and a failed linear probe cannot prove it is absent.

Second: do the small model and the 14B teacher organize the same questions in related internal patterns? Yes, strongly. After learning a translation on one set of questions, an early small-model state identified the matching teacher state on unseen questions about 42% of the time, compared with about 0.24% by chance. That result repeated in both trained seeds.

The practical consequence is straightforward: the next architecture should treat the small model's early state as a good address for retrieving teacher-derived content. It should not assume that the content is already inside the small model waiting for a better reader. The map is there; the missing information may not be.
