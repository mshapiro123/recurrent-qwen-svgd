# Handoff: Stage 2A T3 Memory Screen

**Date:** 2026-08-18  
**Program:** Paper Two, Stage 2A  
**Status:** complete, analyzed, packaged, and compute released  
**Registered verdict:** `SCREEN_BELOW_PROCEED_THRESHOLD`  
**Data boundary:** reused 1,024-row DEV panel only; CONFIRM and EVAL-E remain sealed

## 0. Bottom line

The teacher-fingerprint memory arm produced a small positive effect in both seeds, but it did not clear its locked screen threshold. T3a-C finished at `+3/1,024` and `+8/1,024` correct rows relative to the frozen `502/1,024` base. The mean was `+5.5` rows, below the preregistered `+8` mean required to open T3-full, although both seeds were positive. The scripted verdict is therefore **`SCREEN_BELOW_PROCEED_THRESHOLD`**. No main campaign or sealed evaluation is authorized by this result.

The literal n-gram arm was descriptively stronger, at `+4` and `+10` rows, mean `+7`. It did not have authority to replace the T3a gate, and its one- to two-row advantage over T3a was not statistically resolved in either seed. The correct scientific statement is that teacher-fingerprint memory did not outperform a matched literal surface-memory baseline in this screen.

Both registered controls were flat in the locked pooled sense: shuffled teacher values ended at `-2` rows and frozen random values at `-1`, inside the `[-3,+3]` equivalence band. Neither exceeded the `+3` seed-1 escalation trigger. This supports a bounded content-specificity reading because trainable gates and injection alone did not reproduce the positive pooled endpoints. The matched seed-0 contrasts favored T3a over shuffled by five rows and over random by four; they favored T3b by six and five rows. These contrasts are directional, not statistically resolved.

The battery decomposition prevents a stronger claim. Both controls gained `+5/+6` rows on MBPP while losing `-6` on GSM8K and `-2` on Tier-1. Their pooled equivalence is therefore cancellation, not behavioral inertness. Content-bearing memories improved that balance, mainly through MMLU and reduced GSM8K harm, but MBPP's apparent gain was largely generic to the trained injection recipe. Any successor should use battery-stratified safety and utility telemetry rather than relying on the pooled net alone.

## 1. Question and rationale

Stage 2A followed the P3.5 Branch-C return to effect-size work. The preceding diagnostics established two facts:

1. Student layer-6 states provide a strong address for corresponding teacher states after split-fit alignment.
2. The available KP-1R probe did not establish that the missing answer content was already decodable from the student.

The screen therefore tested whether non-DEV teacher content could be stored behind the measured address system and delivered through the already validated scratchpad-to-flow-to-bridge path. It compared that concept-keyed memory against a literal n-gram memory of nearly identical trainable size, plus two controls that retained trainable gate and injection machinery while destroying or replacing value content.

This was a screen, not a confirmation experiment. Its purpose was to decide whether teacher-fingerprint memory had enough task effect to justify T3-full. The answer under the locked rule is no.

## 2. Locked design

### 2.1 Frozen lineage and initialization

- Student substrate: Qwen2.5-0.5B lineage, frozen.
- Seed-0 initialization: P3.5 Arm-S EMA step 4,400, SHA `a047e2e7b35320376a736492c79d913b8690937da785efa2af002c8f54d26ca6`.
- Seed-1 initialization: P3.5 Arm-S EMA step 4,400, SHA `e36cddb76407c8f853ccb43824c77cf01d15f144780726dd9aec23215467fccb`.
- Frozen sidecar digests remained seed-specific and unchanged across every arm initialized from that seed.
- Inference used the existing post-initializer scratch write, then the frozen flow and AnchoredBridge path. There was no new substrate writer.

### 2.2 Memory population

- Non-DEV reference rows only; DEV overlap required to be zero.
- Admission required teacher correctness and 14B/32B family concurrence.
- `5,039` verified training rows remained after the held-out non-DEV validation exclusion.
- The automatic sizing rule selected `4,096` memory slots.
- Geometry used student layer 6 for keys and teacher layer 12 for values, each in a frozen PCA-128 basis fitted only on non-DEV rows.
- T3a training used leave-one-out retrieval so a row could not retrieve its own owned slot.

### 2.3 Arms

| Arm | Seeds | Trainable parameters | Role |
|---|---:|---:|---|
| T3a-C concept memory | 0, 1 | 540,685 | student layer-6 fingerprint keys, trainable teacher-layer-12 values |
| T3b literal n-gram | 0, 1 | 540,681 | matched 2- and 3-gram hash memory |
| Shuffled values | 0 | 16,397 | retrieval-specificity control; values permuted then frozen |
| Random values | 0 | 16,397 | capacity/perturbation control; values random then frozen |

The controls retained trainable compatibility, gate, and injection maps. Their lower trainable counts are intentional because their values were frozen. A control seed 1 was permitted only if seed 0 exceeded `+3` rows. Neither did.

### 2.4 Objective and dose

- `L = 0.5 * L_CE + 0.5 * L_KL`.
- Forward teacher-to-student KL at temperature 1 over the cached top-128 14B lattice.
- CE on the teacher token, which is gold by admission.
- KP-1R-repaired answer-bearing positions only; prompt, formatting-only tokens, position zero, and invalid MBPP spans excluded.
- AdamW, learning rate `5e-4`, batch 128, 50 warmup steps, cosine landing over steps 1,081-1,200.
- EMA decay `0.999`; EMA step 1,200 was the primary endpoint.
- Training amplitude sampled from `[0.02,0.11]`; registered score read fixed at `0.05`.
- BF16 model execution on one NVIDIA A100-SXM4-40GB.

## 3. Locked readings and integrity

T3a opened T3-full only if its two-seed mean paired gain was at least `+8` rows and each seed was strictly positive. Controls were flat if each executed control remained within `[-3,+3]`. The prewritten verdicts were:

- `T3_FULL_PROCEEDS`: T3a passes and controls are flat.
- `SCREEN_POSITIVE_CONTROL_AMBIGUOUS`: T3a passes but a control is not flat.
- `SCREEN_BELOW_PROCEED_THRESHOLD`: T3a misses its gate.

All four training arms completed the registered 1,200 steps. Six checkpoints per run landed at steps 200 through 1,200. All final endpoints were read on the same frozen 1,024-row DEV panel. The aggregate and every arm summary state `confirm_scored=false` and `eval_e_scored=false`. No control escalation occurred. No non-finite loss, non-finite gradient, frozen-lineage change, identity failure, memory/DEV overlap, or sealed-partition contact was recorded.

## 4. Registered endpoint results

| Condition | Correct | Net vs 502 | Fixes | Regressions | Exact paired p |
|---|---:|---:|---:|---:|---:|
| T3a concept, seed 0 | 505/1,024 | **+3** | 49 | 46 | 0.8376 |
| T3a concept, seed 1 | 510/1,024 | **+8** | 54 | 46 | 0.4841 |
| T3b n-gram, seed 0 | 506/1,024 | **+4** | 47 | 43 | 0.7520 |
| T3b n-gram, seed 1 | 512/1,024 | **+10** | 54 | 44 | 0.3634 |
| Shuffled values, seed 0 | 500/1,024 | **-2** | 44 | 46 | 0.9161 |
| Random values, seed 0 | 501/1,024 | **-1** | 46 | 47 | 1.0000 |

T3a's mean was `+5.5/1,024`, or `+0.537` percentage point. Both seeds were positive, but the mean missed the `+8` threshold by 2.5 rows. T3b's descriptive mean was `+7/1,024`, or `+0.684` point. No individual arm-versus-base paired sign test was significant at 0.05.

The screen is not a near-threshold discretionary pass. Its registered gate was on the two-seed T3a mean, and that mean was 31.25% below the required eight-row gain. T3a seed 1 touching `+8` does not satisfy a mean gate.

## 5. Battery decomposition

| Condition | ARC-C | ARC-E | GSM8K | MBPP | MMLU | Tier-1 |
|---|---:|---:|---:|---:|---:|---:|
| T3a concept, seed 0 | 0 | +1 | -4 | +6 | +2 | -2 |
| T3a concept, seed 1 | -1 | +2 | -3 | +7 | +5 | -2 |
| T3b n-gram, seed 0 | 0 | +1 | -4 | +6 | +3 | -2 |
| T3b n-gram, seed 1 | 0 | +2 | 0 | +6 | +4 | -2 |
| Shuffled values, seed 0 | 0 | 0 | -6 | +5 | +1 | -2 |
| Random values, seed 0 | 0 | 0 | -6 | +6 | +1 | -2 |

Four patterns matter:

1. **MBPP is positive in every arm.** Concept memory scored `+6/+7`, literal n-gram `+6/+6`, and controls `+5/+6`. The code gain is therefore not evidence that teacher content was retrieved. Much of it belongs to the shared trained injection path or its optimization context.
2. **GSM8K is the main harm pocket.** Concept memory reduced the control loss from `-6` to `-4/-3`; T3b seed 1 reached zero. The system is not generally safe across workloads merely because its pooled net is positive.
3. **MMLU carries the clearest content-linked separation.** T3a gained `+2/+5` and T3b `+3/+4`, versus `+1` in both controls. This is still exploratory and row-small.
4. **Tier-1 loses two rows in every condition.** The invariant loss across content and control arms points to the common gate/injection recipe rather than memory identity. It should become an explicit stratum-level constraint in any successor.

The control equivalence result is valid under its registered pooled estimator, but it must not be paraphrased as no workload-specific effect.

## 6. Matched arm comparisons

### 6.1 T3a versus T3b

| Seed | T3a only correct | T3b only correct | T3a minus T3b | Exact p |
|---|---:|---:|---:|---:|
| 0 | 5 | 6 | -1 | 1.0000 |
| 1 | 4 | 6 | -2 | 0.7539 |

The arms changed 40 and 38 predictions, respectively, but only 11 and 10 of those changes crossed correctness in opposite directions. The literal memory's small endpoint edge is not resolved; equivalently, the concept-keyed memory has no measured advantage over a literal n-gram baseline.

### 6.2 Content arms versus controls, seed 0

| Contrast | Left only correct | Control only correct | Net rows | Exact p |
|---|---:|---:|---:|---:|
| T3a vs shuffled | 9 | 4 | +5 | 0.2668 |
| T3a vs random | 6 | 2 | +4 | 0.2891 |
| T3b vs shuffled | 8 | 2 | +6 | 0.1094 |
| T3b vs random | 6 | 1 | +5 | 0.1250 |

Every contrast points in the content-bearing direction, but none resolves at 0.05. The appropriate reading is **bounded evidence that useful value structure matters**, not proof that the teacher-fingerprint values or the literal table are causally superior.

## 7. Replication and row stability

The seed-level net differences do not come from unrelated row churn.

| Arm | Fix intersection / union | Fix Jaccard | Regression intersection / union | Regression Jaccard |
|---|---:|---:|---:|---:|
| T3a concept | 48/55 | 0.873 | 42/50 | 0.840 |
| T3b n-gram | 46/55 | 0.836 | 41/46 | 0.891 |

T3a predictions differed across seeds on 64 rows and T3b on 61, but most correctness-changing rows were shared. The small positive effects are systematic on this reused DEV panel even though their net magnitude is below the registered threshold. This strengthens the claim that the screen measured a real, repeatable behavior while leaving the generalization and confirmation questions open.

## 8. Mechanism telemetry

All arms produced nonzero deployed writes. Mean realized writeback ratios were approximately `0.00107-0.00118`, with maxima near `0.00215`, well below the operating ceiling. T3a queried roughly 2,482 distinct slots per seed on DEV; shuffled and random controls queried 2,481 and 2,482. T3b uses its literal hash path, so the concept-memory slot-count telemetry is structurally inapplicable and correctly reports zero.

Concept-memory retrieval score and entropy were essentially identical across seeds, and the trained compatibility gate mean was `0.585` and `0.563`. Control gates were somewhat higher (`0.592` shuffled, `0.618` random), yet pooled accuracy was lower. This is directionally consistent with content quality, not gate openness alone, explaining the positive arm differences. It is not enough to isolate value content because each arm trained its own gate and injection map.

## 9. Interpretation

### 9.1 What is supported

1. A teacher-fingerprint memory can be trained end to end through the frozen scratchpad/flow/bridge path without violating lineage or sealed-data boundaries.
2. T3a produced positive DEV net rows in both seeds and highly overlapping fix/regression populations.
3. Literal n-gram memory produced a similarly sized or slightly larger two-seed effect under a nearly identical parameter budget.
4. Shuffled and random value controls were flat under the registered pooled equivalence rule.
5. Content-bearing arms improved the pooled balance relative to controls, especially on MMLU and GSM8K.

### 9.2 What is not supported

1. T3a did not pass its screen and does not authorize T3-full.
2. No capability gain is confirmed; this is reused DEV.
3. Teacher-fingerprint memory did not beat literal n-gram memory.
4. The result does not establish that teacher content is generally useful, that student knowledge is absent, or that fingerprint retrieval recovers answers.
5. Pooled control equivalence does not imply battery-level harmlessness.
6. MBPP's gain cannot be attributed to memory content because both controls also gained strongly.
7. No condition passed an individual 0.05 paired test, and the two seeds must not be pooled as 2,048 independent rows.

### 9.3 Program consequence

The fingerprint geometry remains a valid addressing result, but this screen does not support promoting it as the primary memory architecture. The most defensible next design treats T3b as a serious baseline and separates three effects that the current arm matrix partly confounds:

- generic benefit or harm from training the injection path;
- immediate causal value of the retrieved content under a fixed trained gate/map;
- workload-specific routing, especially MBPP benefit versus GSM8K/Tier-1 harm.

## 10. Recommended strategy decisions

### D1. Do not open T3-full from this screen

Recommended. Bank the registered miss and preserve all endpoints. A larger teacher-memory campaign is not justified by a `+5.5` mean against an `+8` screen, particularly when T3b reached `+7`.

### D2. Authorize a no-training crossed-value causal audit before choosing the next memory architecture

Recommended. On fixed seed-0 T3a and T3b checkpoints, evaluate the identical DEV rows while swapping only the memory values among correct, shuffled, and random assignments, keeping the trained gate and injection map fixed. This removes the current between-arm co-adaptation confound. The audit must be preregistered as diagnostic DEV reuse and cannot alter this screen's verdict.

### D3. Make workload-specific gating a prerequisite of any successor campaign

Recommended. The shared recipe consistently traded MBPP gains against GSM8K and Tier-1 losses. Any next campaign should report and constrain target-primary batteries separately, with Tier-1 and GSM8K floors that cannot be hidden by MBPP gains.

### D4. Decide whether literal memory becomes the Stage 2A baseline

Recommended default: yes. T3b is cheaper conceptually, matched T3a on every major positive battery, and was descriptively stronger overall. Teacher-fingerprint memory should have to beat T3b under a fixed-map content audit or a redesigned admission/value target before regaining lead status.

### D5. Keep the teacher-fingerprint line available as a targeted MMLU/GSM8K mechanism probe

Recommended. The teacher line may still be the better route for nonliteral transfer, but this screen does not show it. A targeted probe should test retrieval relevance and value usefulness on the rows where T3a improves over controls, rather than repeating a larger pooled training run.

## 11. Limitations

- All scores use a repeatedly inspected 1,024-row DEV panel.
- Only two seeds were run for T3a and T3b; controls used seed 0 unless the predeclared escalation fired, which it did not.
- The control equivalence interval was defined on pooled rows, not each battery.
- Each arm trained its own gate and injection map, so endpoint arm differences do not isolate value content perfectly.
- T3b slot telemetry is structurally different from concept-memory slot telemetry.
- The small row deltas have wide uncertainty; none of the paired arm-versus-base reads is individually significant.
- No CONFIRM, EVAL-E, or external battery was scored.

## 12. Plain-language summary

We tested two ways to give the small recurrent model an external memory. The first used an internal fingerprint from the question to retrieve teacher-derived content. The second used a simpler table indexed by literal two- and three-token patterns. Both helped a little on the development set. The teacher-fingerprint version added three correct answers in one run and eight in the other. The literal version added four and ten.

Our rule required the teacher-fingerprint version to average at least eight extra correct answers before we spent more on it. It averaged five and a half, so it did not pass. The simpler literal memory averaged seven and was not worse than the teacher memory. That is important: the sophisticated addressing geometry is real, but this experiment did not show that it delivers more useful task content than a simple surface-form table.

The two controls ended near zero overall, which is reassuring, but their zero was a cancellation. They improved the code problems and hurt the math problems. The content-bearing memories reduced some of that math harm and added a few multiple-choice wins. The next design should therefore focus less on a single pooled score and more on deciding when a memory write is useful for each workload.

No sealed test was opened, no result was promoted, and the GPU was shut down after all receipts were copied and verified.

## 13. Canonical artifacts

- Drive run root: `MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_stage2a_t3_screen_20260817`.
- Aggregate receipt: `receipts/summary.json`, SHA-256 `3e4eaa7ced4489b8253a016dcae7cd0e04eea56778481de06a47250a6fd1f7bb`.
- Local analysis: `outputs/stage5/stage5_paper2_stage2a_t3_screen_20260817/analysis/analysis_summary.json`, SHA-256 `8169451f48f1d1f9c5f857ef0fcc1158a5a5d8d318d3299fdab45fb7ad950723`.
- Figure: `docs/figures/paper2_stage2a_t3_screen_20260818.png`, SHA-256 `397c31a5ab081bffeb3e07c0bdbd8a2f771e78684cb6609f1863eb5fadec4a7f`.
- Vector figure: `docs/figures/paper2_stage2a_t3_screen_20260818.svg`, SHA-256 `4931a3b5172aa0788c65b1e1458700d15b6df38fd2a5726f06d79060f8913888`.
- Final 63-file receipt archive: `artifacts/stage2a_t3_20260818/stage2a_t3_final_receipts.tar.gz`, 6,557,801 bytes, SHA-256 `e30a294b827ab01d5d333d0c78a0dfca39bfbcbb57156608856063bd2d60ab92`.
- Archive manifest: `artifacts/stage2a_t3_20260818/stage2a_t3_final_receipts_manifest.json`, SHA-256 `39e22bfefd7b5ef7cb2cc7af31be78d912ba7fe0380e2a7c4fb0df383eef1dca`.
- Executed lock: `training/paper2_stage2a_preregistration.json`, registered source hash `58725a8fd0baca83a3dd27326afe310d21202d813356ea5fef42b7a57f8fffd7`.

| Drive artifact | File ID |
|---|---|
| This handoff | `1FdmtjME_rBpPMeF9u7MAHHN69TiCc-8K` |
| Analysis summary | `13swNSlTHqUgOBNf8alJM6hwioSpy1B6f` |
| Figure PNG / SVG | `1jt0mWIXpN88i_QLsku9NBIf92Bk9d04i` / `1rNz44Fq35XA5EMOMNj1HFZ6XZB75P1nv` |
| Receipt archive | `1X9K8LSjUn_0k5NKXKJN9gdIVRCOdoK5j` |
| Receipt manifest | `1m8eiKnzgFEwYY8dbV4adysEA1kGsZHnh` |

## 14. Closeout state

- Six planned arm/seed runs completed: T3a seeds 0/1, T3b seeds 0/1, shuffled seed 0, random seed 0.
- No control seed-1 escalation was eligible.
- All 63 receipt files are locally packaged and hash-verified.
- CONFIRM and EVAL-E remain sealed.
- Colab reports no active sessions after explicit teardown.
