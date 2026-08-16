# Handoff: P3.5 Stabilized Landing and Probe-Reader Comparison

**Date:** 2026-08-15  
**Program:** Paper Two, Phase 3, P3.5  
**Status:** complete, remotely verified, packaged, and analyzed  
**Data boundary:** DEV only; CONFIRM and EVAL-E remain sealed

## 0. Bottom line

The stabilized landing protocol did not preserve or increase the P3.4 endpoint effect. Under the registered primary read, EMA at step 4,400 with the gate ceiling pinned to `0.02`, Arm S finished at `+4/1,024` rows in seed 0 and `+6/1,024` in seed 1. The mean `+5.0/1,024` falls below the preregistered Branch-C boundary of `+8`, so the locked reading is **Branch C: effect-size work returns to Stage 2A**. Neither seed met its inherited endpoint benchmark (`+8` and `+9`).

The landing protocol did reduce late-window row churn, but the continuous task margins and causal capture remained essentially flat. EMA helped relative to the raw endpoint in both seeds, but it did not recover the late P3.4 mean. A selection-barred endpoint matrix also shows strong seed dependence in the gate-ceiling response: seed 0 rose from `+4` at `0.02` to `+15` at `0.08`, while seed 1 stayed at `+6`. This makes write amplitude a measured next variable, but it cannot alter the registered Branch-C verdict.

Arm R's probe reader did not earn promotion. Its registered primary finished at `+2/1,024`, two rows below the matched Arm S seed-0 endpoint. The paired row comparison contained two Arm-S-only correct rows and no Arm-R-only correct rows (two-sided exact `p=0.5`). At Arm S's `92.676%` recall, Arm R's precision was `61.454%`, only `0.129` percentage point above Arm S's fixed-threshold precision. `pi_dep` was identical and the row-minimum margin was lower by only `0.000031`. This is no material reader advantage and includes a small task loss.

## 1. Locked questions and design

P3.5 asked two bounded questions on the completed P3.4 mechanism:

1. Can a terminal learning-rate decay plus EMA convert the late-training oscillation into a stable endpoint?
2. Does a detached four-probe reader improve selection relative to the current mean-pooled reader?

The registered matrix was:

- **Arm S:** stabilized landing, seeds 0 and 1.
- **Arm R:** the same seed-0 landing with only the control/gate reader changed from mean pooling to detached four-probe pooling.
- Steps `4,001` through `4,400`, with registered looks at `4,100`, `4,200`, `4,300`, and `4,400`.
- EMA primary, raw secondary.
- Gate ceiling `0.02` primary, `0.08` secondary and selection-barred.
- Repaired exact-reader audit cache v2 as the sole causal instrument.
- CONFIRM and EVAL-E sealed.

The Arm-S branch rule was locked before training:

- mean endpoint at least `+10`: Branch A, draft P3.6;
- mean `+8` to below `+10`: Branch B, margin tie-breaker and lever queue;
- mean below `+8`: Branch C, return effect-size work to Stage 2A.

Arm R had no arbitrary composite score. Its task net, matched-recall gate precision, `pi_dep`, and row-minimum margin are reported separately.

## 2. Arm S primary results

| Registered read | Seed 0 | Seed 1 | Joint reading |
|---|---:|---:|---:|
| Fixed base | 502/1,024 | 502/1,024 | unchanged panel reference |
| EMA step 4,400 at ceiling 0.02 | 506/1,024 | 508/1,024 | mean 507/1,024 |
| Net correct rows | **+4** | **+6** | **mean +5.0** |
| Fixes / regressions | 49 / 45 | 50 / 44 | positive, small net |
| Inherited endpoint benchmark | +8 | +9 | neither met |
| Registered branch |  |  | **C: below +8 mean** |

Registered look trajectories were:

- Seed 0: `+3`, `+2`, `+2`, `+4`.
- Seed 1: `+7`, `+10`, `+9`, `+6`.

Adjacent changed-row counts under the primary EMA read were `5/8/6` in seed 0 and `5/3/7` in seed 1. The landing therefore reduced the much larger late P3.4 wobble, but stabilization centered the endpoint around a smaller effect rather than preserving the prior `+8/+9` benchmarks.

## 3. Endpoint sensitivity matrix

| Endpoint condition | Seed 0 net rows | Seed 1 net rows |
|---|---:|---:|
| Raw, ceiling 0.02 | +3 | +3 |
| EMA, ceiling 0.02 | **+4 primary** | **+6 primary** |
| Raw, ceiling 0.08 | +11 | +4 |
| EMA, ceiling 0.08 | +15 | +6 |

The seed-0 `0.08` read is a real secondary result, not a selectable endpoint. Its battery deltas were ARC-Challenge `+2`, ARC-Easy `+3`, GSM8K `+2`, MBPP `+7`, MMLU `+3`, and Tier-1 `-2`. Under the primary `0.02` read, seed 0 changed ARC-Challenge `+1`, ARC-Easy `+1`, GSM8K `-4`, MBPP `+6`, MMLU `+2`, and Tier-1 `-2`.

The cross-seed contrast matters. Raising the ceiling helped seed 0 materially but did not change seed 1's EMA net. The next amplitude study therefore needs replication and a locked dose curve; seed-0-only ceiling selection would be invalid.

## 4. Mechanism and safety

| Endpoint telemetry, primary | Seed 0 | Seed 1 |
|---|---:|---:|
| `pi_dir` | 15.761% | 15.548% |
| `pi_dep` | 28.302% | 27.586% |
| Collateral `chi` | 0 | 0 |
| Gate precision, fixed threshold | 61.325% | 47.185% |
| Gate recall, fixed threshold | 92.676% | 95.557% |
| Mean answer-token margin | 7.1507 | 7.1488 |
| Mean row-minimum margin | 1.0893 | 1.0884 |

The causal machinery remained active and selective. `pi_dir` stayed near 15-16% and deployed capture near 28%, but neither rose through the landing window. Continuous margins were also nearly flat. This supports a plateau reading rather than hidden continuous progress obscured only by discrete row churn.

Seed 1 satisfied every observe-only loss-share contract. Seed 0 breached the CE and KL share floors at all four registered looks. Those contracts were deliberately observe-only in this landing study, so the run is valid, but the seed-0 imbalance is a real diagnostic and must follow any next recipe.

No non-finite value, frozen-lineage violation, registered collateral event, or sealed-partition contact occurred. `confirm_scored=false` and `eval_e_scored=false` in every completed receipt.

## 5. Arm R reader comparison

| Step | Arm S seed 0 | Arm R seed 0 | R minus S |
|---|---:|---:|---:|
| 4,100 | +3 | +3 | 0 |
| 4,200 | +2 | +2 | 0 |
| 4,300 | +2 | +4 | +2 |
| 4,400 | **+4** | **+2** | **-2** |

At the registered endpoint, Arm S corrected 506 rows and Arm R corrected 504 against the same 502-row base. The row-paired comparison is unusually clean: Arm S alone was correct on two rows, Arm R alone on none, and the other 1,022 rows tied. The exact two-sided sign/McNemar probability is `0.5`; this is descriptive, not a resolved difference.

| Component | Arm S seed 0 | Arm R seed 0 | R minus S |
|---|---:|---:|---:|
| Task net rows | +4 | +2 | -2 |
| Gate precision at fixed 0.5 threshold | 61.325% | 61.374% | +0.050 point |
| Gate recall at fixed 0.5 threshold | 92.676% | 92.676% | 0 |
| Arm R precision at Arm S matched recall |  | 61.454% | +0.129 point versus S fixed-threshold precision |
| `pi_dep` | 28.302% | 28.302% | 0 |
| Mean row-minimum margin | 1.089273 | 1.089243 | -0.000031 |

The selection-barred endpoint matrix likewise does not reveal a stable reader benefit:

| Endpoint condition | Arm S seed 0 | Arm R seed 0 | R minus S |
|---|---:|---:|---:|
| Raw, ceiling 0.02 | +3 | +3 | 0 |
| EMA, ceiling 0.02 | +4 | +2 | -2 |
| Raw, ceiling 0.08 | +11 | +13 | +2 |
| EMA, ceiling 0.08 | +15 | +14 | -1 |

The probe reader therefore changed the gate scores slightly without producing a useful selection or task advantage. It is not promoted. The implementation remains available as an engineering primitive, but the scientific lever stays the mean-pooled reader.

## 6. Interruption and recovery receipt

Colab terminated the original A100 session at `2026-08-15T22:11:18Z` while Arm R's final EMA task evaluation was in progress. No stop command was issued by the coding task.

Durable state before termination:

- optimizer resume: step `4,380`, three complete registered looks;
- raw step-4,400 checkpoint SHA `2993884a2e7d9731c1cfa5fc1f00a4c8d8475a5420f2c73ddbbf0e3eac84cad8`;
- EMA step-4,400 checkpoint SHA `33207f0d0fb35cc2d2d069301015708eb15e50f81445710fa8492086bec2b6f1`;
- interrupted EMA task file: `852/1,024` rows;
- CONFIRM and EVAL-E untouched.

The interrupted state is retained under the remote `receipts/superseded` and `private/superseded` trees. Recovery used the unchanged registered runner on the same A100 accelerator class, replayed exactly steps 4,381 through 4,400, and regenerated the incomplete final read and score bundle.

The recovered checkpoint archives have different file hashes because PyTorch reserialized their containers. That byte-level difference is not treated as scientific identity. Direct tensor checks established exact identity for the raw trainable state and EMA state, with schedule hashes, objective weights, and generator state also identical. All four registered looks are present, `stop_reason=null`, the 16-condition score bundle is complete, and every remote integrity check passed. The recovery receipt records both original and recovered archive hashes rather than hiding the transport-level difference.

## 7. Sidecar v2 desk track

The authorized no-training desk build landed in parallel:

- detached deterministic `ProbePool`;
- `fast_wht` with dense-Hadamard equivalence, norm, and gradient tests;
- deterministic literal n-gram memory with causal access and zero substrate contact;
- gated sidecar injection with exact zero-gate identity and no input mutation;
- canonical expert projection, relevance-weighted expert distance, deterministic k-medoids, cluster routing aggregation, and ridge-plus-SVD low-rank initialization.

The T1 artifact preflight found that the P3.4 cached states promised for fingerprint construction are absent. No substitute was made. T1 therefore requires separate authorization for a fixed-row, no-training state-extraction pass. No T2/T3 training is authorized by this desk build.

## 8. Interpretation

### Supported

1. Terminal decay plus EMA makes the late endpoint less volatile at the row level.
2. Stabilization did not preserve the prior effect magnitude; the registered mean fell to `+5/1,024`.
3. Causal capture and continuous margins plateaued rather than improving through the landing.
4. Gate-ceiling response is seed-dependent. Write amplitude is worth a replicated measurement, not post-hoc selection.
5. The probe reader showed no material componentwise advantage and finished two task rows below its matched mean-reader control.

### Not supported

- No confirmed capability gain.
- No eligibility for P3.6 or the sealed exam.
- No claim that EMA or decay improved the underlying model rather than only the endpoint variance.
- No claim that ceiling `0.08` is better in general.
- No Arm R promotion; its complete componentwise endpoint read did not improve the matched task result.
- No task-level claim outside this reused DEV panel.

## 9. Recommended next decision

The registered branch sends effect-size work to Stage 2A. The strongest measured candidate is a replicated, preregistered amplitude curve, because seed 0 exposed substantial unused effect at `0.08` while seed 1 did not. That study should hold the endpoint, reader, and data fixed; evaluate several predeclared ceilings on both seeds; and use a task/safety Pareto read rather than maximizing DEV accuracy.

In parallel, the seed-0 CE/KL starvation suggests the next training recipe should not assume the landing's objective allocation remained healthy. Any Stage 2A training proposal should first separate amplitude-limited conversion from objective-share drift.

The four-probe reader should remain an engineering option rather than a promoted scientific lever. Its matched-recall precision increase was `0.129` percentage point, `pi_dep` was unchanged, the margin difference was negligible, and the registered task read lost two paired rows. Keeping the mean reader also preserves continuity with the replicated Arm S results.

### Questions for strategy

1. Should the next Stage 2A experiment be a replicated, no-selection amplitude surface with the P3.5 EMA endpoints fixed and several predeclared ceilings, or should objective-share repair precede any amplitude read?
2. If training resumes, should the seed-0 CE/KL imbalance become a shaper only after a fresh observe-mode calibration, consistent with the tripwire-versus-shaper doctrine?
3. Should the no-training P3.4 state-extraction pass needed by the Sidecar v2 fingerprint track run in parallel, with its row manifest and checkpoint hashes locked before extraction?

## 10. Limitations

- All task results are exploratory DEV reads on a reused 1,024-row panel.
- Only two Arm S seeds and one Arm R seed were registered.
- The secondary ceiling matrix is selection-barred and cannot nominate a checkpoint or change the branch.
- Arm R required a deterministic 20-step replay after Colab termination. Tensor state identity passed, but the reserialized checkpoint archives are not byte-identical; both hash sets and the superseded partial artifacts are retained.
- Zero measured collateral applies only to the registered audit population and write magnitudes.
- The T1 fingerprint track is build-blocked on absent state artifacts, not scientifically tested.

## 11. Plain-language summary

We tried to make the small improvement from the previous phase settle into a more reliable endpoint. The new landing procedure did make the scores move less from checkpoint to checkpoint, but the final improvement became smaller: four extra correct answers in one run and six in the other, out of 1,024 questions. Our rule required an average of at least eight to keep this line moving directly toward confirmation. The observed average was five, so the correct registered decision is to return to mechanism and effect-size work rather than spend the sealed exam.

The internal correction mechanism itself is still alive. It captures useful correction directions, opens selectively, and caused no measured collateral failures. The new clue is amplitude: one seed gained fifteen rows when read at a wider, predeclared secondary ceiling, while the other seed did not benefit. That is not a result we may select after seeing it, but it is a good next experimental question.

The alternative probe reader behaved almost exactly like the original reader internally, but it did not improve the final task result. It finished with two extra correct answers over base versus four for the matched original reader. The two systems differed on only two rows, both favoring the original reader. The tiny gate-precision change is not enough to justify changing the scientific default.

## 12. Canonical artifacts

- Final receipt archive: `artifacts/p35_20260815/p35_final_receipts.tar.gz`, 50,245,177 bytes, SHA-256 `6b65846ddb64459539a9a239e2b33535120a62c930156943010187f5d0607b8b`.
- Transport manifest: `artifacts/p35_20260815/p35_final_receipts_manifest.json`, SHA-256 `a6585bda3aec13502a30b5e860cbce9787ac139a0a3cba098d4f56ddc308a73c`.
- Analysis: `outputs/stage5/stage5_paper2_phase3_p35_20260815/analysis/analysis_summary.json`, SHA-256 `e0fbea353c9a62f70bea1f14bfa36e1ea99824817724b91c9b63a595c39b596d`.
- Trajectory figure: `docs/figures/paper2_p35_results_20260815.{png,svg}`; PNG SHA-256 `4148823f69cf2bba6276873fa6f42ea64ff7c6cb6d2688d32c798fa77b0d2c09`, SVG SHA-256 `5dc0940f87addbc4899544f32cbc5d7cd24b1ce159b20df06b6115beda569adf`.
- Endpoint figure: `docs/figures/paper2_p35_endpoint_matrix_20260815.{png,svg}`; PNG SHA-256 `58317ccf68e2bca43ac02782c5e98481bb9a8759abc32f8cfeec936b16eb3395`, SVG SHA-256 `c5ffb22a5ae0038ce2d46b492041bcad16b6db285b0d2359ef198d1cb2eda9db`.
- Interruption receipt: `docs/PAPER2_P35_ARM_R_INTERRUPTION_RECEIPT_20260815.json`.
- Recovery receipt: `docs/PAPER2_P35_ARM_R_RECOVERY_RECEIPT_20260816.json`.
- Sidecar v2 desk receipt: `docs/PAPER2_SIDECAR_V2_DESK_BUILD_RECEIPT_20260815.json`.

| Drive artifact | File ID |
|---|---|
| Receipt archive | `1mEoo_UoQO-hUzY9IiEt3PRjaH06cKmm2` |
| Transport manifest | `1bLoIzUXmKKlBOxNBnJ_XEFjk4CZLHuTV` |
| Analysis summary | `1Dp6E9V_gIjnt0Vx76lf63qDkVgTi_Pih` |
| Trajectory PNG / SVG | `1PPOOlHYBxD65cwcRUF5ZplsxjWV1BueT` / `1kMz-8W_K2BAkaNX4AhFaErdatPm3nCU8` |
| Endpoint PNG / SVG | `1CWJamDP3T9_1iWctde2Ld3G8X5_0md5P` / `15KwjrfJF80y1BY8TQ1U9x0bsGwOc24xN` |
| Interruption / recovery receipts | `10Z8nF-Ucr1qH1y4oGAaFE5QDj_FoEb78` / `1ogsUHhQvqB9gFzEOO3bOhgCq4C46LseY` |
| Sidecar v2 desk receipt | `1Pp3M0ezb9x91ZWE6UHkUXLa1d9Q45kLn` |
| This handoff | `1F4jqPll2hefdfobpXPXx9lk5iLosh6zA` |
