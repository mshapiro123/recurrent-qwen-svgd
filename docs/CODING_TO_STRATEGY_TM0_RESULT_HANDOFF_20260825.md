# Coding to Strategy Handoff - TM-0 Trajectory-Memory Recon and Jet Geometry

Date: 2026-08-25. Status: COMPLETE REGISTERED RESULT. Machine keys: `STITCH-DEAD` and `ROTATION-ABSENT`. TM-2 displacement fitting did not open under G-TM1. No training, optimizer, or injection occurred. CONFIRM and EVAL-E remain sealed and unscored. D5 and the Step-2 block remain in force.

## 1. Executive result

TM-0 completed over the frozen 6,144-row prompt-only panel with Qwen2.5-0.5B student states and Qwen2.5-7B/14B teacher states. The two registered questions both returned negative answers.

1. **The teacher states cannot be transported into the student frame with the registered cross-fitted linear stitch.** Raw-space reconstruction looks nearly perfect, but that result is carried by broad shared scale and mean structure. In the whitened frame, no tested layer reaches the required 0.20 relative-MSE improvement on both held-out halves. The registered key is `STITCH-DEAD`.
2. **Successful teacher computation does not exhibit reproducible turning planes under the primary active-token-mean jet estimator.** Curvature is physically present, but the success strata do not show stronger plane consistency or pivot structure than `D_none` and the smooth-noise control. The registered key is `ROTATION-ABSENT`.

Plain language: the larger models clearly move through their hidden states, but this experiment does not find a stable coordinate map or a success-specific rotation pattern that the small model can reuse. The trajectory-memory intervention therefore does not earn construction under the present charter.

## 2. Authority and scope

- TM-0 r2 charter: Drive `1MVnzhL0oYoxm_B5tFJDzfFNAsKajmIXg`, 20,221 B, SHA-256 `3103ca7f81367f3a47cea0ec1b2f92de73ce0eae27240aae46e57eaa044e9460`.
- Ratified r3 execution order: Drive `1gQHhrUHlRN_l2cf3GfC1_icQnKIU009a`, 14,769 B, SHA-256 `c93392b5...bfc2`.
- Binding preflight rulings: Drive `1w-rRoDDkhUQKvAGkQluqcMMtan-OywRF`, 9,867 B.
- Ratified r4 jet amendment: Drive `1GDZE-YnYU-RNHoBcMWBKcuWXjW3pxyaH`, 10,910 B, SHA-256 `aa354b8bd6735d2780ff7afb25925e9cb08cc325898495f6dd22146dd880080a`.
- Core r4 implementation commit: `1492ea65`. Durable recovery and score-transport commits: `5bf7a361`, `24457ea1`, and `a2e64590`.

The experiment remained forward-only. Qwen2.5-32B was deferred under S-2; no quantized substitute was introduced. The state cache used prompt-only inputs and sequential execution. Gold answers appeared only in the separately pinned correctness reader.

## 3. Population, lineage, and score recovery

The frozen panel contains 6,144 rows: 4,963 GSM8K, 1,128 ARC-Challenge, 40 MBPP, six ARC-Easy, six MMLU, and one Tier-1 row. It contains 6,034 verified-train rows and 110 DEV rows. Panel SHA-256: `e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca`.

State caches were verified shard-by-shard: 96 shards per model, 6,144 rows each, prompt-only and sequential. Model-bundle hashes are:

- student: `6cf589410562eb23e6ec7aaa5f322301fc583b0133a30f68300f0c51a60429ab`;
- teacher 7B: `fbba8216b379965a43dc357e95c6806936e7dbf0b4ec7e4475cde55fa33da752`;
- teacher 14B: `5a373ccf410758c1a892b5a173dca56addb0baf37f6d8baaf41066d7ba7b92d3`.

The 7B correctness pass survived one Colab backend reset through a 2,452-row durable snapshot. Completion then used deterministic transport partitions on two A100s. Three sources merged to 6,144 rows with 648 overlap rows agreeing exactly. The merged score SHA-256 is `e884180f8545fd964c444bbad304506216ea124498042dcc074ae13b07f766f9`. The official scorer re-read the merged file in `cached_complete_no_model_load` mode and confirmed revision `a09a35458c702b33eeacc393d103063234e8bc28`, reader lineage, generation batch 8, full coverage, and zero CONFIRM rows.

Both A100 sessions were terminated after local verification; `colab sessions` returned no active sessions.

## 4. TM-1 calibration and stitch gate

The exact debiased linear-CKA calibration is stable across two disjoint 512-row subsets:

| Teacher | Subset A selected layer | Subset B selected layer | Stability gate |
|---|---:|---:|---|
| 7B | 7 | 8 | pass, difference 1 |
| 14B | 10 | 10 | pass, difference 0 |

The cross-fitted ridge stitch then tested the calibrated layers and registered neighboring/depth-control layers. Every layer failed the whitening-aware MSE gate on at least one held-out half; in fact, no layer reached 0.20 on either model.

| Teacher | Best whitened relative-MSE improvement across all tested cells | Registered threshold | Best raw-space improvement | Result |
|---|---:|---:|---:|---|
| 7B | 0.119 | 0.200 on both halves | 0.995 | fail |
| 14B | 0.083 | 0.200 on both halves | 0.995 | fail |

The cosine-over-random gate passes comfortably (roughly 0.32-0.55), so the map is not random. The decisive failure is scale-sensitive reconstruction after whitening. Raw-space values near 0.995 are therefore not evidence that teacher corrections are transportable; they mostly reflect broad shared structure.

Registered key: **`STITCH-DEAD`**. TM-2 displacement fitting and every downstream injection remain prohibited by G-TM1.

## 5. TM-2g-J jet result

The r4 analyzer reads exact scalar Gram invariants in per-model whitened frames. It uses a frozen sparse WHT/JL metric and 256 frozen Gaussian plane probes; no learned projector or cross-scale orientation comparison is present. The primary gate uses active-token-mean states on GSM8K for `D_7>0.5` and `D_14>0.5` in both teachers.

### Primary decisive cells

| Teacher | Success stratum | Rows | Plane consistency minus `D_none` (95% CI) | Plane consistency minus smooth noise | Two-half balanced accuracy | Claim |
|---|---|---:|---:|---:|---:|---|
| 7B | `D_7>0.5` | 1,902 | +0.0012 [-0.0005, +0.0030] | -0.0821 | 0.586 / 0.595 | fail |
| 7B | `D_14>0.5` | 1,426 | +0.0005 [-0.0013, +0.0023] | -0.0827 | 0.638 / 0.662 | fail |
| 14B | `D_7>0.5` | 1,902 | -0.0023 [-0.0037, -0.0009] | -0.0354 | 0.604 / 0.590 | fail |
| 14B | `D_14>0.5` | 1,426 | -0.0027 [-0.0040, -0.0012] | -0.0355 | 0.635 / 0.628 | fail |

The classifiers can distinguish some success strata above chance, but the distinguishing information is not the registered consistent-plane or pivot signature. Success trajectories are materially *less* plane-consistent than the step-norm-matched smooth-noise null in every primary cell. The full per-layer profiles for success and `D_none` nearly coincide.

Registered key: **`ROTATION-ABSENT`**.

### Secondary and prediction-ledger reads

- Curvature is nonzero on all rows at all analyzed layers for both teachers. This is a calibration fact, not evidence of success-specific geometry.
- The last-active-token secondary shows a small positive GSM8K plane-consistency contrast in three of four teacher/stratum cells (about +0.0014 to +0.0026). It was not the primary estimator and does not change the key. It is a reasonable future hypothesis only under a newly registered estimator.
- P-TMg-4 is directionally correct in all four measured cells: hard-but-teacher-solvable rows show more pivot events than `D_all`, but absolute pivot rates are tiny and do not satisfy the registered claim gate.
- P-TMg-5 fails in all four cells: late velocity-acceleration alignment is not more negative on success rows.
- The failed-loop post-diction is `ARCHIVE-ABSENT`; no matched per-loop state archive was regenerated or substituted.

## 6. R-1 rider

The no-refit GSM8K granularity read from the banked W2-prime fit survives weakly:

| Seed | GSM8K conditional cosine (95% CI) | Pooled cosine | GSM8K minus pooled |
|---:|---:|---:|---:|
| 0 | 0.290 [0.252, 0.329] | 0.306 | -0.016 |
| 1 | 0.313 [0.276, 0.349] | 0.323 | -0.010 |

Interpretation: a weak deployable state-to-correction relation remains within GSM8K, but it is not strengthened by the current hemispheres and it does not rescue trajectory memory after `STITCH-DEAD`.

## 7. Interpretation

TM-0 separates three superficially similar statements:

1. **Representations correlate across scale.** True. CKA and raw-space fits are high.
2. **A teacher state can be reconstructed in the student's usable metric.** Not established. The whitened cross-fitted stitch fails everywhere tested.
3. **Successful computation follows reusable rotation planes.** Not established under the primary estimator. Curvature exists, but the success-conditioned plane signal does not clear controls.

The important lesson is that raw representational similarity is not enough to justify transporting a correction. The same broad structure that creates high CKA and near-perfect raw MSE can disappear when each direction is put on equal footing. Likewise, visible curvature is not enough: a memory requires success-specific, stable geometry, not merely a curved trajectory.

The joint W2-prime/TM-0 branch is now negative on both student-side conditional hemispheric information and teacher-side registered trajectory transport. Under the current decision table, TM-1-prime should not be built from these artifacts.

## 8. Limitations and do-not-claim list

- The panel is dominated by GSM8K. ARC-Challenge provides a useful secondary population; MBPP and the remaining batteries are underpowered for geometry claims.
- The stitch tests cross-fitted linear ridge maps. It does not prove that every nonlinear, layerwise, or jointly trained map is impossible.
- The r4 metric is a frozen approximate WHT/JL plus RMT-whitening implementation. Scalar invariants are stitch-free, but the approximation should remain disclosed.
- The 32B rung was not run. No conclusion extends to 32B internal geometry.
- The last-token secondary signal is not a registered positive and must not be selected post hoc.
- The failed-loop archive was absent, so P-TMg-3 is unscored rather than wrong.
- Do not claim that teacher and student states are unrelated, that teachers do not reason geometrically, that all trajectory memory is impossible, or that the weak R-1 relation is causally useful.

## 9. Registered disposition and requested strategy decision

Bank the wave under the paired keys **`STITCH-DEAD`** and **`ROTATION-ABSENT`**. Do not construct TM-1-prime or run TM-2 injections under this charter.

Recommended next action: apply the r3/r4 branch map literally and prepare the honest-options memo for the accumulated boundary. If strategy wants to retain one low-cost scientific thread, register the last-token jet effect as a fresh estimator hypothesis and test it on a balanced, independently frozen population. It should not delay closure of the present TM line.

## 10. Receipts and verification

- Consolidated CPU status: `artifacts/tm0_20260825/results/tm0_cpu_pipeline_status.json`, status `COMPLETE_STITCH_DEAD_JET_COMPLETE`.
- CKA summary SHA-256: `a7b7eaa7b575611f90d7b88062989e750578ce697868ba4b2c85ca2bee8e0fca`.
- Stitch summary SHA-256: `ed5b9e0e660f80901350289ffc9f215a140cc9fc49b4a8cc46d0260598cf2a22`.
- Jet summary SHA-256: `1e94db80588687a979d46d3ea1a6a98b2317713fb6ab76537259c2b5ed952582`.
- R-1 receipt SHA-256: `8c9fcb3d4e59178ebee8a894dbdce09c6d1109fd088ffc37973c592fb30cfacd`.
- Merged 7B score SHA-256: `e884180f8545fd964c444bbad304506216ea124498042dcc074ae13b07f766f9`; 6,144 rows; 648 exact overlap rows.
- Tests: 61 relevant tests passed in 9.37 seconds across TM-0 and the inherited Bicameral contracts.
- Independent end-to-end rerun reproduced `STITCH-DEAD` and `ROTATION-ABSENT`.
- Figure QC: all PNG/SVG pairs render; no missing series, obscured legends, clipping, or blank panels.
- Compute teardown: no active Colab sessions at closeout.

Primary figures:

- `artifacts/tm0_20260825/results/tm1_stitch_gates.svg`
- `artifacts/tm0_20260825/results/tm2g_jet_profiles.svg`
- `artifacts/tm0_20260825/results/tm2g_jet_decisive_contrasts.svg`

## 11. Plain-language close

The teachers move, but we did not find a reusable route map. Their hidden states look similar to the student's until we remove the easy shared scale structure; then the translation fails. Their trajectories curve, but successful examples do not curve through stable planes that separate them from failures and matched noise. That closes this version of trajectory memory before we spend a training run on it.
