# Coding to Strategy Handoff - Bicameral W2-prime Phase D Desk Gate

Date: 2026-08-25. Status: COMPLETE CPU-ONLY RESULT; registered key `HEMISPHERES-UNINFORMATIVE`. Phase G did not open and remains prohibited. D5 remains in force; Step-2 remains blocked; CONFIRM and EVAL-E remain sealed.

## 1. Executive result

The repaired W2-prime desk battery completed on both seeds under the ratified authority chain. The primary prompt-only FS-1 map narrowly clears the registered D1 conditional-cosine threshold in both seeds, but the hemisphere-conditioned map does not improve prediction over a matched-rank single-stream base-state map. The relative risk reduction is slightly negative in both seeds. The richer prompt-only trajectory feature set FS-2-prime fails both gates.

The registered machine key is therefore **`HEMISPHERES-UNINFORMATIVE`**. This is a desk-gate negative for the present two-hemisphere representation, not a claim that correction directions are wholly unpredictable. Phase G was not staged and no GPU was provisioned.

Plain language: the model's prompt state contains a weak, reproducible clue about the desired correction, but splitting that state into the two current hemispheres does not add useful information beyond what the ordinary base state already contains. The extra depth history makes prediction worse, not better.

## 2. Authority and lock

- Governing charter: Drive `1jfIkThIq_ts5_oxS_Rck-sTiQ6El4bvd`, 13,699 B, SHA-256 `f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4`.
- Binding D4 rulings: Drive `1RQrwGdlmVK_WLqL7zuEm_tsQsHhqZT4S`, 12,618 B, SHA-256 `34352161cb69612bfc996658fab0f2d24eed381cc3895eda99a7c5a3d2e835fd`, verified from exact Drive bytes.
- Machine lock: 3,029 B, SHA-256 `bd727b9eba6e524392b33847247c909e7b6514ca57dafdb18caffb6391b0fffc`.
- Implementation/result commit: `1582e7faf639d1dce6c037aebbec895a56bfeaa8`.

R-W2P-1 through R-W2P-3 are implemented exactly in the decision path: `L0a` loss-gradient is the sole gate-bearing target; `L0d = h_gold - h_free` is diagnostic only; the gold-conditioned W3 FS-2 is retained as `BLOCKED_SOURCE_CONFLICT`; FS-2-prime uses only prompt-only D4 states; and SL-3 nested cross-fitting separates hyperparameter selection from every reported outer-fold prediction.

## 3. Registered feature and estimator contract

The pre-fit receipt was written before any fit: 1,693 B, SHA-256 `f96fa283f4b560dd0308ec225a4c163e8c67578c277e1e6f4be229894a6be6d9`.

- FS-1: interface mean `m18` and difference `d18`.
- FS-2-prime: FS-1 plus prompt-only mean and difference histories at sites 8, 12, and 16, and adjacent deltas through site 18.
- FS-0 control for FS-1: base state at site 18 with matched total rank.
- FS-0-prime control for FS-2-prime: base states at sites 8/12/16/18 plus adjacent deltas, concatenated, with matched total rank.
- Selection: four outer folds; three inner folds; ranks `{2,4,8,16,32}`; seven fixed ridge multipliers; per-block inner selection followed by an outer-train joint refit.
- D1: held-out conditional cosine against the population-mean-centered target. Secondary read uses opposite-fold, within-cluster rank-8 R-S0-A nuisance removal plus the held-out cluster direction.
- D2: paired held-out normalized-risk reduction over the matched single-stream control. Gate is at least 5% in both seeds.

One first-pass CPU receipt is retained as superseded archaeology. It had the correct decision metrics but an incomplete secondary nuisance read and omitted two requested D2 control tables. No GPU or scientific state was touched. The corrected contract-complete rerun produced the numbers below and is the sole result of record.

## 4. Primary results

### D1 and D2 gates, `L0a` sole gate family

| Feature set | Seed | Conditional cosine (95% bootstrap CI) | D1 | Relative risk reduction vs matched base (95% CI) | D2 |
|---|---:|---:|---|---:|---|
| FS-1 | 0 | 0.306 [0.272, 0.339] | pass | -0.7% [-5.9%, 4.3%] | fail |
| FS-1 | 1 | 0.323 [0.290, 0.357] | pass | -1.1% [-6.5%, 3.8%] | fail |
| FS-2-prime | 0 | 0.248 [0.211, 0.285] | fail | -10.6% [-20.7%, -1.5%] | fail |
| FS-2-prime | 1 | 0.249 [0.213, 0.286] | fail | -6.9% [-14.9%, 0.8%] | fail |

Both seeds select the minimum rank pair `(2,2)` for FS-1 and `(2,2,2,2)` for FS-2-prime. The agreement is useful: the negative is not caused by divergent model selection.

### Nuisance-deflated and control reads

- FS-1 R-S0-A residual cosine is only 0.0115 / 0.0183 by seed, far below the raw 0.306 / 0.323. The nominal D1 signal is therefore dominated by broad shared structure rather than a robust row-specific residual map.
- The mean-only state improves over the base state by 2.1% / 3.3%, but both confidence intervals include zero and both are below the 5% gate.
- The hemisphere-difference block carries 62.5% / 62.5% of the fitted FS-1 contribution energy. That is not evidence of unique information: the full `(m,d)` fit is still worse than the matched base-state control.
- Raw `(h_A,h_B)` coordinates lower risk in seed 0 but not seed 1. The coordinate sensitivity does not replicate and cannot rescue the registered `(m,d)` gate.

### Target-family diagnostic

The diagnostic-only teacher-forced state delta `L0d` is more conditionally predictable than `L0a` (FS-1 cosine 0.404 / 0.429), but hemispheric conditioning is worse than the base control by 12.2% / 9.9%. This supports its pre-registered demotion: predictability alone does not make a target causally useful or hemisphere-specific.

## 5. D3 and D4 diagnostics

The FS-1 frozen-map branch-pair shuffle raises normalized risk to 1.234 / 1.282, and permuting all inputs raises it to 1.270 / 1.287. The fitted map is using row-specific prompt information; the D2 negative means that information is also available in the ordinary base path.

No D4 site has replicated cross-hemisphere incremental value. Seed 0 crosses 5% in both directions at site 8, but seed 1 does not; seed 1 has a smaller A-to-B effect at site 12, not replicated by seed 0. The interface-site effects are near zero and inconsistent in sign.

The A-3 composition desk read remains strongly battery-dependent: `L1-L2` is +0.073 on GSM8K and +0.164 on MBPP, but -0.872 on ARC-Challenge. This reinforces the existing warning that the banked clusters and branch effects are entangled with task family.

## 6. Interpretation

The W3 headline cosine of about 0.885 cannot survive the deployability correction. Under prompt-only features and fully nested held-out estimation, the honest number is about 0.31 for FS-1, and the cluster-aware nuisance-deflated residual is about 0.01-0.02. That is a major downgrade, but it is informative rather than ambiguous.

The result separates two claims:

1. **A weak deployable state-to-correction relation exists.** D1 narrowly passes in both seeds; row shuffles damage the map.
2. **The current hemispheric split does not add predictive value.** D2 fails in both seeds, the richer trajectory features worsen risk, and the site screen does not replicate.

The strongest reading is that the current branches are correlated transforms of information already present in the base state, not complementary specialists for this target. This closes Phase G for this W2-prime design. It does not close conditional correction maps generally, a single-stream map, a different specialization objective, or a future architecture in which cross-hemisphere exchange is trained rather than inherited.

## 7. Limitations and do-not-claim list

- The desk panel has 256 rows and is dominated by GSM8K (216 rows); ARC-Challenge has 32, MBPP 5, and the remaining cohorts have one row each. Only pooled, GSM8K, and ARC-Challenge reads have useful support.
- D1's raw pass is close to the 0.30 threshold and its seed-0 confidence interval crosses that threshold. The registered verdict uses the point-estimate rule, but the result is not a strong-margin pass.
- R-S0-A clustering is target-family-specific and frozen deterministically before each fit; small clusters contain 21-22 rows for `L0a`, so the secondary residual read is low-power.
- Closed-form maps test predictability, not task capability. No generative scoring occurred.
- Do not claim that the two branches encode abstract reasoning modes, that the 0.885 W3 value was deployable, that `L0d` is useful because it is predictable, or that all conditional correction maps are impossible.

## 8. Registered disposition and recommended next decision

Machine key: **`HEMISPHERES-UNINFORMATIVE`**. `phase_g_authorized_by_desk_result=false`. Phase G must not run under the current charter.

Recommended strategy decision: bank W2-prime as a clean architecture-specific negative and compare the surviving options against the evidence. The cheapest live direction is a parameter-matched single-stream conditional map, because the base state matched or beat both hemispheric feature sets. Any renewed bicameral program should first change the source of complementary information - specialization training, task-balanced routing, or an explicit exchange mechanism - and must earn a new desk D2 advantage before generative GPU scoring.

## 9. Receipts and verification

- Final Phase-D summary: Drive `1a40vtQRK4g7ehQBItgm2XBA3YN7fELa1`, 549,319 B, SHA-256 `8a36b286db0c16ead48a4678202d41e41ac6e75966a240c16c75bec2326bdce5`.
- Receipt manifest: Drive `1aGPsrHYrJ_PRt55N2h7KqGKMbq1UAmfb`, 3,218 B, SHA-256 `4379fb7591d61ccae064fde421ca44774a640c5da20f512f7512839012d73960`.
- Figure PNG: Drive `1cETuaUa-BDYcVQlWQGU9WAOVAM6ju-rB`, 206,224 B, SHA-256 `1692ce40646045b7be64259ee672b3f1d6f0335aec1dc225a779de0b0e00fc9b`.
- Figure SVG: Drive `1HMdQYJm2HmcNVcsbPSYSik3IFP4DxCNt`, 119,868 B, SHA-256 `3e06c2ad9ceb985e234c5b964589e88c9edebdf2813f59e6c73d42904f964a8c`.
- Full public/private artifact bundle: Drive `1kEt7j7sUMD_Mz-WNhU8X121EWZ_8aZfc`, 1,207,253 B, SHA-256 `b088a19059c00683a29a7b266aa0348f92b375738529c1013dea1bb9660490f2`.
- Tests: 20 passed in 5.92 seconds (`test_paper2_bicameral_w2p`, base Bicameral tests, Stage-0 tests).
- Independent assertions: all summary values finite; key recomputed as `HEMISPHERES-UNINFORMATIVE`; optimizer absent; optimizer steps 0; CONFIRM false; EVAL-E false; Phase G false.
- Corrected CPU fit elapsed approximately 149 seconds from pre-fit receipt to final summary.
- GPU spend for D1-D3: zero. This CPU wave opened no Colab session. A live CLI session enumeration was not refreshed at closeout because local Colab CLI authentication had expired.

Figure: `artifacts/bicameral_w2p_20260825/figures/paper2_bicameral_w2p_phase_d.svg`.

## 10. Plain-language close

The two-hemisphere idea did not earn its expensive test. We can weakly predict the correction from the prompt state, but the ordinary single-stream state predicts it at least as well as the current two branches. The safety gate saved the generative GPU wave. The useful surviving insight is narrower: there is a small prompt-only correction signal worth pursuing, but this inherited hemispheric decomposition is not yet the mechanism that isolates it.
