# P3.3 Verification Handoff: The Write Path Is Live, the 0.15 Negative-Control Premise Is Not

Date: 2026-08-12. Status: read-only verification complete; i1 not launched; P3.4 unauthorized.

## 0. Executive reading

The suspicious zero-collateral and perfect-retention results are not explained by a dead evaluation write path. Both seeds apply nonzero hidden-state writes to every negative and retention row; fixed-pair margins move on every gate-open row in FP32 telemetry; and the identical deployed path registers 101 positive corrections. The serving-precision BF16 re-score also reproduces every cached base token across all 34,816 rows per seed.

The registered V3 control nevertheless fails as written: forcing the trained direction to radius 0.15 changes no negative top-1 prediction in either seed. This does not identify a dead instrument. The same negative path begins registering trained-direction flips at radius 0.6 and registers 1,107 pooled flips at radius 1.0. The result instead falsifies V3's assumption that the max-confidence negative slice must contain a boundary within radius 0.15 along the learned direction.

The corrected BF16 values of record are `pi_dir = 557/3,738 = 14.901%` and `pi_dep = 101/515 = 19.612%`. The earlier 14.64% and 13.87% estimates came from deleting FP32-reader-mismatched rows rather than rerunning all rows through the BF16 serving reader. They were useful sensitivity estimates, but they are not the canonical re-score.

## 1. Authorization and sequence

Governing strategy memo: `STRATEGY_P33_RESULT_RESPONSE_I1_20260812.md`, Drive `1wZ5DQjXFUu70DS2GUyLWlMxLX758nhsw`, 7,755 bytes, SHA-256 `556354c04dd39f7592a34d8ba1631e1e8f594ba9b8f098289d0cd65a49b6dee7`.

The prescribed order was followed:

1. V1 nonzero-delta check.
2. V2 same-path check.
3. V3 forced-open negative control at radius 0.15.
4. BF16 serving-reader re-score.
5. i1 only after the verification result.

No optimizer was constructed, no parameter was updated, and no CONFIRM or task-level capability partition was scored. Because V3 did not pass under its literal criterion, i1 was not launched.

## 2. Implementation and controls

The verification uses the final step-1,000 P3.3 checkpoints:

| Seed | Final checkpoint SHA-256 |
|---:|---|
| 0 | `84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e` |
| 1 | `e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067` |

The frozen Phase 2 state is reconstructed from each checkpoint's asserted migrated source, then the 16 trained P3.3 tensors are overlaid. The first attempt correctly stopped before scoring when the verifier tried to reconstruct the model from those 16 tensors alone. The fixed loader asserts both source and final checkpoint hashes.

Canonical token decisions use an end-to-end BF16 tied-embedding matrix multiplication. V1 margin telemetry uses the same BF16-selected winner/runner-up pair but evaluates that fixed pair in FP32, because BF16 quantizes many real sub-token margin movements to zero. Both values are retained in the private rows.

The positive and negative audits call one `_model_components` path and one deployed-hidden reader, tagged `p33_model_components_deployed_hidden_bf16_reader_v1`. Additional diagnostic states are evaluated by the same BF16 output projection.

## 3. Results

### 3.1 Canonical BF16 re-score

| Metric | Seed 0 | Seed 1 | Pooled |
|---|---:|---:|---:|
| `pi_dir` | 281/1,869 = 15.035% | 276/1,869 = 14.767% | 557/3,738 = **14.901%** |
| `pi_dep` | 50/257 = 19.455% | 51/258 = 19.767% | 101/515 = **19.612%** |
| BF16 base-reader match | 17,408/17,408 | 17,408/17,408 | 34,816/34,816 per seed pair |
| Deployed negative collateral | 0/12,288 | 0/12,288 | **0/24,576** |
| Retention harm | 0/1,024 | 0/1,024 | **0/2,048** |

Pooled document-bootstrap intervals are 13.282–16.521% for `pi_dir` and 14.807–24.341% for `pi_dep`. The registered `pi_dir` verdict remains in the middle band.

### 3.2 V1: nonzero write and margin movement

V1 passes in both seeds.

| Population | Seed | Gate-open rows | Exact-zero FP32 margin deltas on open rows | Nonzero hidden writes |
|---|---:|---:|---:|---:|
| Negative | 0 | 1,855 | 0 | 12,288/12,288 |
| Negative | 1 | 1,786 | 0 | 12,288/12,288 |
| Retention | 0 | 184 | 0 | 1,024/1,024 |
| Retention | 1 | 172 | 0 | 1,024/1,024 |

Median absolute FP32 fixed-pair margin movement is 0.0196 and 0.0194 on negatives and 0.0215 and 0.0216 on retention rows. BF16 reports nonzero margin movement on only about one third of rows because its score quantum is visible in 0.0625 increments; this is quantization, not absence of a write.

### 3.3 V2: same deployed path

V2 passes. The same deployed path records 50 and 51 positive corrections while recording zero negative collateral. Positive and negative rows do not use separate writeback implementations.

### 3.4 V3 and the radius diagnostic

V3 fails literally at the registered radius:

| Trained-direction radius | Seed 0 negative flips | Seed 1 negative flips | Pooled rate |
|---:|---:|---:|---:|
| 0.15 | 0 | 0 | 0.000% |
| 0.30 | 0 | 0 | 0.000% |
| 0.60 | 12 | 11 | 0.094% |
| 1.00 | 630 | 477 | 4.504% |

A runner-up-directed control at radius 0.15 also yields zero flips. That control establishes that the max-confidence agreement population itself has no sampled top-1 boundary at the assumed radius, even along the local runner-up direction. At larger radii the reader and path plainly register changes.

## 4. Interpretation

The combined evidence rejects the dead-path explanation:

- Every row receives a nonzero hidden perturbation.
- Every gate-open row has a nonzero FP32 fixed-pair margin change.
- The exact same deployed path changes positive predictions.
- The negative reader registers changes once the trained-direction radius reaches the observed boundary regime.

The unresolved item is therefore the strategy memo's V3 inference, not the evaluator. V3 assumed that prior V-series collateral at radius 0.15 transfers to a newly selected max-confidence negative population and to the learned P3.3 direction. Neither transfer is guaranteed, and both are contradicted here. Zero collateral at gamma 0.02 is now strongly supported as selectivity on this slice, but it should be described as bounded to these max-confidence negatives and the token-retention panel.

The BF16 re-score also corrects an earlier methodological shortcut. Filtering rows on which the FP32 reader happened to agree with the BF16 cache changed the evaluation population and disproportionately removed easy flips. A full BF16 rerun preserves all rows and is the only canonical result.

## 5. Decision requested from strategy

Recommended ruling:

1. Bank V1 and V2 as passed.
2. Bank V3 as a failed radius assumption, not an instrument failure, using the 0.15-to-1.0 curve as the positive control.
3. Accept 14.901% `pi_dir` and 19.612% `pi_dep` as the canonical BF16 P3.3 values of record.
4. Authorize the already specified i1 iteration without changing its losses, cohorts, gamma, audit slices, or reader.
5. Scope the zero-collateral claim to the registered max-confidence agreement and retention populations.

If strategy insists that V3 must produce a flip at exactly 0.15, i1 remains blocked and the correct next action is to redesign the positive-control population before any training. Silently changing the radius or treating a 0.6 result as a 0.15 pass would be improper.

## 6. Artifacts

| Artifact | Location | SHA-256 |
|---|---|---|
| Public summary | Drive `1fAvl-_DCmotHJgi59vUgnLLPm-6kRuFz` | `4980e986ba67a4dcdcd712f512ea57ac5abd9a41f19e0e5eb97efd234c5a7d2b` |
| Full row archive | Drive `1JI1Ec6RHaRvklbBPFZzi0KZVZfmvKeCP` | `bdeb4b32a7dec89070332f200327052881f59f11a5b6af6b3f5dbb732f864b93` |
| Transport receipt | Drive `1ftfvwf9zbs9wlDuR-mkDOVLmbHCDtQ_0` | recorded inside the receipt |
| Repo summary | `outputs/stage5/stage5_paper2_phase3_p33_verification_20260812/summary.json` | `4980e986ba67a4dcdcd712f512ea57ac5abd9a41f19e0e5eb97efd234c5a7d2b` |

Implementation commits: `1965b1b1` (verification target), `eb7f0986` (source-lineage reconstruction), and `83421cfc` (precision and radius diagnostics).

## 7. Plain-language summary

The evaluator is not asleep. The bridge nudges every checked row, those nudges measurably move output margins, and the same code changes many wrong answers. The reason no protected answer changes is that these protected rows are unusually far from a decision boundary and the deployed write is tiny. Even a write 7.5 times larger still changes none of them; at 30 times the deployed scale a few begin to move, and at 50 times many do. The original check assumed radius 0.15 had to be large enough on this population. The data show that assumption was wrong. The scientific question is now whether strategy accepts that localization and lets the aim-focused iteration proceed.
