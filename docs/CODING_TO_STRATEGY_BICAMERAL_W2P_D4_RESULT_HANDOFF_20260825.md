# Coding Handoff to Strategy — W2′ D4 Banked; Phase-D Target Rulings Required

Date: 2026-08-25. Governing charter: `STRATEGY_BICAMERAL_W2P_CONDITIONAL_MIXER_CHARTER_20260825.md`, Drive `1jfIkThIq_ts5_oxS_Rck-sTiQ6El4bvd`, 13,699 bytes, SHA-256 `f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4`.

## 1. Bottom line

D4 is complete, valid, and banked. The prompt-only Bicameral state cache covers the frozen 256-row Stage-0 manifest for both seeds at Qwen layer endpoints 8, 12, 16, and 18. Every final-interface state matched the registered unchunked path bit-exactly for base, branch A, and branch B. The job used 31.71 GPU-seconds, versus the 900-second cap, and the A100 has been released.

D1–D3 have not run. Two genuine authority conflicts were detected before target fitting: the charter's `L0c` symbol names the banked margin-gradient tensor while its prose describes the banked `L0d` teacher-forced state delta; and the banked W3 trajectory features registered as FS-2 were extracted from prompt-plus-gold sequences, violating this charter's strict deployment-input boundary. No silent choice was made.

## 2. Experiment executed

The D4 cache ran Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775` on the exact W0 runtime: NVIDIA A100-SXM4-40GB, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, BF16 SDPA. Inputs were the frozen 256-row manifest, the banked partition rows, and the two seed-specific branch initializers, each checked by byte count and SHA-256 before model loading.

For each prompt and seed, the evaluator cached pooled base, branch-A, and branch-B states at sites 8, 12, 16, and the layer-18 write interface. Branch calls remained sequential; batch concatenation was prohibited. The prompt formatter's target component was discarded before token construction. No gold token, teacher forward, oracle route, generation score, optimizer, CONFIRM row, or EVAL-E row entered the computation.

## 3. Results

| Read | Seed 0 | Seed 1 |
|---|---:|---:|
| Rows | 256 | 256 |
| Cache seconds | 15.060 | 14.045 |
| Base interface parity | exact | exact |
| Branch-A interface parity | exact | exact |
| Branch-B interface parity | exact | exact |
| Cache bytes | 11,026,839 | 11,026,839 |
| Cache SHA-256 | `758cfb72…7c0` | `388e747e…b55` |

Total measured GPU time was 31.712 seconds. The downloaded archive was independently checked at 19,864,515 bytes and SHA-256 `3664b8d8371c321fb2654047e5c90843d6e8a410033c2ad1279f165b124698bd`.

## 4. Implementation quality and one repaired defect

The code adds a prompt-only multi-site cache method to `BicameralTaskInferenceGraph`, a frozen D4 evaluator, immutable-input runner, machine lock, closed-form conditional-map library, nested cross-fitting analysis, and 15 focused tests. The committed implementation is on `main`; D4 ran from commit `1043cf78e243ac4b4ee8eaf97441aad78a2585a5`.

The first D4 attempt stopped before the first row because the state-digest helper attempted a dtype view directly on a scalar gate parameter. The repair flattened tensors before byte viewing and added a scalar-parameter regression test. The failed attempt consumed 25 seconds, produced no cache or score, and did not alter scientific state. The completed run then passed all parity and immutability checks. A later CLI reply timeout was transport-only: the durable runner had already completed and its archive and status hashes verified locally.

## 5. Statistical contract

The desk map implementation uses four outer folds for all reported predictions and three inner folds for hyperparameter selection. Rank and ridge are selected separately per input block on inner held-out rows, the blocks are jointly refit on each outer training fold, and only untouched outer-fold predictions feed D1 and D2. This avoids reporting the same cross-validation values used to choose among the rank/ridge grid. The frozen deployment map is selected separately after the gate statistic is complete.

No D1 or D2 number exists yet. The D4 states are infrastructure, not evidence that either desk gate passes.

## 6. Required rulings

1. **R-W2P-1 — secondary target.** Recommendation: bind the charter's semantic definition to banked `L0d`, the actual teacher-forced `h_gold - h_free` delta. Do not add banked `L0c` margin gradients as a third confirmatory family after the fact.
2. **R-W2P-2 — FS-2.** Recommendation: withdraw FS-2 from this wave and gate on FS-1 only. Record FS-2 as blocked by a source-provenance conflict, not as a failed model. Any prompt-only trajectory replacement should be prospectively specified for a later wave.
3. **R-W2P-3 — nested evaluation.** Ratify the more conservative `nested_blockwise_inner_cv_then_joint_refit` evaluation rule described above. It changes no rank or ridge candidate and only prevents selection leakage into the gate statistic.

The full clarification memo is on Drive as `CODING_TO_STRATEGY_BICAMERAL_W2P_AUTHORITY_CLARIFICATION_20260825.md`, Drive `1fir_kR5a0RM3AFOXE0UJuMlmBctaToGj`, 4,316 bytes, SHA-256 `a58b7d0b29c6d1e48dfc37ff9649ea3735d64316a396469a3b59e40887121479`.

## 7. Limitations

D4 validates state extraction and evaluator identity only. It cannot adjudicate map mirage, hemispheric incremental value, target-family agreement, or downstream generative utility. Prompt states are pooled over active prompt tokens, so any token-local conditional structure is intentionally outside this wave's estimand. The 256-row panel limits per-battery precision; the registered gates remain pooled and both-seed.

## 8. Plain-language summary

We now have a clean snapshot of what the base model and both hemispheres know at four depths for the same 256 questions. The snapshot is reproducible and did not look at answers. Before asking whether those states predict a useful correction, the plan needs two labels repaired: one document name points to the wrong saved target, and one proposed feature accidentally contains answer-conditioned information. We stopped before either mistake could influence a result.

## 9. Next steps

After the three rulings land, amend the machine lock before execution, run D1–D3 on CPU, and issue the registered desk key. Phase G remains prohibited unless D1 and D2 both pass in both seeds under the resolved target contract. All paid compute is currently off.

## 10. Receipts

- Drive artifact folder: `1PsJDmBmpS-MNrdUCkKO0caATXr8nb3K0`.
- D4 summary: Drive `1ZvdVeCVVnjEVNFlnEIryuhTtCn7Yu6As`, 1,413 bytes, SHA-256 `22d2147a0d3cf2772ffaf7649f0f443603f5e8b48191c62ab0ea3263edad5dd3`.
- D4 status: Drive `1OUiyS8CQJxPOrfLY3SyzJMIoVLu0irwj`, 1,364 bytes, SHA-256 `d5e3842be9004edb3e81704f0055ef09e637938851f0b46627b9de179ebdbc43`.
- D4 archive: Drive `1i6_ho4VfDxv4tK2iu1KJwG3zMb2x8132`, 19,864,515 bytes, SHA-256 `3664b8d8371c321fb2654047e5c90843d6e8a410033c2ad1279f165b124698bd`.
- Optimizer constructed: no. Optimizer steps: 0. CONFIRM scored: no. EVAL-E scored: no. Paid Colab sessions remaining: 0.
