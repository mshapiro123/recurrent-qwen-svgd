# Paper Two Bicameral Stage-0 Result Handoff

Date: 2026-08-23. Coding-agent result handoff to strategy. Governing charter: `STRATEGY_BICAMERAL_2BS_CHARTER_DRAFT_20260823.md`, Drive `1xeAJYfq2lIvIm76nszshH4xmaeAkiDl2`, SHA-256 `51083fa2917cf24c73fd9bd0b7e4673fa5536507f6283f635dbed3ddcc0272ac`. Interim ruling: Drive `1xs1vIOWW_duv4ISKugKIJj014ijNglkx`, SHA-256 `2e8ea1a14a6e11c6665dbb2b6f13d861400a41311a30a5802e935649fe4156bf`.

## 1. Executive result

Stage-0 completed without training, optimizer construction, or sealed-partition contact. The geometry results replicate across both seeds and support two registered predictions: k=2 clustering is strong, while the Hadamard diagonal bank can express only about one tenth of the measured correction field. However, two independent registered checks prevent Step-1 from launching as currently written:

1. The cross-fitted residual correlation remains about 0.0154 after the nuisance-deflation loop reaches its rank-8 cap. This is more than fifteen times the registered escalation threshold of 0.001. The receipt therefore returns `RHO_ESCALATE_AT_RANK_CAP` on both seeds. Step-2 sizing is not valid under the current power model.
2. On the exact pinned Qwen/A100 path, batch-concatenating two identical branches is not bit-exact to evaluating the base batch once. The registered T1 hard gate fails before any optimizer step. The conflict is between two locked requirements: batch-concat execution and exact all-gates-zero identity.

The architecture code is otherwise complete and tested. The forward-only cache path is inexpensive and fits well inside the eight-hour planning cap, but Step-1 remains unschedulable until strategy resolves the identity conflict and supplies fixed operating values for the frozen conditioning gates.

## 2. Work completed

- Ported the byte-locked reference into `models/bicameral.py` as `BicameralTaskInferenceGraph` for Qwen2.5-0.5B layers 6 through 17.
- Preserved the residual-spectral combiner and epsilon-inside-square-root repairs.
- Implemented WHT128 over seven blocks, f=0.8 complementary masks, E=8/top-4 occupancy-routed banks, the 256-parameter mu/delta combiner, RMS-cap telemetry, branch-state caching, and closed-form combiner fitting.
- Enforced a 2,308-parameter total inventory and a 256-parameter combiner-only Step-1 trainable set.
- Added the registered k=2/k=3 clustering, initializers, rho_reach, common-mode, and R-S0-A cross-fitted residual-correlation analysis.
- Added CI for T1 identity, T2 cold-start gradients, closed-form fit, parameter inventory, frozen Step-1 trainable set, and the requirement that operating gates cite a measurement receipt.
- Ran 80 focused tests successfully on CPU.
- Ran one forward-only cost and identity probe on the exact pinned runtime, then terminated the A100. No paid Colab sessions remain.

Code state: branch `codex/bicameral-stage0`; implementation commit `b6865ae4`.

## 3. Geometry results

### Clustering

| Seed | k=2 silhouette | k=2 sizes | k=3 silhouette | k=3 sizes |
|---|---:|---:|---:|---:|
| 0 | 0.7643 | 38 / 218 | 0.7325 | 3 / 37 / 216 |
| 1 | 0.7662 | 38 / 218 | 0.7650 | 14 / 24 / 218 |

P-6 is confirmed. The k=2 split is replicated, but it is predominantly task-family-associated: the large cluster contains 216 GSM8K rows, while the small cluster is mainly ARC-Challenge. The E=3 trigger remains negative because its small clusters are unstable across seeds.

### Reachability

| Seed | Small cluster rho_reach | Large cluster rho_reach |
|---|---:|---:|
| 0 | 0.0847 | 0.1228 |
| 1 | 0.0853 | 0.1222 |

P-5 is confirmed more strongly than predicted. Per-band state rescaling reaches only 8.5% to 12.3% of correction energy. The structured bank is therefore secondary shaping, not the main correction mechanism; the LoRA-family prediction remains favored for Step-2 if that phase becomes feasible.

### Common mode and residual correlation

The global common mode explains 71.31% and 71.38% of correction energy. Under R-S0-A, MP spikes persist in 100% of splits at every nuisance rank from 1 through 8.

| Seed | Terminal rank | Pooled rho_res | SD | Registered result |
|---|---:|---:|---:|---|
| 0 | 8 | 0.015329 | 0.006857 | `RHO_ESCALATE_AT_RANK_CAP` |
| 1 | 8 | 0.015408 | 0.006789 | `RHO_ESCALATE_AT_RANK_CAP` |

The large cluster alone remains near 0.0100; the small cluster is near 0.0461 and low-power. Both are far above 0.001. Amendment A1's rank-m generalization is directionally correct but does not discharge the escalation because the measured rank reaches the cap with substantial residual structure intact.

## 4. Runtime and identity result

Pinned runtime was verified exactly: NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu128, CUDA 12.8. Model: `Qwen/Qwen2.5-0.5B-Instruct`, config commit `7ae557604adf67be50417f59c2c2f167def9a775`.

At batch 8 and sequence length 256, middle branch caching took 27.227 ms per batch, or 3.403 ms per row. Peak allocated GPU memory was 1.173 GiB. The closed-form 256-row combiner fit took 2.55 ms. Projected cache-only costs are 0.057 minutes per 1,000 rows, 0.026 minutes for the 461-row slice, and 0.116 minutes for the 2,048-row panel. The exact two-seed cache formula is:

`2 * 0.0034034 * (training_rows + 461 + 2048) / 3600` GPU-hours.

The training-row count is not bound in the charter, so no single total is asserted. Using the charter's inherited 73-minute/four-cell evaluator measurement, the planning envelope remains approximately 4.9 A100-hours for two seeds before contingency, below the eight-hour cap. This is a planning estimate, not launch authority.

T1 fails on the real substrate:

- Base logits versus zero-gate batch-concat graph: 8,969,555 of 9,723,904 values differ; maximum absolute BF16 difference 0.375.
- Single-batch middle versus the first half of an identical doubled-batch middle: 49,192 of 57,344 hidden values differ; maximum absolute difference 0.25.

The second comparison localizes the failure to batch-dependent frozen-middle numerics, before recombination or the coda. This is not a WHT round-trip defect.

## 5. Interpretation

The positive structural result is real but narrower than the original story. There are two stable correction families in these artifacts, yet they align strongly with benchmark family and are not evidence for abstract reasoning modes. The Hadamard bank captures only a small fraction of their correction directions, so it cannot be the primary writer.

The residual-correlation result is the main scientific negative. Even an eight-direction nuisance model does not restore the near-independent-row regime assumed by the registered power calculation. More rows alone cannot be claimed to solve this. The supervision model needs either a better structured-error decomposition or a different estimand before Step-2 can be sized.

The T1 failure is an engineering-contract contradiction, not evidence against Bicameral specialization. The exact reference obtains identity by evaluating branches separately. The charter adds batch concatenation, which changes BF16 kernel numerics on the real transformer. Preserving exact identity therefore requires changing the execution schedule or introducing a new estimator; silently relaxing equality is not acceptable.

## 6. Limitations

- Geometry uses the 256-row preserved arm-6 artifacts; the 38-row cluster gives weak high-dimensional covariance estimates.
- Cluster semantics are confounded with battery identity.
- The cost probe used synthetic token IDs and fixed lengths. It measures the actual graph but not the final training-slice length distribution.
- The cost probe did not score DEV, CONFIRM, or EVAL-E and did not construct an optimizer.
- Step-1's training-slice row count is not bound, and its fixed callosum/bank operating gates are unspecified.
- No claim is made that knowledge, reasoning mode, or task capability is represented by a cluster.

## 7. Questions requiring strategy ruling

1. **T1 versus batch-concat.** Recommended ruling: retain exact equality and replace batch-concat with two sequential calls to the same frozen middle, matching the byte-locked reference. Then rerun T1 and the cost probe. Alternatives are to weaken T1 or add a straight-through identity override; neither is recommended.
2. **Frozen conditioning gates at Step-1.** Combiner-only fitting cannot use cluster masks or bank initializers if `g_A`, `g_B`, `s_A`, and `s_B` remain zero. No receipted fixed values exist. Strategy must either bind values through a no-training amplitude measurement or authorize these gates as trainable, which would change Step-1's estimand.
3. **R-S0-A escalation.** Does strategy want a redesigned nuisance model before Step-1, or should Step-1 proceed strictly as an interface screen while Step-2 remains blocked? Recommendation: allow Step-1 only after T1/gate repair, but prohibit any Step-2 sizing claim until a new residual model is locked.
4. **Training-slice population.** Bind its source, row count, sequence-length distribution, and hashes before the revised cost projection is accepted.
5. **Amendment A1.** Rank-m deflation can be ratified as a general rule, but it does not by itself clear the observed rank-cap escalation.

## 8. Next-step map

1. Strategy rules on the four execution/data questions above and Amendment A1.
2. Coding agent implements only the ratified T1 execution repair and gate contract.
3. Re-run T1/T2 and the pinned cost probe; update the exact total using the bound training manifest.
4. If all preflights pass and the cost remains below eight A100-hours, Step-1 may be scheduled under the existing BIC keys.
5. Step-2 remains unauthorized and correlation-blocked.

## 9. Receipts

- Geometry concise receipt: `docs/receipts/paper2_bicameral_stage0_geometry_20260823.json`, 2,969 B, SHA-256 `6149f8458a34e9dc5d4e636dc1f25d0094ca771968141fb8e0aecb69fb5df24e`.
- Full geometry receipt: 1,006,416 B, SHA-256 `67a38f34d5c8d4c52b5904a9e3104f150bff03bf1eca5dc4190923d71d583dc5` (private artifact package).
- Runtime/cost receipt: `docs/receipts/paper2_bicameral_step1_cost_probe_20260823.json`, 3,176 B, SHA-256 `2ef0f71561aa6cbdad61292073f0bc963e7ca5cd27004be3406725498e87383b`.
- Batch-concat identity diagnostic: `docs/receipts/paper2_bicameral_batch_concat_identity_20260823.json`, 1,450 B, SHA-256 `45e537295545f52b1967d1757a64da5b787dc91ed67cb2e480ff81e2631c3d46`.
- Reference source: Drive `1KR-35tkctPSqfx22rI83D7tmyEN5zf3q`, 13,452 B, SHA-256 `14231e8480177769e0f230724d710f3e47aae52fd791a81f1f11adb25f850cb2`.
- Tests: 80 passed.
- Compute closeout: `colab sessions` returned no active sessions after teardown.
