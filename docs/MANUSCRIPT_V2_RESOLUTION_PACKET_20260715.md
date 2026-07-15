# Manuscript v2 Resolution Packet

**Status:** Evidence work complete; authoritative v2 manuscript file still required

**Date:** 2026-07-15

**Purpose:** Supply artifact-backed answers, figure files, citation checks, and Phase G-alpha margin inputs for strategy's manuscript v2.

## 1. Authority-transfer blocker

The handoff identifies `docs/PAPER_ONE_v2_20260715.md` as the authoritative draft and says it was delivered alongside the handoff. That file is not present in the repository, the attached handoff directory, or the existing project Drive folder. The referenced `paper/04_architecture_section.md` is also absent.

Therefore this pass does **not** manufacture a substitute v2 or claim that its inline markers have been removed. It resolves the underlying evidence, corrects the known numerical error in the retained v1 and closeout record, extends the claim ledger, and creates the four requested figures. Once strategy supplies the actual v2 file, integration is a bounded editorial pass.

## 2. W3 calibration dispute

**Artifact:** `outputs/stage5/stage5_inverse_rendered_width_gate_20260714/summary.json`

The exact W3 calibration result was **276/384 (71.875%)**, against a preregistered pooled gate of **288/384 (75%)**. Per-depth results were:

| Depth | Correct | Total | Required | Result |
|---:|---:|---:|---:|:---|
| 1 | 95 | 96 | 58 | Pass |
| 2 | 82 | 96 | 58 | Pass |
| 3 | 61 | 96 | 58 | Pass |
| 4 | 38 | 96 | 58 | Fail |

The v1 manuscript and `docs/PART1_DETERMINISTIC_PROGRAM_CLOSEOUT_20260715.md` were wrong where they reported `288/384` as achieved. `docs/STAGE5_COMPLETE_HANDOFF_20260715.md` was correct. The retained v1 and closeout table are corrected in this change.

Required v2 sentence:

> W3 achieved 276/384 on calibration, below the preregistered 288/384 pooled gate, and also failed the depth-4 gate at 38/96; the earlier v1 manuscript and closeout table incorrectly reported the gate value as the achieved count.

W4's before-column must consequently use `276/384`, not `288/384`.

## 3. Support-depth frontier and seed resolution

The canonical metric is the interpolated first crossing below the registered same-reader accuracy bar of `0.71`.

| Trained support | Canonical frontier | Primary artifact |
|---:|---:|:---|
| 4 | 5.745806 | `outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json` |
| 6 | 9.005455 | `outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json` |
| 8 | 11.612000 | `outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json` |

The slower support-6 seed had a pre-dose frontier of `7.771765`. After the final fixed 2,000-step dose, it reached **10.000000**. It therefore reached the registered `8.0-10.0` target band exactly at its upper edge. The support-6 result is real but path-dependent: raw seed outcomes spanned `6.951111-9.005455`, while the two dosed seed outcomes reached `9.504` and `10.000`.

Claim boundary: the monotonic central trend supports a bounded cross-support frontier law on this frozen synthetic family. It is not a seed-invariant scaling law, and dose is part of the result.

## 4. Trainable-parameter accounting and architecture constants

### 4.1 Verified constants

| Item | Value | Receipt |
|:---|:---|:---|
| Backbone | Qwen2.5-0.5B-Instruct | training config and official model config |
| Hidden width | 896 | official model config |
| Intermediate width | 4,864 | official model config |
| Layers | 24 | official model config |
| Split indices | `a=6`, `b=18` | `anneal_to_outcome_train_config.yaml` |
| Recurrent layers | 12, layers 6-17 | split arithmetic |
| Query heads / KV heads | 14 / 2 | official model config |
| Q/K/V projection bias | Yes | exact recurrent-block parameter arithmetic and Qwen2 implementation |
| Tied input/output embeddings | Yes | official model config |
| Production tail damper | Retired / inactive | training forward passes `path=None`, `strength=0.0` |

### 4.2 Parameter reconciliation

The historical training summary reports:

| Optimizer-marked group | Parameters |
|:---|---:|
| Twelve-layer recurrent block | 178,948,608 |
| Bridge module | 3,214,849 |
| Halting | 0 |
| Re-entry adapter | 0 |
| Latent module | 0 |
| **Reported trainable total** | **182,163,457** |

The bridge total requires a compatibility distinction:

| Bridge component | Parameters | Forward-active in split mode? |
|:---|---:|:---:|
| Prelude LayerNorm | 1,792 | Yes |
| Prelude projection, 896 x 896, no bias | 802,816 | Yes |
| State projection, 896 x 896 plus bias | 803,712 | Yes |
| Scalar identity-biased gate | 1 | Yes |
| Legacy concatenation projection, 1792 -> 896 plus bias | 1,606,528 | **No** |
| **Bridge optimizer-marked total** | **3,214,849** |  |
| **Forward-active bridge total** | **1,608,321** |  |

`convert_to_split_projection()` freezes the legacy concatenation projection, but the later training setup marks all bridge parameters trainable again. Split-mode forward execution bypasses the legacy projection, so it receives no functional gradient while remaining counted by the historical trainable-parameter summary.

The correct manuscript treatment is:

- Preserve `182,163,457` as the historically reported optimizer-marked count.
- Report `180,556,929` as the forward-active trained parameter count: `178,948,608 + 1,608,321`.
- Explain that the `1,606,528` difference is an inactive compatibility projection, not a halting head, loop embedding, normalization adapter, or latent module.
- Do not call this parameter-efficient fine-tuning.

## 5. Reference verification

| Work | Verified reference | Claim-fit note |
|:---|:---|:---|
| GRAM | Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, Sungjin Ahn. *Generative Recursive Reasoning*. arXiv:2605.19376 (2026). | Supports stochastic recursive latent trajectories, amortized variational inference, and multiple sampled trajectories. Early unguided noise/SVGD experiments do not test this method. |
| Recurrent-depth generalization | Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao. *Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers*. arXiv:2604.07822 (2026). | Supports implicit composition, depth extrapolation under increased recurrence, and the overthinking limitation. |
| Compressed CoT | Jeffrey Cheng, Benjamin Van Durme. *Compressed Chain of Thought: Efficient Reasoning Through Dense Representations*. arXiv:2412.13171 (2024). | Supports variable-length continuous contemplation tokens produced by compressing explicit chains. It is not a recurrent-depth retrofit. |
| Retrofitted recurrence | Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum. *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*. arXiv:2511.07384 (2025). | Direct precedent for converting pretrained non-recurrent LMs and using a recurrence curriculum. Do not describe it as the same bridge or intermediate-state objective used here. |
| CODI | Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, Yulan He. *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation*. arXiv:2502.21074 (2025). | Supports explicit-to-implicit CoT self-distillation and hidden-state alignment. It does not establish recurrent depth or stochastic trajectory coverage. |
| Continuous-thought superposition | Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, Yuandong Tian. *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*. arXiv:2505.12514 (2025). | Establishes a graph-reachability construction and supporting experiments in which continuous thoughts encode multiple search frontiers. Do not generalize it to arbitrary pretrained-model branching or exact multi-solution coverage. |

The exact v2 Section 2.3 wording cannot be checked for overstatement until the missing v2 file is supplied. The table above gives the allowed claim boundary for that pass.

## 6. Ledger additions and artifact map

The claim ledger now includes:

1. `cross_support_frontier_law`, with support-4/6/8 central points, raw support-6 seed range, final slow-seed result, and the N24 rung.
2. `natural_surface_tail_inversion`, with aggregate active accuracy and deepest-tail minima at steps 2,000, 4,000, and 6,000.

The Phase A receipt remains the evidence for same-reader evaluation and identical row IDs. The root-level `manuscript_v2_artifact_map` records all four figures and the primary receipts. Its `canonical_manuscript` pointer intentionally remains on v1 until the actual v2 file exists.

## 7. Natural-surface tail inversion and keeper rationale

| Step | Relay active | Pointer active | Relay diagonal | Pointer diagonal | Min d11-12 across surfaces |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 8,998/9,984 (90.12%) | 8,639/9,984 (86.53%) | 1,296/1,536 | 1,222/1,536 | **54.69%** |
| 4,000 | 9,042/9,984 (90.56%) | 8,508/9,984 (85.22%) | 1,320/1,536 | 1,173/1,536 | 38.28% |
| 6,000 | 9,413/9,984 (94.28%) | 8,958/9,984 (89.72%) | 1,321/1,536 | 1,213/1,536 | 19.53% |

Later training improved aggregate active accuracy but concentrated the loss in the deepest tail. Step 2,000 is the keeper because it maximizes the worst depth-11/12 result across relay and pointer surfaces and is the checkpoint that passed the later verbal branching-substrate screen. It is a maximin tail-preservation choice, not the aggregate-accuracy winner.

## 8. Figures

| Figure | File | Caption boundary |
|---:|:---|:---|
| 1 | `docs/figures/figure1_macro_architecture.svg` | Macro Prelude/Recurrent Block/Coda split, one-loop identity route, and bridge-only re-entry. |
| 2 | `docs/figures/figure2_split_bridge.svg` | Tensor-detail split bridge and forward-active parameter arithmetic. |
| 3 | `docs/figures/figure3_frontier_vs_support.svg` | Canonical support-4/6/8 frontier with raw and dosed support-6 seed outcomes. |
| 4 | `docs/figures/figure4_phase_a_depth_profile.svg` | Existing Phase A depth profile copied from the registered receipt without alteration. |

All four SVGs were rendered to PNG locally for visual QA. The temporary QA rasters are not part of the manuscript package.

## 9. Phase G-alpha powered-margin inputs

Full machine-readable and Markdown receipts:

- `docs/PHASE_G_ALPHA_MARGIN_LOCK_INPUTS_20260715.json`
- `docs/PHASE_G_ALPHA_MARGIN_LOCK_INPUTS_20260715.md`
- Reproducer: `eval/build_manuscript_v2_receipts.py`

Top line on the 512 frozen verbal-keeper rows:

- Valid: `389/512 = 75.98%`.
- Mean reachable-set size: `3.709`.
- Mean score-distribution entropy: `0.1432` nats, or `0.0478` normalized by `log(20)`.
- Mean top-1 softmax probability: `0.9471`.
- Across-row modal prediction rate: `0.0703`.

Validity by depth and stratum:

| Depth | Set-size stratum | Valid | Rows | Validity |
|---:|:---|---:|---:|---:|
| 1 | 2 | 127 | 128 | 99.22% |
| 2 | 2 | 41 | 64 | 64.06% |
| 2 | 3-4 | 54 | 64 | 84.38% |
| 3 | 2 | 19 | 43 | 44.19% |
| 3 | 3-4 | 35 | 43 | 81.40% |
| 3 | 5-8 | 33 | 42 | 78.57% |
| 4 | 2 | 11 | 32 | 34.38% |
| 4 | 3-4 | 15 | 32 | 46.88% |
| 4 | 5-8 | 24 | 32 | 75.00% |
| 4 | 9-16 | 30 | 32 | 93.75% |

This cross-tab reveals an important design constraint: at equal depth, larger reachable sets are often easier because more outputs count as valid. Power calculations and paired comparisons must stratify jointly by depth and reachable-set size. A pooled margin alone would be vulnerable to composition effects.

The entropy-matching comparator should target the deterministic keeper's per-row score entropy, not its across-row symbol distribution. The model is confident within each row while using many symbols across the dataset.

## 10. Remaining bounded work after v2 arrives

1. Place the supplied strategy-authored file at `docs/PAPER_ONE_v2_20260715.md`.
2. Replace each inline marker using this packet and link the associated receipt.
3. Verify the exact Section 2.3 language against the reference claim boundaries above.
4. Insert the four figure references and captions from the strategy draft/module.
5. Change `canonical_manuscript` to v2 and set `manuscript_v2_status` to `resolved`.
6. Run `rg -n "\[RESOLVE-" docs/PAPER_ONE_v2_20260715.md` and require zero matches.
7. Run the ledger and receipt tests, then update the project Drive folder.

No GPU work, retraining, or G-alpha launch is authorized by this packet.
