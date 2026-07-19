# Arm E Adapter-Budget Recurrent Arm: Publication Handoff

Date: 2026-07-18
Run: `stage5_adapter_budget_arm_e_20260718`
Final receipt commit: `0c56054f19a2cd523cbf856cb349ac1d93394fd2`
Analysis status: complete; Phase G intentionally excluded

## Technical summary

Arm E closes the missing matched-training comparison between the full-block recurrent Arm A and a parameter-efficient recurrent arm. Arm E trained rank-16 LoRA adapters over all recurrent-block projections plus the repaired split bridge while freezing every pretrained Qwen parameter. It used `6,007,425` forward-active trainable parameters, or `3.33%` of Arm A's `180,556,929` forward-active trained parameters, a `30.1x` reduction.

The result is a depth-dependent crossover:

- Across all 1,792 frozen rows, Arm E and Arm A were numerically tied: `1501/1792` (`83.76%`) versus `1506/1792` (`84.04%`), a difference of `-0.28` percentage points. The paired exact McNemar test was not significant (`p=0.813`, two-sided).
- On the directly trained support, depths 1-8, Arm E was better: `1021/1024` (`99.71%`) versus `1005/1024` (`98.14%`), a `+1.56` point difference. A post-hoc grouped paired test gave `p=0.000855`.
- On near extrapolation, depths 9-11, Arm E remained numerically better: `344/384` (`89.58%`) versus `326/384` (`84.90%`), `+4.69` points (`p=0.0693`, post-hoc, two-sided).
- On far extrapolation, depths 12-14, Arm E was worse: `136/384` (`35.42%`) versus `175/384` (`45.57%`), `-10.16` points. The post-hoc grouped paired test gave `p=0.00394`.

Arm E therefore failed the pre-registered full-profile parity gate even though it passed the pooled margin. The failure was caused by per-depth deficits larger than 8 correct rows at depths 12, 13, and 14. The automated verdict, `deficit` with `tail_concentrated` shape, is correct.

The publication-safe conclusion is:

> The iterative mechanism was installed with approximately 3.3% of the full-block trainable parameter budget and matched the full-block arm in pooled accuracy. It was stronger in aggregate through depth 11, with a small depth-10 dip, but its far-horizon extrapolation degraded at depths 12-14. Parameter-efficient installation succeeded within and just beyond the trained support; full-profile depth generalization remained capacity-sensitive in this matched synthetic experiment.

Do not summarize this as either universal budget independence or general PEFT inferiority. Both would erase the observed crossover.

## 1. Question and pre-registered decision

Arm E asked whether the recurrent mechanism and its depth profile were independent of the training budget when the training recipe was matched to Arm A.

The intended single variable was the trainable set:

| Arm | Trainable recurrent substrate | Forward-active trainable parameters |
|:---|:---|---:|
| A | Full 12-layer recurrent block plus repaired split bridge | `180,556,929` |
| E | Rank-16 LoRA on all recurrent projections plus repaired split bridge | `6,007,425` |

Arm E represents `3.327%` of Arm A's forward-active trained parameter count. The whole Qwen2.5-0.5B backbone still participates in inference; `6.0M` is a trainable-parameter budget, not the total model size or an inference-FLOP reduction.

The locked readings were:

1. **Parity:** pooled Arm E accuracy within 3 points of Arm A and no depth worse by more than `8/128`.
2. **Deficit:** above the Arm C floor but failing the parity profile, with the depth shape reported.
3. **Catastrophic recipe alarm:** pooled accuracy below Arm C's `53.1%`.

Paired comparisons used identical frozen row IDs and exact paired sign/McNemar tests.

## 2. Experimental design

### 2.1 Model and initialization

- Backbone: `Qwen/Qwen2.5-0.5B-Instruct`.
- Architecture: fresh recurrent surgery, split `6,18`, repaired split bridge.
- Initialization: fresh base surgery, not a keeper-lineage continuation.
- Recurrent adaptation: LoRA rank 16, alpha 32, over all recurrent-block projections.
- Bridge: repaired split bridge, fully trainable.
- Frozen set: all pretrained Qwen parameters.
- Trainable set:
  - recurrent-block LoRA: `4,399,104`;
  - bridge: `1,608,321`;
  - total: `6,007,425`.
- Halting, latent, and re-entry-adapter trainable parameters: zero.
- Optimizer: AdamW.
- Training seed: 0.

### 2.2 Curriculum

Arm E followed the reconstructed Arm A stage protocol:

| Stage | Support | Steps | LR | Prelude LR multiplier | Supervision |
|:---|---:|---:|---:|---:|:---|
| `primitive_depth1` | 1 | 500 | `2e-5` | 1 | MCQ option text |
| `chain_depth_le2` | 1-2 | 2,000 | `1e-5` | 10 | per-loop labels |
| `chain_depth_le4` | 1-4 | 4,000 | `1e-5` | 10 | per-loop labels |
| `chain_depth_le8` | 1-8 | 2,000 | `1e-5` | 10 | per-loop labels |
| `chain_depth_le8_dose` | 1-8 | 2,000 | `1e-5` | 10 | per-loop labels |

Total optimizer steps: `10,500`. Batch size and gradient accumulation were both 1; weight decay was zero and gradient norm was capped at 0.5.

The final evaluation used the immutable Phase A set:

- 1,792 rows;
- 128 rows at each depth 1-14;
- same row IDs for Arms A and E;
- same-reader full-symbol scoring;
- forced loop count equal to row depth;
- question-only prompt;
- letter-symbol candidate space `A-P`.

### 2.3 Integrity gates

All integrity gates passed:

- Step-zero one-loop identity: max absolute difference `0.0`, threshold `1e-3`.
- Pretrained-base hash:
  `960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f`
  at both start and end.
- Base capability canary: baseline `60/64`; stage checks remained `59/64` or `60/64`, all `green_continue`.
- Trainable count: exactly `6,007,425` at every stage.
- Final checkpoint:
  `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`.
- Immutable training and evaluation data hashes matched the locked manifest.

## 3. Primary frozen-set result

### 3.1 Pooled result

| Arm | Correct | Total | Accuracy |
|:---|---:|---:|---:|
| A, full block | 1,506 | 1,792 | 84.04% |
| E, R16 plus bridge | 1,501 | 1,792 | 83.76% |
| E minus A | -5 |  | -0.28 pp |

Paired outcomes:

- helped by Arm E: 140;
- hurt by Arm E: 145;
- tied: 1,507;
- exact paired two-sided `p=0.8128`.

The pooled result is indistinguishable. Arm E passed the pooled 3-point margin by a wide margin.

### 3.2 Depth profile

The profile, not the aggregate, determined the registered verdict.

| Depth | Arm A | Arm E | E-A correct | E-A pp | Paired p, two-sided |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 0 | 0.00 | 1.000 |
| 2 | 127 | 128 | +1 | +0.78 | 1.000 |
| 3 | 126 | 127 | +1 | +0.78 | 1.000 |
| 4 | 125 | 128 | +3 | +2.34 | 0.250 |
| 5 | 127 | 128 | +1 | +0.78 | 1.000 |
| 6 | 126 | 127 | +1 | +0.78 | 1.000 |
| 7 | 124 | 128 | +4 | +3.13 | 0.125 |
| 8 | 122 | 127 | +5 | +3.91 | 0.125 |
| 9 | 116 | 123 | +7 | +5.47 | 0.143 |
| 10 | 113 | 110 | -3 | -2.34 | 0.711 |
| 11 | 97 | 111 | +14 | +10.94 | 0.0436 |
| 12 | 87 | 75 | -12 | -9.38 | 0.148 |
| 13 | 57 | 43 | -14 | -10.94 | 0.130 |
| 14 | 31 | 18 | -13 | -10.16 | 0.0660 |

The per-depth p-values are descriptive and unadjusted across 14 depths. Depth 11 is nominally significant but should not be presented as an independently corrected discovery.

The pre-registered parity profile required every Arm E deficit to be no worse than 8 rows. Depths 12, 13, and 14 missed that bound by 12, 14, and 13 rows. Therefore:

- pooled parity condition: passed;
- every-depth parity condition: failed;
- catastrophic floor: passed;
- registered verdict: `deficit`;
- registered shape: `tail_concentrated`.

## 4. Post-hoc localization shows a sharp crossover

The following grouped cuts were computed after seeing the registered profile. They localize the result but do not replace the pre-registered gate.

| Segment | Arm A | Arm E | E-A pp | Helped / hurt | Paired p |
|:---|---:|---:|---:|---:|---:|
| Trained support, d1-8 | 1005/1024 | 1021/1024 | +1.56 | 19 / 3 | 0.000855 |
| Near extrapolation, d9-11 | 326/384 | 344/384 | +4.69 | 53 / 35 | 0.0693 |
| Far extrapolation, d12-14 | 175/384 | 136/384 | -10.16 | 68 / 107 | 0.00394 |
| Previously reported tail, d11-14 | 272/512 | 247/512 | -4.88 | 96 / 121 | 0.103 |
| All depths, d1-14 | 1506/1792 | 1501/1792 | -0.28 | 140 / 145 | 0.813 |

This is a crossover rather than a uniform deficit:

1. Arm E is stronger on the supervised horizon.
2. It remains competitive or stronger for three additional loops.
3. It loses at the farthest three depths.
4. Pooling all depths hides both effects.
5. Pooling depths 11-14 also blurs the boundary because Arm E is unusually strong at depth 11.

The clean mechanistic interpretation is capacity-sensitive extrapolation after successful low-budget installation. The evidence does not distinguish whether the far-tail limitation is caused by LoRA rank, frozen base geometry, bridge capacity, optimization, or their interaction. A rank sweep was explicitly outside scope.

## 5. The extra depth-8 dose was necessary

On the same balanced depth-1-through-8 smoke set:

| Checkpoint | Correct / 128 | Accuracy |
|:---|---:|---:|
| End of initial depth-8 stage | 115/128 | 89.84% |
| Extra dose, step 1,000 | 124/128 | 96.88% |
| Extra dose, step 2,000 | 127/128 | 99.22% |

The final frozen evaluation confirmed `1021/1024` across depths 1-8. This rules out the initial `115/128` smoke result as a terminal adapter-capacity ceiling within trained support. Continued matched training almost eliminated in-support errors. The remaining limitation appears only at far extrapolation.

The dose ledger also shows uneven active-label exposure: at the final depth-8 dose checkpoint, loops 1-8 received `[2000, 1752, 1503, 1253, 1003, 753, 501, 251]` active labels. Arm E nevertheless reached near-perfect trained-support accuracy. The result should not be described as label-dose equalized.

## 6. Evaluation erratum

An initial smoke-evaluation implementation passed an empty `value_prefix`, generating numeric candidates `0-15` for rows whose true symbols were letters `A-P`. This produced artificial `0/128` summaries. The defect was found before final interpretation and fixed in commit:

`b5440d161d340c0bedf271f82fa62da6d076fb20`

The resume run invalidated cached summaries that did not declare the canonical contract and recomputed all available stage-end checkpoints using:

- prediction space: `full_symbols`;
- prompt style: `question_only`;
- value prefix: `letter:`.

Training, checkpoints, immutable data, and canary results were unaffected. Several invalid intermediate-checkpoint summaries are retained in Git history because their transient checkpoints were not available after the interrupted runtime. They are explicitly listed in:

`outputs/stage5/stage5_adapter_budget_arm_e_20260718/evaluation_reader_erratum.json`

Publication rule: cite only Arm E final-symbol summaries with `value_prefix: "letter:"`.

## 7. Manuscript interpretation

### 7.1 Claim supported

The paper can state:

> A rank-16 adapter-plus-bridge arm trained 6.0M forward-active parameters, 3.3% of the full-block arm's trainable budget, and achieved nearly identical pooled accuracy on the frozen 1,792-row depth evaluation (83.76% versus 84.04%; paired p=0.813). The depth profile showed a crossover: the adapter arm was stronger through the trained horizon and near extrapolation, but weaker at depths 12-14. This separates parameter-efficient mechanism installation from far-horizon depth generalization.

An even shorter abstract-safe form is:

> The recurrent mechanism could be installed with 6.0M trainable parameters while preserving pooled performance, but far-horizon extrapolation remained sensitive to recurrent-block training capacity.

### 7.2 Claims not supported

Do not claim:

- full-profile budget independence;
- that 6.0M parameters are the total model size;
- inference-compute or memory equivalence;
- that LoRA is generally inferior to full fine-tuning;
- that rank 16 is optimal;
- natural-language or benchmark generalization from this synthetic family;
- learned-halting success from Arm E, because the halting trainable count was zero;
- statistical significance for every individual depth without multiplicity qualification;
- seed robustness, because Arm E used one locked training seed.

### 7.3 Recommended figure treatment

Add Arm E as the fifth curve in the existing depth-profile figure.

Recommended annotations:

- label Arm A as `full block, 180.6M trainable`;
- label Arm E as `R16 + bridge, 6.0M trainable`;
- shade depths 1-8 as trained support;
- mark the d11-to-d12 crossover;
- include pooled results in the caption, not as a substitute for the curve.

Recommended caption:

> Matched full-block and adapter-budget recurrent arms on 128 frozen rows per depth. Arm E used 6.0M forward-active trainable parameters versus 180.6M for Arm A. Pooled accuracy was nearly identical (83.76% versus 84.04%), but Arm E's aggregate advantage over depths 1-11 reversed at depths 12-14. The registered parity gate therefore returned a tail-concentrated deficit.

## 8. Publication decision

Arm E closes the parameter-efficient question without requiring another training run.

The result strengthens Paper One because it yields a two-budget finding rather than a simple smaller-model replication:

1. **Installation budget:** approximately 6M trainable parameters are sufficient to install the iterative mechanism and achieve near-perfect performance through trained depth 8.
2. **Characterization budget:** full-block training improves the far extrapolation tail.
3. **Aggregate caution:** pooled accuracy alone would incorrectly suggest complete equivalence.
4. **Scientific value:** the crossover measures where adapter capacity begins to bind instead of treating PEFT as globally successful or unsuccessful.

No further rank sweep or Arm E continuation is recommended before submission. A second training seed would improve variance characterization but is not required to interpret the registered matched-row result; list single-seed training as a limitation. If reviewers demand robustness, replicate only the final locked R16 protocol, not a rank sweep.

## 9. Questions for strategy review

1. Should the manuscript headline use `parameter-efficient installation with capacity-sensitive extrapolation`, or the more conservative `trainable-budget depth crossover`?
2. Should the post-hoc d1-8 and d12-14 grouped tests appear in the main text or only in the appendix?
3. Is the single-seed limitation acceptable for submission, given the large paired frozen set and the registered profile gate?
4. Should the depth-profile figure annotate the extra 2,000-step dose stage, or leave curriculum detail to the methods table?
5. Does the abstract need the exact `3.3%` budget ratio, or should that remain in Results to avoid confusing trainable parameters with total model size?

## 10. Canonical artifact map

Primary:

- Final summary:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json`
- Registered paired profile:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/adapter_budget_depth_profile.json`
- Final 1,792-row evaluation:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/eval/final_phase_a_1792/summary.json`
- Row-level Arm E predictions:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/eval/final_phase_a_1792/rows.jsonl`
- Arm A comparator rows:
  `outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/eval/same_reader_final_rows.jsonl`
- Post-hoc grouped analysis:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/adapter_budget_posthoc_segments.json`
- Evaluation erratum:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/evaluation_reader_erratum.json`

Lineage:

- Final checkpoint:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/train/chain_depth_le8_dose/unfrozen_recurrent_step_2000.pt`
- Final checkpoint SHA-256:
  `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`
- Final receipt commit:
  `0c56054f19a2cd523cbf856cb349ac1d93394fd2`
- Evaluator correction commit:
  `b5440d161d340c0bedf271f82fa62da6d076fb20`

Reproducibility:

```bash
python eval/analyze_adapter_budget_segments.py \
  --arm_a_rows outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/eval/same_reader_final_rows.jsonl \
  --arm_e_rows outputs/stage5/stage5_adapter_budget_arm_e_20260718/eval/final_phase_a_1792/rows.jsonl \
  --output_json outputs/stage5/stage5_adapter_budget_arm_e_20260718/adapter_budget_posthoc_segments.json
```

## Bottom line

Arm E is a successful parameter-efficient mechanism-installation result and a failed full-profile parity result. Both statements are necessary.

The 6M-parameter arm reproduces and slightly improves the installed recurrent computation through the trained horizon, and it remains strong through depth 11. Its weakness emerges only in far extrapolation. The correct publication story is not that adapter training matches full-block training everywhere, nor that adapters fail. It is that mechanism installation and depth extrapolation have different capacity requirements.
