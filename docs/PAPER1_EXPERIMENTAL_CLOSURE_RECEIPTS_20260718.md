# Paper 1 Experimental Closure Receipts

**Scope:** evidence and claim boundaries only; manuscript prose was not edited.

## Guardrail Battery

- Four checkpoint lineages; ARC Easy `n=5,197`, ARC Challenge `n=2,590`.
- Primary family: eight cyclic-label comparisons; Bonferroni correction.
- Most adverse result: `stage5_lineage_regression_battery_current_4_stage5_n24_support12_rung_20260707_140139` / `arc_easy`, delta `-14`, raw `p=0.038477`, corrected `p=0.307818`.
- Every comparison remained above the locked `-3%` margin.

| Checkpoint | Benchmark | Reader | Delta | Wins | Losses | Raw p | Corrected p |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | arc_easy | cyclic_label_aggregated | +8 | 61 | 53 | 0.512255 | 1.000000 |
| 1 | arc_challenge | cyclic_label_aggregated | +4 | 45 | 41 | 0.746534 | 1.000000 |
| 2 | arc_easy | cyclic_label_aggregated | -5 | 21 | 26 | 0.560065 | 1.000000 |
| 2 | arc_challenge | cyclic_label_aggregated | +2 | 12 | 10 | 0.831812 | 1.000000 |
| 3 | arc_easy | cyclic_label_aggregated | -8 | 16 | 24 | 0.268187 | 1.000000 |
| 3 | arc_challenge | cyclic_label_aggregated | +1 | 13 | 12 | 1.000000 | 1.000000 |
| 4 | arc_easy | cyclic_label_aggregated | -14 | 13 | 27 | 0.038477 | 0.307818 |
| 4 | arc_challenge | cyclic_label_aggregated | -2 | 11 | 13 | 0.838820 | 1.000000 |
| 1 | arc_easy | content_question_only | +30 | 181 | 151 | 0.111341 | descriptive |
| 1 | arc_challenge | content_question_only | +8 | 69 | 61 | 0.539420 | descriptive |
| 2 | arc_easy | content_question_only | -5 | 21 | 26 | 0.560065 | descriptive |
| 2 | arc_challenge | content_question_only | +0 | 9 | 9 | 1.000000 | descriptive |
| 3 | arc_easy | content_question_only | -5 | 18 | 23 | 0.532709 | descriptive |
| 3 | arc_challenge | content_question_only | +1 | 12 | 11 | 1.000000 | descriptive |
| 4 | arc_easy | content_question_only | +3 | 24 | 21 | 0.765992 | descriptive |
| 4 | arc_challenge | content_question_only | +5 | 12 | 7 | 0.359283 | descriptive |

## Bounded PEFT Canary

- Baseline: `0.9375` on `64` arithmetic rows.
- Six interval checks stayed at `0.9531`.
- Identity maximum absolute difference: `0.0`.
- No permutation control was run for this bounded canary.

## Early-Era Telemetry

| Archive | Cells | Expected loops | Halt entropy |
|---|---:|---:|---:|
| extended_fold0_random32_rep05 | 35 | 3.0535 | 1.1808 |
| extended_fold0_within_group_dim8_rep2 | 35 | 3.0659 | 1.1726 |
| extended_fold1_random32_rep05 | 35 | 3.0631 | 1.1746 |
| extended_fold1_within_group_dim8_rep2 | 35 | 3.0694 | 1.1704 |
| recreated_current_random32_rep05 | 70 | 3.0546 | 1.1798 |
| recreated_current_within_group_dim8_rep2 | 70 | 3.0733 | 1.1681 |
| original_stage4_exact_phase1_vs_phase2 | 39 | 2.8102 | 1.3024 |

## Structural Receipts

- Historical bridge gate: `0.0`; delta RMS: `0.0`; weight-gradient RMS: `0.0`.
- Historical R16 optimizer-marked parameters: `7,613,953`.
- Historical R16 forward-active parameters: `6,007,425`.
- Prospective Arm E excludes the bypassed legacy concat tensors.

## Literature Claim Fit

- **Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence** (arXiv:2511.07384): Supports contextualizing pretrained-model recurrence retrofitting. Do not describe its method as LoRA-based or as requiring auxiliary adapters.
- **Hierarchical Reasoning Model** (arXiv:2506.21734): Task-specific hierarchical recurrent reasoning comparator.
- **Tiny Recursive Models** (arXiv:2510.04871): Task-specific recursive reasoning comparator.
- **Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models** (arXiv:2411.12580): Supports a relationship between procedural pretraining knowledge and reasoning; does not justify a strict memorization-versus-procedure dichotomy.

## Do Not Claim

- No broad natural-capability parity from the bounded guardrails.
- No calibrated useful early learned-halting policy from aggregate telemetry.
- No GRAM-style width conclusion from pre-repair stochastic experiments.
- No description of McLeish et al. as a LoRA or auxiliary-adapter recipe.
- No budget-independence or capacity-limit conclusion before Arm E lands.
