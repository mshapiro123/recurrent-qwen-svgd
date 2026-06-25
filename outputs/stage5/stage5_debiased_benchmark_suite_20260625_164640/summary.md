# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260625_164640

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260625_154210/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/phase1/phase1_step_75.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1377.27`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0391` (base `86/128`, recurrent `91/128`)
  - paired evidence
    - aggregate `mean`: recurrent `91` / `128`, base `86` / `128`, delta `5`, W/L/T `5/0/123`, p `0.0625`
  - routing buckets
    - `ambiguous_proxy`: n `34`, delta `2`, W/L `2/0`, mean margin delta `0.02478482716662042`, mean loops `1.135963053387754`
    - `base_confident_direct_proxy`: n `73`, delta `0`, W/L `0/0`, mean margin delta `-0.05150832692543938`, mean loops `1.2039905170871785`
    - `conceptual_reasoning_proxy`: n `12`, delta `2`, W/L `2/0`, mean margin delta `0.3361271160344283`, mean loops `1.3422338242332141`
    - `deep_numeric_proxy`: n `9`, delta `1`, W/L `1/0`, mean margin delta `0.44760413136747146`, mean loops `1.331249048312505`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0078` (base `74/128`, recurrent `73/128`)
  - paired evidence
    - aggregate `mean`: recurrent `73` / `128`, base `74` / `128`, delta `-1`, W/L/T `5/6/117`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `0`, W/L `4/4`, mean margin delta `0.07681924325448496`, mean loops `1.0922824254742376`
    - `base_confident_direct_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.008526744472013938`, mean loops `1.0658369977731963`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.030640560388565063`, mean loops `1.142482429742813`
    - `deep_numeric_proxy`: n `17`, delta `0`, W/L `1/1`, mean margin delta `0.011725727249594295`, mean loops `1.2089441856917214`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0156` (base `96/128`, recurrent `98/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `98` / `128`, base `96` / `128`, delta `2`, W/L/T `3/1/124`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `0`, W/L `1/1`, mean margin delta `-0.039064862153359824`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `0.09257494518501885`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `2`, W/L `2/0`, mean margin delta `0.11608092095702886`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.14274879896806347`, mean loops `None`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0078` (base `72/128`, recurrent `71/128`)
  - paired evidence
    - aggregate `mean`: recurrent `71` / `128`, base `72` / `128`, delta `-1`, W/L/T `3/4/121`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `36`, delta `-2`, W/L `1/3`, mean margin delta `0.07932277385973269`, mean loops `1.2274701236574739`
    - `base_confident_direct_proxy`: n `52`, delta `0`, W/L `0/0`, mean margin delta `-0.15529357083141804`, mean loops `1.3138314353731961`
    - `conceptual_reasoning_proxy`: n `26`, delta `1`, W/L `2/1`, mean margin delta `-0.013769341776004205`, mean loops `1.3482622412534861`
    - `deep_numeric_proxy`: n `14`, delta `0`, W/L `0/0`, mean margin delta `0.07049456132309777`, mean loops `1.2520475096645811`
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0234` (base `43/128`, recurrent `46/128`)
  - paired evidence
    - aggregate `mean`: recurrent `46` / `128`, base `43` / `128`, delta `3`, W/L/T `4/1/123`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `-1`, W/L `0/1`, mean margin delta `-0.14279069982725998`, mean loops `1.1351311434616989`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.13977037776600232`, mean loops `1.11724132028493`
    - `conceptual_reasoning_proxy`: n `39`, delta `3`, W/L `3/0`, mean margin delta `0.027162781128516562`, mean loops `1.2032621846749232`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.0780013531446457`, mean loops `1.144555014371872`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0078` (base `68/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `67` / `128`, base `68` / `128`, delta `-1`, W/L/T `2/3/123`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `0`, W/L `1/1`, mean margin delta `-0.010593212642815224`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-0.33304561989422404`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `0`, W/L `1/1`, mean margin delta `-0.007544842213392257`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-1`, W/L `0/1`, mean margin delta `0.016834559628651262`, mean loops `None`
### gpqa_lite
- score target `label`
- score target `content_question_only`
- score target `cyclic_label_aggregated`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_debiased_benchmark_suite_20260625_164640/gpqa_diamond_16.jsonl
