# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260627_152542

- Status: `completed_with_failures`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_131940_readout/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent max loops: `4`
- Forced loop count: `None`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1174.46`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0156` (base `74/128`, recurrent `72/128`)
  - paired evidence
    - aggregate `mean`: recurrent `72` / `128`, base `74` / `128`, delta `-2`, W/L/T `4/6/118`, p `0.75390625`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `-1`, W/L `3/4`, mean margin delta `0.018607684859523067`, mean loops `1.0916592318702627`
    - `base_confident_direct_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.030349657140873575`, mean loops `1.0651906783516343`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.05442984104156494`, mean loops `1.1405946999788283`
    - `deep_numeric_proxy`: n `17`, delta `0`, W/L `1/1`, mean margin delta `0.027810124789967257`, mean loops `1.207587587482789`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0312` (base `96/128`, recurrent `100/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `100` / `128`, base `96` / `128`, delta `4`, W/L/T `5/1/122`, p `0.21875`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `2`, W/L `3/1`, mean margin delta `-0.04322214515081474`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `0.16124333332535004`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `2`, W/L `2/0`, mean margin delta `0.11166633311659098`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.11251524887565109`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0234` (base `43/128`, recurrent `46/128`)
  - paired evidence
    - aggregate `mean`: recurrent `46` / `128`, base `43` / `128`, delta `3`, W/L/T `4/1/123`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `-1`, W/L `0/1`, mean margin delta `-0.16399919781191596`, mean loops `1.134631022810936`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.13582539558410645`, mean loops `1.1166578802195462`
    - `conceptual_reasoning_proxy`: n `39`, delta `3`, W/L `3/0`, mean margin delta `0.037788498095977`, mean loops `1.20258636199511`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.06876718103885651`, mean loops `1.1438444167375565`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0156` (base `68/128`, recurrent `66/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `66` / `128`, base `68` / `128`, delta `-2`, W/L/T `1/3/124`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `0`, W/L `1/1`, mean margin delta `-0.028714101985385713`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-0.2616586289824108`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `-0.05673535525798798`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-1`, W/L `0/1`, mean margin delta `-0.013505736188519564`, mean loops `None`
### gpqa_lite
- score target `content_question_only`
- score target `cyclic_label_aggregated`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_debiased_benchmark_suite_20260627_152542/gpqa_diamond_16.jsonl

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `3` (base `43/128`, recurrent `46/128`)
