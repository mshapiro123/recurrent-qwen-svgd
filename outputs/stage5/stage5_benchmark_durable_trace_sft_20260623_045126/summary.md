# Stage 5 Benchmark Suite - stage5_benchmark_durable_trace_sft_20260623_045126

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_traced_capability_ladder_sft_durable_20260623_044343/summary.json`
- Checkpoint: `outputs/stage5/stage5_traced_capability_ladder_sft_durable_20260623_044343/phase1/phase1_step_75.pt`
- Benchmarks: `['arc_challenge', 'arc_easy', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `621.15`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0781` (base `24/64`, recurrent `19/64`)
  - paired evidence
    - aggregate `mean`: recurrent `19` / `64`, base `24` / `64`, delta `-5`, W/L/T `4/9/51`, p `0.266845703125`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `0`, W/L `1/1`, mean margin delta `0.1526227617263794`, mean loops `3.108931700388591`
    - `base_confident_direct_proxy`: n `8`, delta `-2`, W/L `0/2`, mean margin delta `-0.7207058668136597`, mean loops `3.104396052658558`
    - `conceptual_reasoning_proxy`: n `20`, delta `-2`, W/L `2/4`, mean margin delta `-0.09527989625930786`, mean loops `3.1144159018993376`
    - `deep_numeric_proxy`: n `11`, delta `-1`, W/L `1/2`, mean margin delta `-0.097278892993927`, mean loops `3.110860555460959`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0156` (base `37/64`, recurrent `36/64`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `36` / `64`, base `37` / `64`, delta `-1`, W/L/T `1/2/61`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `0/0`, mean margin delta `0.7119275044511866`, mean loops `None`
    - `base_confident_direct_proxy`: n `27`, delta `0`, W/L `0/0`, mean margin delta `-2.137688462294776`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `11`, delta `0`, W/L `1/1`, mean margin delta `0.37799064662646165`, mean loops `None`
    - `deep_numeric_proxy`: n `8`, delta `-1`, W/L `0/1`, mean margin delta `0.41825737121204537`, mean loops `None`
### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.1094` (base `41/64`, recurrent `34/64`)
  - paired evidence
    - aggregate `mean`: recurrent `34` / `64`, base `41` / `64`, delta `-7`, W/L/T `3/10/51`, p `0.09228515625`
  - routing buckets
    - `ambiguous_proxy`: n `21`, delta `-5`, W/L `1/6`, mean margin delta `-0.05421308676401774`, mean loops `3.103098818234035`
    - `base_confident_direct_proxy`: n `23`, delta `-1`, W/L `0/1`, mean margin delta `-1.0605169586513354`, mean loops `3.100679999849071`
    - `conceptual_reasoning_proxy`: n `10`, delta `0`, W/L `1/1`, mean margin delta `-0.3625349164009094`, mean loops `3.1110164165496825`
    - `deep_numeric_proxy`: n `10`, delta `-1`, W/L `1/2`, mean margin delta `-0.6140902400016784`, mean loops `3.114169031381607`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `52/64`, recurrent `52/64`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `52` / `64`, base `52` / `64`, delta `0`, W/L/T `0/0/64`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `15`, delta `0`, W/L `0/0`, mean margin delta `0.4179199249173204`, mean loops `None`
    - `base_confident_direct_proxy`: n `40`, delta `0`, W/L `0/0`, mean margin delta `-2.881833683872828`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.3229322101920843`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.17191325966268778`, mean loops `None`
### gpqa_lite
- score target `content_question_only`
- score target `cyclic_label_aggregated`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_benchmark_durable_trace_sft_20260623_045126/gpqa_diamond_16.jsonl
