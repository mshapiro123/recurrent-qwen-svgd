# Stage 5 Benchmark Suite - stage5_learned_loop_control_ce8_benchmark_l4_20260623_072057

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `289.55`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0156` (base `37/64`, recurrent `38/64`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `38` / `64`, base `37` / `64`, delta `1`, W/L/T `3/2/59`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `1`, W/L `1/0`, mean margin delta `0.4055046051464699`, mean loops `None`
    - `base_confident_direct_proxy`: n `27`, delta `0`, W/L `0/0`, mean margin delta `-1.4650771372123725`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `11`, delta `2`, W/L `2/0`, mean margin delta `0.22205767763609235`, mean loops `None`
    - `deep_numeric_proxy`: n `8`, delta `-2`, W/L `0/2`, mean margin delta `0.1970598604530096`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0469` (base `24/64`, recurrent `27/64`)
  - paired evidence
    - aggregate `mean`: recurrent `27` / `64`, base `24` / `64`, delta `3`, W/L/T `6/3/55`, p `0.5078125`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `3`, W/L `3/0`, mean margin delta `-0.33251476287841797`, mean loops `1.989057434797287`
    - `base_confident_direct_proxy`: n `8`, delta `-2`, W/L `0/2`, mean margin delta `-0.7075192630290985`, mean loops `1.9339533634483814`
    - `conceptual_reasoning_proxy`: n `20`, delta `2`, W/L `3/1`, mean margin delta `0.001343512535095215`, mean loops `2.089423781633377`
    - `deep_numeric_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `-0.022330712188373913`, mean loops `1.8708137734369799`
### gpqa_lite
- score target `cyclic_label_aggregated`
- score target `content_question_only`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_learned_loop_control_ce8_benchmark_l4_20260623_072057/gpqa_diamond_16.jsonl
