# Stage 5 Benchmark Suite - stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft_benchmark_suite

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_challenge', 'arc_easy', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `274.77`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-10`, accuracy delta `-0.0781` (base `72/128`, recurrent `62/128`)
  - paired evidence
    - aggregate `mean`: recurrent `62` / `128`, base `72` / `128`, delta `-10`, W/L/T `18/28/82`, p `0.18392482137699062`
  - routing buckets
    - `ambiguous_proxy`: n `37`, delta `8`, W/L `14/6`, mean margin delta `0.8371171054707186`, mean loops `3.068859427361875`
    - `base_confident_direct_proxy`: n `51`, delta `-16`, W/L `0/16`, mean margin delta `-2.2088623626668955`, mean loops `3.076501695548787`
    - `conceptual_reasoning_proxy`: n `26`, delta `-2`, W/L `2/4`, mean margin delta `0.4900937080383301`, mean loops `3.074487798489057`
    - `deep_numeric_proxy`: n `14`, delta `0`, W/L `2/2`, mean margin delta `0.49501015565225054`, mean loops `3.070752311320532`
### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-15`, accuracy delta `-0.1172` (base `86/128`, recurrent `71/128`)
  - paired evidence
    - aggregate `mean`: recurrent `71` / `128`, base `86` / `128`, delta `-15`, W/L/T `7/22/99`, p `0.008130058646202087`
  - routing buckets
    - `ambiguous_proxy`: n `33`, delta `2`, W/L `4/2`, mean margin delta `0.6317896867791811`, mean loops `3.0622890663869455`
    - `base_confident_direct_proxy`: n `74`, delta `-18`, W/L `0/18`, mean margin delta `-3.0852815529608444`, mean loops `3.064758643427411`
    - `conceptual_reasoning_proxy`: n `11`, delta `2`, W/L `2/0`, mean margin delta `0.41281244971535425`, mean loops `3.077426558191126`
    - `deep_numeric_proxy`: n `10`, delta `-1`, W/L `1/2`, mean margin delta `0.7227189652621746`, mean loops `3.0734275758266447`
### gpqa_lite
- score target `label`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft_benchmark_suite/gpqa_diamond_16.jsonl
