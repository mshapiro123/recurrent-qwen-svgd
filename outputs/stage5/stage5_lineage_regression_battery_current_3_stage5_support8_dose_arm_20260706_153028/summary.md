# Stage 5 Benchmark Suite - stage5_lineage_regression_battery_current_3_stage5_support8_dose_arm_20260706_153028

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_support8_dose_arm_20260706_153028/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `1`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `7798.63`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0010` (base `3115/5197`, recurrent `3110/5197`)
  - paired evidence
    - aggregate `mean`: recurrent `3110` / `5197`, base `3115` / `5197`, delta `-5`, W/L/T `18/23/5156`, p `0.5327092552361137`
  - routing buckets
    - `ambiguous_proxy`: n `2211`, delta `-4`, W/L `11/15`, mean margin delta `0.0015701693762809861`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `1519`, delta `0`, W/L `0/0`, mean margin delta `-5.96919401724767e-06`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `864`, delta `1`, W/L `3/2`, mean margin delta `0.0014592857203549808`, mean loops `1.0`
    - `deep_numeric_proxy`: n `603`, delta `-2`, W/L `4/6`, mean margin delta `-0.0022821312223500873`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-8`, accuracy delta `-0.0015` (base `3976/5197`, recurrent `3968/5197`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `3968` / `5197`, base `3976` / `5197`, delta `-8`, W/L/T `16/24/5157`, p `0.26818725105476915`
  - routing buckets
    - `ambiguous_proxy`: n `1131`, delta `-12`, W/L `7/19`, mean margin delta `0.0010094314103113306`, mean loops `None`
    - `base_confident_direct_proxy`: n `3246`, delta `0`, W/L `0/0`, mean margin delta `0.0002884968893616267`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `453`, delta `3`, W/L `6/3`, mean margin delta `0.003104302742406438`, mean loops `None`
    - `deep_numeric_proxy`: n `367`, delta `1`, W/L `3/2`, mean margin delta `0.0031051326000455636`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0004` (base `886/2590`, recurrent `887/2590`)
  - paired evidence
    - aggregate `mean`: recurrent `887` / `2590`, base `886` / `2590`, delta `1`, W/L/T `12/11/2567`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `1216`, delta `-1`, W/L `5/6`, mean margin delta `0.0012464285338003385`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `230`, delta `0`, W/L `0/0`, mean margin delta `-0.004427954878496087`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `631`, delta `2`, W/L `4/2`, mean margin delta `-0.00037389601656071924`, mean loops `1.0`
    - `deep_numeric_proxy`: n `513`, delta `0`, W/L `3/3`, mean margin delta `-0.001306492450409233`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0004` (base `1522/2590`, recurrent `1523/2590`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `1523` / `2590`, base `1522` / `2590`, delta `1`, W/L/T `13/12/2565`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `798`, delta `-2`, W/L `5/7`, mean margin delta `0.0006396192679945738`, mean loops `None`
    - `base_confident_direct_proxy`: n `996`, delta `0`, W/L `0/0`, mean margin delta `0.0014275824281538506`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `417`, delta `1`, W/L `4/3`, mean margin delta `-0.0005620503676221441`, mean loops `None`
    - `deep_numeric_proxy`: n `379`, delta `2`, W/L `4/2`, mean margin delta `0.003160731666684545`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `1` (base `886/2590`, recurrent `887/2590`)
