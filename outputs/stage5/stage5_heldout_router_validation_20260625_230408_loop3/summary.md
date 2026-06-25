# Stage 5 Benchmark Suite - stage5_heldout_router_validation_20260625_230408_loop3

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'open_hard_arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `3`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1092.85`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-9`, accuracy delta `-0.0703` (base `71/128`, recurrent `62/128`)
  - paired evidence
    - aggregate `mean`: recurrent `62` / `128`, base `71` / `128`, delta `-9`, W/L/T `6/15/107`, p `0.0783538818359375`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `0`, W/L `3/3`, mean margin delta `-0.2501328647136688`, mean loops `1.1489684653282166`
    - `base_confident_direct_proxy`: n `42`, delta `-9`, W/L `0/9`, mean margin delta `-1.9378377099831898`, mean loops `1.132487420051817`
    - `conceptual_reasoning_proxy`: n `24`, delta `1`, W/L `2/1`, mean margin delta `-0.023907825350761414`, mean loops `1.2210755720734596`
    - `deep_numeric_proxy`: n `12`, delta `-1`, W/L `1/2`, mean margin delta `0.06407638390858968`, mean loops `1.252320056160291`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-6`, accuracy delta `-0.0469` (base `103/128`, recurrent `97/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `97` / `128`, base `103` / `128`, delta `-6`, W/L/T `0/6/122`, p `0.03125`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `-5`, W/L `0/5`, mean margin delta `0.3287500062957406`, mean loops `None`
    - `base_confident_direct_proxy`: n `86`, delta `0`, W/L `0/0`, mean margin delta `-3.5937500080292732`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `13`, delta `-1`, W/L `0/1`, mean margin delta `0.5048076725349977`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.2500000139698386`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0698` (base `11/43`, recurrent `8/43`)
  - paired evidence
    - aggregate `mean`: recurrent `8` / `43`, base `11` / `43`, delta `-3`, W/L/T `4/7/32`, p `0.548828125`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `4/3`, mean margin delta `-0.1167108735867909`, mean loops `1.14441137441567`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.26492544015248615`, mean loops `1.3369924773772557`
    - `deep_numeric_proxy`: n `9`, delta `-3`, W/L `0/3`, mean margin delta `-0.14182187451256645`, mean loops `1.2178230384985607`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0465` (base `23/43`, recurrent `21/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `21` / `43`, base `23` / `43`, delta `-2`, W/L/T `1/3/39`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `1/2`, mean margin delta `0.46180557667846894`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-2.126302093228636`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.43750001505638164`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `0`, W/L `0/0`, mean margin delta `0.6227678474304932`, mean loops `None`
### open_hard_arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-6`, accuracy delta `-0.0469` (base `39/128`, recurrent `33/128`)
  - paired evidence
    - aggregate `mean`: recurrent `33` / `128`, base `39` / `128`, delta `-6`, W/L/T `10/16/102`, p `0.32693958282470703`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `-1`, W/L `5/6`, mean margin delta `-0.06868764292448759`, mean loops `1.1505627655424178`
    - `base_confident_direct_proxy`: n `12`, delta `-5`, W/L `0/5`, mean margin delta `-1.6406601170698802`, mean loops `1.1765334034959476`
    - `conceptual_reasoning_proxy`: n `32`, delta `1`, W/L `3/2`, mean margin delta `0.002322383224964142`, mean loops `1.2704503564164042`
    - `deep_numeric_proxy`: n `20`, delta `-1`, W/L `2/3`, mean margin delta `-0.10888477265834809`, mean loops `1.3554657042026519`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0156` (base `75/128`, recurrent `73/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `73` / `128`, base `75` / `128`, delta `-2`, W/L/T `3/5/120`, p `0.7265625`
  - routing buckets
    - `ambiguous_proxy`: n `40`, delta `0`, W/L `2/2`, mean margin delta `0.8515625031432137`, mean loops `None`
    - `base_confident_direct_proxy`: n `56`, delta `0`, W/L `0/0`, mean margin delta `-2.9070870409840217`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.900000013015233`, mean loops `None`
    - `deep_numeric_proxy`: n `12`, delta `-3`, W/L `0/3`, mean margin delta `0.1770833389212688`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-3` (base `11/43`, recurrent `8/43`)
- `open_hard_arc_challenge` `content_question_only`: delta `-6` (base `39/128`, recurrent `33/128`)
