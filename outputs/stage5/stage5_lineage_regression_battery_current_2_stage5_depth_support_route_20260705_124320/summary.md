# Stage 5 Benchmark Suite - stage5_lineage_regression_battery_current_2_stage5_depth_support_route_20260705_124320

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_depth_support_route_20260705_124320/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `1`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `7838.91`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0010` (base `3115/5197`, recurrent `3110/5197`)
  - paired evidence
    - aggregate `mean`: recurrent `3110` / `5197`, base `3115` / `5197`, delta `-5`, W/L/T `21/26/5150`, p `0.5600646295802107`
  - routing buckets
    - `ambiguous_proxy`: n `2211`, delta `-7`, W/L `12/19`, mean margin delta `0.0009827982753933095`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `1519`, delta `0`, W/L `0/0`, mean margin delta `-0.0017006855122033823`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `864`, delta `0`, W/L `4/4`, mean margin delta `0.0018870350042427028`, mean loops `1.0`
    - `deep_numeric_proxy`: n `603`, delta `2`, W/L `5/3`, mean margin delta `-9.228044481419805e-05`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-5`, accuracy delta `-0.0010` (base `3976/5197`, recurrent `3971/5197`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `3971` / `5197`, base `3976` / `5197`, delta `-5`, W/L/T `21/26/5150`, p `0.5600646295802107`
  - routing buckets
    - `ambiguous_proxy`: n `1131`, delta `-6`, W/L `11/17`, mean margin delta `0.0017978198048220872`, mean loops `None`
    - `base_confident_direct_proxy`: n `3246`, delta `0`, W/L `0/0`, mean margin delta `0.001074077495659313`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `453`, delta `2`, W/L `7/5`, mean margin delta `0.0018625813908977784`, mean loops `None`
    - `deep_numeric_proxy`: n `367`, delta `-1`, W/L `3/4`, mean margin delta `0.0016462327681522444`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `886/2590`, recurrent `886/2590`)
  - paired evidence
    - aggregate `mean`: recurrent `886` / `2590`, base `886` / `2590`, delta `0`, W/L/T `9/9/2572`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `1216`, delta `1`, W/L `5/4`, mean margin delta `0.0023376799041503354`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `230`, delta `0`, W/L `0/0`, mean margin delta `-0.0019678546682648035`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `631`, delta `0`, W/L `2/2`, mean margin delta `-0.0009930263702918535`, mean loops `1.0`
    - `deep_numeric_proxy`: n `513`, delta `-1`, W/L `2/3`, mean margin delta `-0.001900279208233482`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0008` (base `1522/2590`, recurrent `1524/2590`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `1524` / `2590`, base `1522` / `2590`, delta `2`, W/L/T `12/10/2568`, p `0.8318119049072266`
  - routing buckets
    - `ambiguous_proxy`: n `798`, delta `-1`, W/L `4/5`, mean margin delta `0.003198099529577627`, mean loops `None`
    - `base_confident_direct_proxy`: n `996`, delta `0`, W/L `0/0`, mean margin delta `-0.0011420703234513807`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `417`, delta `1`, W/L `4/3`, mean margin delta `-0.00038719299384470994`, mean loops `None`
    - `deep_numeric_proxy`: n `379`, delta `2`, W/L `4/2`, mean margin delta `0.0032157028374887133`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `0` (base `886/2590`, recurrent `886/2590`)
