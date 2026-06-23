# Stage 5 Benchmark Suite - stage5_arc_mix_survival_depth_full_balanced_20260623

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_mix_survival_depth_after_aggregate_gate_20260623_depth_routing_probe/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_survival_depth_after_aggregate_gate_20260623_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `3288.35`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `21`, accuracy delta `0.0410` (base `298/512`, recurrent `319/512`)
  - paired evidence
    - aggregate `mean`: recurrent `319` / `512`, base `298` / `512`, delta `21`, W/L/T `45/24/443`, p `0.015432297453701265`
  - routing buckets
    - `ambiguous_proxy`: n `223`, delta `17`, W/L `29/12`, mean margin delta `0.37075387375771734`, mean loops `1.789838107578423`
    - `base_confident_direct_proxy`: n `147`, delta `-1`, W/L `0/1`, mean margin delta `-0.4318040858979533`, mean loops `1.807478480152532`
    - `conceptual_reasoning_proxy`: n `83`, delta `6`, W/L `12/6`, mean margin delta `0.2817936805357416`, mean loops `1.964452042278037`
    - `deep_numeric_proxy`: n `59`, delta `-1`, W/L `4/5`, mean margin delta `0.11105071538585727`, mean loops `1.8922335225646778`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `406/512`, recurrent `406/512`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `406` / `512`, base `406` / `512`, delta `0`, W/L/T `5/5/502`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `110`, delta `-1`, W/L `2/3`, mean margin delta `0.19335613745180044`, mean loops `None`
    - `base_confident_direct_proxy`: n `326`, delta `0`, W/L `0/0`, mean margin delta `-1.4716869353408142`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `43`, delta `1`, W/L `2/1`, mean margin delta `0.3063018260047186`, mean loops `None`
    - `deep_numeric_proxy`: n `33`, delta `0`, W/L `1/1`, mean margin delta `0.15020714165699303`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `10`, accuracy delta `0.0334` (base `98/299`, recurrent `108/299`)
  - paired evidence
    - aggregate `mean`: recurrent `108` / `299`, base `98` / `299`, delta `10`, W/L/T `29/19/251`, p `0.19341265286193732`
  - routing buckets
    - `ambiguous_proxy`: n `151`, delta `4`, W/L `14/10`, mean margin delta `0.13224930834296522`, mean loops `1.8222701354663629`
    - `base_confident_direct_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `-0.35161211490631106`, mean loops `1.9098331916332245`
    - `conceptual_reasoning_proxy`: n `78`, delta `4`, W/L `10/6`, mean margin delta `0.26326444286566514`, mean loops `1.9989439803056228`
    - `deep_numeric_proxy`: n `45`, delta `3`, W/L `5/2`, mean margin delta `0.02869215210278829`, mean loops `1.802374106645584`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `177/299`, recurrent `177/299`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `177` / `299`, base `177` / `299`, delta `0`, W/L/T `8/8/283`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `99`, delta `0`, W/L `3/3`, mean margin delta `0.2609990579428885`, mean loops `None`
    - `base_confident_direct_proxy`: n `120`, delta `0`, W/L `0/0`, mean margin delta `-1.0082934846005325`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `51`, delta `0`, W/L `3/3`, mean margin delta `0.376056719228954`, mean loops `None`
    - `deep_numeric_proxy`: n `29`, delta `0`, W/L `2/2`, mean margin delta `0.25506971447846327`, mean loops `None`
