# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260623_145438

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `3359.23`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `18`, accuracy delta `0.0352` (base `298/512`, recurrent `316/512`)
  - paired evidence
    - aggregate `mean`: recurrent `316` / `512`, base `298` / `512`, delta `18`, W/L/T `43/25/444`, p `0.038460053348927506`
  - routing buckets
    - `ambiguous_proxy`: n `223`, delta `15`, W/L `28/13`, mean margin delta `0.3382960027376098`, mean loops `1.8047602891654713`
    - `base_confident_direct_proxy`: n `147`, delta `-1`, W/L `0/1`, mean margin delta `-0.4433823101932094`, mean loops `1.8227357430658104`
    - `conceptual_reasoning_proxy`: n `83`, delta `6`, W/L `12/6`, mean margin delta `0.23579248462814884`, mean loops `1.9787350580634842`
    - `deep_numeric_proxy`: n `59`, delta `-2`, W/L `3/5`, mean margin delta `0.08867964047496601`, mean loops `1.906527306366775`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `406/512`, recurrent `406/512`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `406` / `512`, base `406` / `512`, delta `0`, W/L/T `4/4/504`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `110`, delta `-1`, W/L `2/3`, mean margin delta `0.2004584256390279`, mean loops `None`
    - `base_confident_direct_proxy`: n `326`, delta `0`, W/L `0/0`, mean margin delta `-1.488840547369832`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `43`, delta `1`, W/L `2/1`, mean margin delta `0.3132349502641794`, mean loops `None`
    - `deep_numeric_proxy`: n `33`, delta `0`, W/L `0/0`, mean margin delta `0.16050191343846645`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `10`, accuracy delta `0.0334` (base `98/299`, recurrent `108/299`)
  - paired evidence
    - aggregate `mean`: recurrent `108` / `299`, base `98` / `299`, delta `10`, W/L/T `28/18/253`, p `0.18392482137699062`
  - routing buckets
    - `ambiguous_proxy`: n `151`, delta `4`, W/L `14/10`, mean margin delta `0.11503127237029423`, mean loops `1.8363620337139954`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.3577294397354126`, mean loops `1.9253419303894044`
    - `conceptual_reasoning_proxy`: n `78`, delta `4`, W/L `10/6`, mean margin delta `0.2427951808159168`, mean loops `2.0139368894772653`
    - `deep_numeric_proxy`: n `45`, delta `2`, W/L `4/2`, mean margin delta `0.007545442051357694`, mean loops `1.815333221576832`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `177/299`, recurrent `177/299`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `177` / `299`, base `177` / `299`, delta `0`, W/L/T `6/6/287`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `99`, delta `-1`, W/L `2/3`, mean margin delta `0.2641207687674335`, mean loops `None`
    - `base_confident_direct_proxy`: n `120`, delta `0`, W/L `0/0`, mean margin delta `-1.0122188972935935`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `51`, delta `1`, W/L `3/2`, mean margin delta `0.38716831202984003`, mean loops `None`
    - `deep_numeric_proxy`: n `29`, delta `0`, W/L `1/1`, mean margin delta `0.25915272509272413`, mean loops `None`
