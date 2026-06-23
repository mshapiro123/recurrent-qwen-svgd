# Stage 5 Benchmark Suite - stage5_arc_mix_survival_depth_after_aggregate_gate_20260623_depth_routing_probe_debiased_gate

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_mix_survival_depth_after_aggregate_gate_20260623_depth_routing_probe/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_survival_depth_after_aggregate_gate_20260623_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `774.26`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `8`, accuracy delta `0.0625` (base `71/128`, recurrent `79/128`)
  - paired evidence
    - aggregate `mean`: recurrent `79` / `128`, base `71` / `128`, delta `8`, W/L/T `12/4/112`, p `0.076812744140625`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `7`, W/L `7/0`, mean margin delta `0.3987627363204956`, mean loops `1.6522232455015182`
    - `base_confident_direct_proxy`: n `42`, delta `0`, W/L `0/0`, mean margin delta `-0.48621255478688646`, mean loops `1.6699769040421832`
    - `conceptual_reasoning_proxy`: n `24`, delta `2`, W/L `4/2`, mean margin delta `0.16819619139035544`, mean loops `1.7536898888647556`
    - `deep_numeric_proxy`: n `12`, delta `-1`, W/L `1/2`, mean margin delta `0.192806214094162`, mean loops `1.724744737148285`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `103/128`, recurrent `103/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `103` / `128`, base `103` / `128`, delta `0`, W/L/T `1/1/126`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `0.08281958315521479`, mean loops `None`
    - `base_confident_direct_proxy`: n `86`, delta `0`, W/L `0/0`, mean margin delta `-1.3908813295828615`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `13`, delta `1`, W/L `1/0`, mean margin delta `0.24895574610966903`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.03038859274238348`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `11/43`, recurrent `11/43`)
  - paired evidence
    - aggregate `mean`: recurrent `11` / `43`, base `11` / `43`, delta `0`, W/L/T `2/2/39`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `2/1`, mean margin delta `0.05597481983048575`, mean loops `1.6655871186937605`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.01833273967107137`, mean loops `1.8600846727689107`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.0426942507425944`, mean loops `1.6389904585149553`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0465` (base `23/43`, recurrent `21/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `21` / `43`, base `23` / `43`, delta `-2`, W/L/T `1/3/39`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `0/1`, mean margin delta `0.15780457803824297`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.6173742778288821`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.1566665261052549`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `0`, W/L `1/1`, mean margin delta `0.30107217960591826`, mean loops `None`
