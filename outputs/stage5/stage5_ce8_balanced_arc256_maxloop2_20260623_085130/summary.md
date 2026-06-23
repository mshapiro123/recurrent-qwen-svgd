# Stage 5 Benchmark Suite - stage5_ce8_balanced_arc256_maxloop2_20260623_085130

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1264.60`

## Recurrent vs Base

### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `202/256`, recurrent `204/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `204` / `256`, base `202` / `256`, delta `2`, W/L/T `2/0/254`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `1`, W/L `1/0`, mean margin delta `0.22561510468367488`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.318620035207376`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `1`, W/L `1/0`, mean margin delta `0.1744313015203391`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.25613512587733567`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-19`, accuracy delta `-0.0742` (base `146/256`, recurrent `127/256`)
  - paired evidence
    - aggregate `mean`: recurrent `127` / `256`, base `146` / `256`, delta `-19`, W/L/T `7/26/223`, p `0.0013187271542847157`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-14`, W/L `5/19`, mean margin delta `-0.07894186397730293`, mean loops `1.6299043045205586`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.3835629927260535`, mean loops `1.6429341730049678`
    - `conceptual_reasoning_proxy`: n `37`, delta `-2`, W/L `1/3`, mean margin delta `-0.3587983073414983`, mean loops `1.744996148186761`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.2949895724173515`, mean loops `1.7369533354236233`
### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `154/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `154` / `256`, delta `-1`, W/L/T `8/9/239`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `4/4`, mean margin delta `0.2427384857693879`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.9778618688718846`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `0.3325599542146342`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `1/3`, mean margin delta `0.20098537051429355`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0117` (base `87/256`, recurrent `90/256`)
  - paired evidence
    - aggregate `mean`: recurrent `90` / `256`, base `87` / `256`, delta `3`, W/L/T `16/13/227`, p `0.711071103811264`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `3`, W/L `8/5`, mean margin delta `-0.15988579077449272`, mean loops `1.6647785838541946`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.44115696430206297`, mean loops `1.698814527988434`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `7/5`, mean margin delta `-0.00633766833278868`, mean loops `1.748435550679763`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.10460873444875081`, mean loops `1.6636240810707763`
