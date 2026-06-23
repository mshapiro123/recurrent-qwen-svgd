# Stage 5 Benchmark Suite - stage5_ce8_arc128_loopctrl_on_20260623_073517

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `530.67`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0078` (base `68/128`, recurrent `69/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `69` / `128`, base `68` / `128`, delta `1`, W/L/T `4/3/121`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `2`, W/L `2/0`, mean margin delta `0.27623867994362067`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-1.360226071313145`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `1`, W/L `2/1`, mean margin delta `0.4437926237285137`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-2`, W/L `0/2`, mean margin delta `0.2187450352524008`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0234` (base `43/128`, recurrent `46/128`)
  - paired evidence
    - aggregate `mean`: recurrent `46` / `128`, base `43` / `128`, delta `3`, W/L/T `9/6/113`, p `0.60723876953125`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `1`, W/L `3/2`, mean margin delta `-0.27136721693236254`, mean loops `1.9119709288251812`
    - `base_confident_direct_proxy`: n `11`, delta `-2`, W/L `0/2`, mean margin delta `-0.4501088099046187`, mean loops `1.9521956145763397`
    - `conceptual_reasoning_proxy`: n `39`, delta `3`, W/L `5/2`, mean margin delta `0.013304316080533542`, mean loops `2.056774902038085`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.030261173844337463`, mean loops `1.8278570488095283`
