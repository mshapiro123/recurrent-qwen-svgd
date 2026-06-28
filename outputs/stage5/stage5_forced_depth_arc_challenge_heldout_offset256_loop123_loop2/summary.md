# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop2

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `2`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `234.79`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0698` (base `11/43`, recurrent `8/43`)
  - paired evidence
    - aggregate `mean`: recurrent `8` / `43`, base `11` / `43`, delta `-3`, W/L/T `1/4/38`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `-2`, W/L `1/3`, mean margin delta `-0.10925443683351789`, mean loops `1.3030755051544733`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.27379008134206134`, mean loops `1.5904409090677898`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `0.053775515821244985`, mean loops `1.413278533352746`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0465` (base `23/43`, recurrent `21/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `21` / `43`, base `23` / `43`, delta `-2`, W/L/T `0/2/41`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-2`, W/L `0/2`, mean margin delta `0.4392361089250901`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-2.057291669480037`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.4895833271245162`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `0`, W/L `0/0`, mean margin delta `0.6116071521703687`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-3` (base `11/43`, recurrent `8/43`)
