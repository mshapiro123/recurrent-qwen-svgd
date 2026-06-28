# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260628_031842_loop3

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `3`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2408.74`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0195` (base `88/256`, recurrent `83/256`)
  - paired evidence
    - aggregate `mean`: recurrent `83` / `256`, base `88` / `256`, delta `-5`, W/L/T `26/31/199`, p `0.5966417603730826`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `2`, W/L `15/13`, mean margin delta `0.03153127528787628`, mean loops `1.390775893261116`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.5013008451461792`, mean loops `1.4155764365196228`
    - `conceptual_reasoning_proxy`: n `72`, delta `-3`, W/L `8/11`, mean margin delta `0.0360036657916175`, mean loops `1.5561795851422682`
    - `deep_numeric_proxy`: n `36`, delta `-1`, W/L `3/4`, mean margin delta `-0.30923474000559914`, mean loops `1.4876426955064137`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-7`, accuracy delta `-0.0273` (base `155/256`, recurrent `148/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `148` / `256`, base `155` / `256`, delta `-7`, W/L/T `15/22/219`, p `0.3240086000878364`
  - routing buckets
    - `ambiguous_proxy`: n `82`, delta `-5`, W/L `8/13`, mean margin delta `0.6462144242804043`, mean loops `None`
    - `base_confident_direct_proxy`: n `107`, delta `-2`, W/L `0/2`, mean margin delta `-2.467289715402795`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `6/5`, mean margin delta `0.8559027835519777`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `1/2`, mean margin delta `0.5274621291283631`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-5` (base `88/256`, recurrent `83/256`)
