# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260628_031842_loop2

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `2`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2408.25`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0039` (base `88/256`, recurrent `87/256`)
  - paired evidence
    - aggregate `mean`: recurrent `87` / `256`, base `88` / `256`, delta `-1`, W/L/T `18/19/219`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `10/9`, mean margin delta `0.08252285021107371`, mean loops `1.390775893261116`
    - `base_confident_direct_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `-0.33676737308502197`, mean loops `1.4155764365196228`
    - `conceptual_reasoning_proxy`: n `72`, delta `1`, W/L `6/5`, mean margin delta `0.09409442709551917`, mean loops `1.5561795851422682`
    - `deep_numeric_proxy`: n `36`, delta `-2`, W/L `2/4`, mean margin delta `-0.23654374645815957`, mean loops `1.4876426955064137`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `155/256`, recurrent `154/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `154` / `256`, base `155` / `256`, delta `-1`, W/L/T `9/10/237`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `82`, delta `1`, W/L `8/7`, mean margin delta `0.5814278442450473`, mean loops `None`
    - `base_confident_direct_proxy`: n `107`, delta `0`, W/L `0/0`, mean margin delta `-2.164894848153056`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `-1`, W/L `1/2`, mean margin delta `0.7750000074298845`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `0/1`, mean margin delta `0.46117424492188025`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-1` (base `88/256`, recurrent `87/256`)
