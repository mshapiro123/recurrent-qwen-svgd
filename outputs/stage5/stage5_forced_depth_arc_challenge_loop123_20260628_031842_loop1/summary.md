# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260628_031842_loop1

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2403.64`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0039` (base `88/256`, recurrent `89/256`)
  - paired evidence
    - aggregate `mean`: recurrent `89` / `256`, base `88` / `256`, delta `1`, W/L/T `3/2/251`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `1/0`, mean margin delta `-0.07088972834067617`, mean loops `1.390775893261116`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.15344327449798584`, mean loops `1.4155764365196228`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `2/0`, mean margin delta `-0.06824799875418346`, mean loops `1.5561795851422682`
    - `deep_numeric_proxy`: n `36`, delta `-2`, W/L `0/2`, mean margin delta `-0.09118095205889808`, mean loops `1.4876426955064137`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `155/256`, recurrent `155/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `155` / `256`, base `155` / `256`, delta `0`, W/L/T `4/4/248`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `82`, delta `1`, W/L `3/2`, mean margin delta `-0.07907774890924432`, mean loops `None`
    - `base_confident_direct_proxy`: n `107`, delta `0`, W/L `0/0`, mean margin delta `0.3780373832400209`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `1/1`, mean margin delta `-0.1284722098821981`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `0/1`, mean margin delta `-0.10748105902563442`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `1` (base `88/256`, recurrent `89/256`)
