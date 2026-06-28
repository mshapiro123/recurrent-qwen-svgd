# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260628_031842_loop4

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `4`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2407.15`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-13`, accuracy delta `-0.0508` (base `88/256`, recurrent `75/256`)
  - paired evidence
    - aggregate `mean`: recurrent `75` / `256`, base `88` / `256`, delta `-13`, W/L/T `26/39/191`, p `0.13603239487789212`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `0`, W/L `15/15`, mean margin delta `0.004912488344239026`, mean loops `1.390775893261116`
    - `base_confident_direct_proxy`: n `25`, delta `-4`, W/L `0/4`, mean margin delta `-0.6232957792282104`, mean loops `1.4155764365196228`
    - `conceptual_reasoning_proxy`: n `72`, delta `-5`, W/L `8/13`, mean margin delta `-0.021271955635812547`, mean loops `1.5561795851422682`
    - `deep_numeric_proxy`: n `36`, delta `-4`, W/L `3/7`, mean margin delta `-0.31463414265049827`, mean loops `1.4876426955064137`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-14`, accuracy delta `-0.0547` (base `155/256`, recurrent `141/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `141` / `256`, base `155` / `256`, delta `-14`, W/L/T `17/31/208`, p `0.05946337525377032`
  - routing buckets
    - `ambiguous_proxy`: n `82`, delta `-3`, W/L `10/13`, mean margin delta `0.6582190010855656`, mean loops `None`
    - `base_confident_direct_proxy`: n `107`, delta `-9`, W/L `0/9`, mean margin delta `-2.531016346404686`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `6/6`, mean margin delta `0.8666666757522358`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `1/3`, mean margin delta `0.5404829588492936`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-13` (base `88/256`, recurrent `75/256`)
