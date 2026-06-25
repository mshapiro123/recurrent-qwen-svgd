# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260625_180322

- Status: `completed`
- Suite profile: `depth_signal_confirmation`
- Source summary: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'open_hard_arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `2626.69`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0078` (base `74/128`, recurrent `73/128`)
  - paired evidence
    - aggregate `mean`: recurrent `73` / `128`, base `74` / `128`, delta `-1`, W/L/T `4/5/119`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `0`, W/L `3/3`, mean margin delta `0.050491780042648315`, mean loops `1.0923999398946762`
    - `base_confident_direct_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.014800699377382124`, mean loops `1.0658344848735912`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.02457212805747986`, mean loops `1.142416763305664`
    - `deep_numeric_proxy`: n `17`, delta `0`, W/L `1/1`, mean margin delta `0.0011480415568632238`, mean loops `1.2092623009401209`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0156` (base `96/128`, recurrent `98/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `98` / `128`, base `96` / `128`, delta `2`, W/L/T `3/1/124`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `0`, W/L `1/1`, mean margin delta `-0.03378008172980377`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `0.09363353948464392`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `2`, W/L `2/0`, mean margin delta `0.1340921537950635`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.13577323924336168`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `6`, accuracy delta `0.0234` (base `87/256`, recurrent `93/256`)
  - paired evidence
    - aggregate `mean`: recurrent `93` / `256`, base `87` / `256`, delta `6`, W/L/T `9/3/244`, p `0.14599609375`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `3/2`, mean margin delta `-0.11545241412108506`, mean loops `1.0940974464907554`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.002705204486846924`, mean loops `1.1144379270076752`
    - `conceptual_reasoning_proxy`: n `72`, delta `4`, W/L `5/1`, mean margin delta `-0.0027040905422634548`, mean loops `1.1897952225473192`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.005765312247806125`, mean loops `1.1670848023560312`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-5`, accuracy delta `-0.0195` (base `154/256`, recurrent `149/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `149` / `256`, base `154` / `256`, delta `-5`, W/L/T `4/9/243`, p `0.266845703125`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-2`, W/L `2/4`, mean margin delta `-0.042158310776844676`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.13335570582389158`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `-0.010836307719970743`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-3`, W/L `0/3`, mean margin delta `-0.01258345796359759`, mean loops `None`
### open_hard_arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `6`, accuracy delta `0.0234` (base `87/256`, recurrent `93/256`)
  - paired evidence
    - aggregate `mean`: recurrent `93` / `256`, base `87` / `256`, delta `6`, W/L/T `9/3/244`, p `0.14599609375`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `3/2`, mean margin delta `-0.11545241412108506`, mean loops `1.0940974464907554`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.002705204486846924`, mean loops `1.1144379270076752`
    - `conceptual_reasoning_proxy`: n `72`, delta `4`, W/L `5/1`, mean margin delta `-0.0027040905422634548`, mean loops `1.1897952225473192`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.005765312247806125`, mean loops `1.1670848023560312`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-5`, accuracy delta `-0.0195` (base `154/256`, recurrent `149/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `149` / `256`, base `154` / `256`, delta `-5`, W/L/T `4/9/243`, p `0.266845703125`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-2`, W/L `2/4`, mean margin delta `-0.042158310776844676`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.13335570582389158`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `-0.010836307719970743`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-3`, W/L `0/3`, mean margin delta `-0.01258345796359759`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `6` (base `87/256`, recurrent `93/256`)
- `open_hard_arc_challenge` `content_question_only`: delta `6` (base `87/256`, recurrent `93/256`)
