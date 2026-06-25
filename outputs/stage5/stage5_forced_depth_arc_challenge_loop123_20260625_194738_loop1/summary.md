# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260625_194738_loop1

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `852.88`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0078` (base `87/256`, recurrent `89/256`)
  - paired evidence
    - aggregate `mean`: recurrent `89` / `256`, base `87` / `256`, delta `2`, W/L/T `7/5/244`, p `0.7744140625`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `-1`, W/L `2/3`, mean margin delta `-0.13339634445624624`, mean loops `1.2029786555586146`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.03518418073654175`, mean loops `1.2389851164817811`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `4/2`, mean margin delta `-0.020833153691556718`, mean loops `1.3237955628169908`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.009342322746912638`, mean loops `1.2916781507708408`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-6`, accuracy delta `-0.0234` (base `154/256`, recurrent `148/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `148` / `256`, base `154` / `256`, delta `-6`, W/L/T `3/9/244`, p `0.14599609375`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-2`, W/L `2/4`, mean margin delta `-0.18287037194748657`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `0.565567143524984`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `-1`, W/L `1/2`, mean margin delta `-0.3774305480558218`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-3`, W/L `0/3`, mean margin delta `-0.17140151878508428`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `2` (base `87/256`, recurrent `89/256`)
