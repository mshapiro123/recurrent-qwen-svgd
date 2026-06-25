# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260625_194738_loop2

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `2`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `844.82`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0078` (base `87/256`, recurrent `85/256`)
  - paired evidence
    - aggregate `mean`: recurrent `85` / `256`, base `87` / `256`, delta `-2`, W/L/T `20/22/214`, p `0.8776143287523155`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `-1`, W/L `9/10`, mean margin delta `-0.05572431019651211`, mean loops `1.2029786555586146`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.49683208465576173`, mean loops `1.2389851164817811`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `9/7`, mean margin delta `0.10755624290969637`, mean loops `1.3237955628169908`
    - `deep_numeric_proxy`: n `36`, delta `0`, W/L `2/2`, mean margin delta `-0.18121273981200325`, mean loops `1.2916781507708408`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `154/256`, recurrent `156/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `156` / `256`, base `154` / `256`, delta `2`, W/L/T `9/7/240`, p `0.803619384765625`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `4`, W/L `7/3`, mean margin delta `0.5264917651896553`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-1.918258101100973`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `0.6729166695692886`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `0/2`, mean margin delta `0.3816287793557752`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-2` (base `87/256`, recurrent `85/256`)
