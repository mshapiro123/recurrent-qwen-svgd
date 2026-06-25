# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260625_194738_loop3

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `3`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `836.04`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `87/256`, recurrent `87/256`)
  - paired evidence
    - aggregate `mean`: recurrent `87` / `256`, base `87` / `256`, delta `0`, W/L/T `29/29/198`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `13/12`, mean margin delta `-0.14597400804845298`, mean loops `1.2029786555586146`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.8145232009887695`, mean loops `1.2389851164817811`
    - `conceptual_reasoning_proxy`: n `72`, delta `1`, W/L `12/11`, mean margin delta `0.06178420368168089`, mean loops `1.3237955628169908`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `4/3`, mean margin delta `-0.2911287960078981`, mean loops `1.2916781507708408`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `154/256`, recurrent `154/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `154` / `256`, base `154` / `256`, delta `0`, W/L/T `12/12/232`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `8/8`, mean margin delta `0.6163837450667602`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `-1`, W/L `0/1`, mean margin delta `-2.293952545662695`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `2`, W/L `4/2`, mean margin delta `0.7993055606105676`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `0/1`, mean margin delta `0.46259468573738227`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `0` (base `87/256`, recurrent `87/256`)
