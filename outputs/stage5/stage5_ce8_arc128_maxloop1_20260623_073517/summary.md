# Stage 5 Benchmark Suite - stage5_ce8_arc128_maxloop1_20260623_073517

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `245.63`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0156` (base `68/128`, recurrent `66/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `66` / `128`, base `68` / `128`, delta `-2`, W/L/T `1/3/124`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `0`, W/L `0/0`, mean margin delta `-0.23231589163530822`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `0.8355978284139975`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `0`, W/L `1/1`, mean margin delta `-0.5556249878183007`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-2`, W/L `0/2`, mean margin delta `-0.13690476274738708`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0078` (base `43/128`, recurrent `44/128`)
  - paired evidence
    - aggregate `mean`: recurrent `44` / `128`, base `43` / `128`, delta `1`, W/L/T `4/3/121`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `-2`, W/L `0/2`, mean margin delta `-0.2946001496808282`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.23483209176497025`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `39`, delta `2`, W/L `3/1`, mean margin delta `-0.1023249320494823`, mean loops `1.0`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.06344329714775085`, mean loops `1.0`
