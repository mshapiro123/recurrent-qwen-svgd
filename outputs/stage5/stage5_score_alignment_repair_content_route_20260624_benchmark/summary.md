# Stage 5 Benchmark Suite - stage5_score_alignment_repair_content_route_20260624_benchmark

- Status: `completed`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Checkpoint: `outputs/stage5/stage5_score_alignment_repair_content_route_20260624/phase1_surface_align/phase1_step_75.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2085.72`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.0273` (base `146/256`, recurrent `139/256`)
  - paired evidence
    - aggregate `mean`: recurrent `139` / `256`, base `146` / `256`, delta `-7`, W/L/T `6/13/237`, p `0.1670684814453125`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-2`, W/L `6/8`, mean margin delta `0.08417586906481597`, mean loops `1.1566133211224765`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `0.021537676666464126`, mean loops `1.1454202376093183`
    - `conceptual_reasoning_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.061710122469309216`, mean loops `1.3170218652970083`
    - `deep_numeric_proxy`: n `31`, delta `-3`, W/L `0/3`, mean margin delta `-0.01004856824874878`, mean loops `1.3251274055050266`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `202/256`, recurrent `204/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `204` / `256`, base `202` / `256`, delta `2`, W/L/T `3/1/252`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `0`, W/L `1/1`, mean margin delta `0.016903280559927226`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `0.01250169027721032`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `2`, W/L `2/0`, mean margin delta `0.13588155872587646`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.06421264060772955`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `4`, accuracy delta `0.0156` (base `87/256`, recurrent `91/256`)
  - paired evidence
    - aggregate `mean`: recurrent `91` / `256`, base `87` / `256`, delta `4`, W/L/T `9/5/242`, p `0.4239501953125`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `4/3`, mean margin delta `-0.12343984115414502`, mean loops `1.2069633248214153`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.1220540738105774`, mean loops `1.242896341085434`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `4/2`, mean margin delta `0.000120971765783098`, mean loops `1.3298433394067817`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `0.005078769392437405`, mean loops `1.296353276129122`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `154/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `154` / `256`, delta `-1`, W/L/T `6/7/243`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-1`, W/L `2/3`, mean margin delta `-0.029144012794634442`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.18250211047577775`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `-0.0070830047751466434`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `1/2`, mean margin delta `0.04676748275982612`, mean loops `None`
