# Stage 5 Benchmark Suite - stage5_content_direct_preserve_m05_arc256_check_20260623_113424

- Status: `completed`
- Source summary: `outputs/stage5/stage5_content_direct_preserve_m05_20260623_1122/summary.json`
- Checkpoint: `outputs/stage5/stage5_content_direct_preserve_m05_20260623_1122/phase1_direct_preserve/phase1_step_150.pt`
- Benchmarks: `['arc_challenge', 'arc_easy']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2064.52`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0078` (base `154/256`, recurrent `152/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `152` / `256`, base `154` / `256`, delta `-2`, W/L/T `6/8/242`, p `0.79052734375`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-2`, W/L `2/4`, mean margin delta `0.21952445609226762`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.9286139668532457`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `0.32085603827921055`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `1/2`, mean margin delta `0.1877927062468547`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `4`, accuracy delta `0.0156` (base `87/256`, recurrent `91/256`)
  - paired evidence
    - aggregate `mean`: recurrent `91` / `256`, base `87` / `256`, delta `4`, W/L/T `17/13/226`, p `0.584664711728692`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `4`, W/L `9/5`, mean margin delta `-0.08483422529406665`, mean loops `None`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.42496581077575685`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `7/5`, mean margin delta `0.06535019063287312`, mean loops `None`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.07428194582462311`, mean loops `None`
### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0039` (base `202/256`, recurrent `203/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `203` / `256`, base `202` / `256`, delta `1`, W/L/T `2/1/253`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `0`, W/L `1/1`, mean margin delta `0.2214749010823046`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.2898578896924542`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `1`, W/L `1/0`, mean margin delta `0.15208852437457868`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.21697969730012118`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-8`, accuracy delta `-0.0312` (base `146/256`, recurrent `138/256`)
  - paired evidence
    - aggregate `mean`: recurrent `138` / `256`, base `146` / `256`, delta `-8`, W/L/T `11/19/226`, p `0.20048842206597328`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-4`, W/L `8/12`, mean margin delta `0.07071591282294969`, mean loops `None`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.33258763147251946`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `37`, delta `-1`, W/L `2/3`, mean margin delta `-0.26044934504740946`, mean loops `None`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.22139548294005856`, mean loops `None`
