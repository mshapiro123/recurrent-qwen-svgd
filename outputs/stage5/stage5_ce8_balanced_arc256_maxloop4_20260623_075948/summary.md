# Stage 5 Benchmark Suite - stage5_ce8_balanced_arc256_maxloop4_20260623_075948

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2057.42`

## Recurrent vs Base

### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `202/256`, recurrent `204/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `204` / `256`, base `202` / `256`, delta `2`, W/L/T `2/0/254`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `1`, W/L `1/0`, mean margin delta `0.23143768464991202`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.3430798232548202`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `1`, W/L `1/0`, mean margin delta `0.17318346311471292`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.25893651521764693`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-19`, accuracy delta `-0.0742` (base `146/256`, recurrent `127/256`)
  - paired evidence
    - aggregate `mean`: recurrent `127` / `256`, base `146` / `256`, delta `-19`, W/L/T `8/27/221`, p `0.001878225477412343`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-16`, W/L `5/21`, mean margin delta `-0.09901033569190462`, mean loops `1.706374662912498`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.39948948877198354`, mean loops `1.7235181387833185`
    - `conceptual_reasoning_proxy`: n `37`, delta `0`, W/L `2/2`, mean margin delta `-0.420843556120589`, mean loops `1.8806181678900848`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.3443260327462227`, mean loops `1.8533506720296797`
### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `154/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `154` / `256`, delta `-1`, W/L/T `8/9/239`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `4/4`, mean margin delta `0.24694752142431192`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-1.0039422192285268`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `0.3454101852710462`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `1/3`, mean margin delta `0.2061154910391479`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0195` (base `87/256`, recurrent `92/256`)
  - paired evidence
    - aggregate `mean`: recurrent `92` / `256`, base `87` / `256`, delta `5`, W/L/T `18/13/225`, p `0.47312965989112854`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `4`, W/L `9/5`, mean margin delta `-0.17835529499906835`, mean loops `1.747156574781025`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.4230860376358032`, mean loops `1.8061177265644073`
    - `conceptual_reasoning_proxy`: n `72`, delta `3`, W/L `8/5`, mean margin delta `-0.009867909881803725`, mean loops `1.8850792302853532`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.11252848969565497`, mean loops `1.7508019217186503`
