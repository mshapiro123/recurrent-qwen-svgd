# Stage 5 Benchmark Suite - stage5_ce8_balanced_arc256_maxloop3_20260623_085130

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1691.89`

## Recurrent vs Base

### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `202/256`, recurrent `204/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `204` / `256`, base `202` / `256`, delta `2`, W/L/T `2/0/254`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `1`, W/L `1/0`, mean margin delta `0.231172006817845`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.3422368235824298`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `1`, W/L `1/0`, mean margin delta `0.17391079225178277`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.25916020045988264`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-19`, accuracy delta `-0.0742` (base `146/256`, recurrent `127/256`)
  - paired evidence
    - aggregate `mean`: recurrent `127` / `256`, base `146` / `256`, delta `-19`, W/L/T `8/27/221`, p `0.001878225477412343`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-16`, W/L `5/21`, mean margin delta `-0.09338246008097115`, mean loops `1.690108984961348`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.40153410200561795`, mean loops `1.705695983341762`
    - `conceptual_reasoning_proxy`: n `37`, delta `0`, W/L `2/2`, mean margin delta `-0.4109949324582074`, mean loops `1.8476988157710514`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.33591292558177827`, mean loops `1.8289926167457335`
### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `154/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `154` / `256`, delta `-1`, W/L/T `8/9/239`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `4/4`, mean margin delta `0.24677991776999256`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-1.0028958762791094`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `0.34504224716478754`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `1/3`, mean margin delta `0.20603754856821263`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0195` (base `87/256`, recurrent `92/256`)
  - paired evidence
    - aggregate `mean`: recurrent `92` / `256`, base `87` / `256`, delta `5`, W/L/T `18/13/225`, p `0.47312965989112854`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `4`, W/L `9/5`, mean margin delta `-0.17576542666287925`, mean loops `1.7306395708060847`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.4339637565612793`, mean loops `1.7821850562095642`
    - `conceptual_reasoning_proxy`: n `72`, delta `3`, W/L `8/5`, mean margin delta `-0.007918785015741983`, mean loops `1.8530049191580877`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.11244959798124102`, mean loops `1.733561405705081`
