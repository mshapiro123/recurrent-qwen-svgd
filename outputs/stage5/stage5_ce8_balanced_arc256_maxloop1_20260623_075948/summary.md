# Stage 5 Benchmark Suite - stage5_ce8_balanced_arc256_maxloop1_20260623_075948

- Status: `completed`
- Source summary: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/summary.json`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `881.41`

## Recurrent vs Base

### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0156` (base `202/256`, recurrent `206/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `206` / `256`, base `202` / `256`, delta `4`, W/L/T `4/0/252`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `2`, W/L `2/0`, mean margin delta `-0.13281249284627847`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `0.9321373370609549`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `2`, W/L `2/0`, mean margin delta `0.04241072413112436`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.15781249860301613`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-15`, accuracy delta `-0.0586` (base `146/256`, recurrent `131/256`)
  - paired evidence
    - aggregate `mean`: recurrent `131` / `256`, base `146` / `256`, delta `-15`, W/L/T `3/18/235`, p `0.0014896392822265625`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-12`, W/L `2/14`, mean margin delta `-0.11269135798438121`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.02258491771561759`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `37`, delta `0`, W/L `0/0`, mean margin delta `-0.2903691530227661`, mean loops `1.0`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.031559161601528045`, mean loops `1.0`
### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-3`, accuracy delta `-0.0117` (base `154/256`, recurrent `151/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `151` / `256`, base `154` / `256`, delta `-3`, W/L/T `5/8/243`, p `0.5810546875`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-1`, W/L `2/3`, mean margin delta `-0.24987139994634983`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `0.7506365791447159`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `-0.4680555498072257`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `1/3`, mean margin delta `-0.2121212177056198`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0039` (base `87/256`, recurrent `88/256`)
  - paired evidence
    - aggregate `mean`: recurrent `88` / `256`, base `87` / `256`, delta `1`, W/L/T `8/7/241`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `-1`, W/L `3/4`, mean margin delta `-0.2340227220116592`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.0011661243438720703`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `72`, delta `1`, W/L `4/3`, mean margin delta `-0.11016592548953162`, mean loops `1.0`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `-0.044359816445244685`, mean loops `1.0`
