# Stage 5 Benchmark Suite - stage5_halt_repair_arc256_check_20260623_102430

- Status: `completed`
- Source summary: `outputs/stage5/stage5_halt_repair_arc128_check_20260623_100230/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_challenge', 'arc_easy']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2057.68`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0039` (base `154/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `154` / `256`, delta `-1`, W/L/T `7/8/241`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `4/4`, mean margin delta `0.21179400129941267`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.8863232500517`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `0.3012883154106223`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `1/2`, mean margin delta `0.18388031413214226`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0195` (base `87/256`, recurrent `92/256`)
  - paired evidence
    - aggregate `mean`: recurrent `92` / `256`, base `87` / `256`, delta `5`, W/L/T `19/14/223`, p `0.48685024166479707`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `3`, W/L `9/6`, mean margin delta `-0.1946077056047393`, mean loops `None`
    - `base_confident_direct_proxy`: n `25`, delta `-3`, W/L `0/3`, mean margin delta `-0.43082396030426023`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `72`, delta `3`, W/L `8/5`, mean margin delta `-0.017811900211705103`, mean loops `None`
    - `deep_numeric_proxy`: n `36`, delta `2`, W/L `2/0`, mean margin delta `-0.11594879958364698`, mean loops `None`
### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0156` (base `202/256`, recurrent `206/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `206` / `256`, base `202` / `256`, delta `4`, W/L/T `4/0/252`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `2`, W/L `2/0`, mean margin delta `0.2125107751150305`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.2103065794974819`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `2`, W/L `2/0`, mean margin delta `0.1637334218248725`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.22645059716887772`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-19`, accuracy delta `-0.0742` (base `146/256`, recurrent `127/256`)
  - paired evidence
    - aggregate `mean`: recurrent `127` / `256`, base `146` / `256`, delta `-19`, W/L/T `9/28/219`, p `0.0025632079923525453`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-15`, W/L `6/21`, mean margin delta `-0.10672366669622518`, mean loops `None`
    - `base_confident_direct_proxy`: n `70`, delta `-1`, W/L `0/1`, mean margin delta `-0.3906397770558085`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `37`, delta `-1`, W/L `2/3`, mean margin delta `-0.4626585116257539`, mean loops `None`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `-0.33851344162417996`, mean loops `None`
