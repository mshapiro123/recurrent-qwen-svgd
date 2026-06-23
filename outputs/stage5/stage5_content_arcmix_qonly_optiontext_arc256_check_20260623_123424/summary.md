# Stage 5 Benchmark Suite - stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424

- Status: `completed`
- Source summary: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/summary.json`
- Checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2093.26`

## Recurrent vs Base

### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `202/256`, recurrent `204/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `204` / `256`, base `202` / `256`, delta `2`, W/L/T `2/0/254`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `60`, delta `1`, W/L `1/0`, mean margin delta `0.21887715377379208`, mean loops `None`
    - `base_confident_direct_proxy`: n `162`, delta `0`, W/L `0/0`, mean margin delta `-1.3405740923565859`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `1`, W/L `1/0`, mean margin delta `0.1377248065546155`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.2319342924747616`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `9`, accuracy delta `0.0352` (base `146/256`, recurrent `155/256`)
  - paired evidence
    - aggregate `mean`: recurrent `155` / `256`, base `146` / `256`, delta `9`, W/L/T `19/10/227`, p `0.13604594767093658`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `7`, W/L `14/7`, mean margin delta `0.3419391048156609`, mean loops `1.6898939094300998`
    - `base_confident_direct_proxy`: n `70`, delta `0`, W/L `0/0`, mean margin delta `-0.2787301074181284`, mean loops `1.7044907416616166`
    - `conceptual_reasoning_proxy`: n `37`, delta `2`, W/L `4/2`, mean margin delta `0.24213242853010022`, mean loops `1.8661552659563116`
    - `deep_numeric_proxy`: n `31`, delta `0`, W/L `1/1`, mean margin delta `0.028928933605071035`, mean loops `1.8371420377685177`
### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `154/256`, recurrent `154/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `154` / `256`, base `154` / `256`, delta `0`, W/L/T `6/6/244`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `1`, W/L `3/2`, mean margin delta `0.2333144337406436`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.9485476887532664`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `1`, W/L `3/2`, mean margin delta `0.347985340287495`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `0/2`, mean margin delta `0.18848070905854303`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `10`, accuracy delta `0.0391` (base `87/256`, recurrent `97/256`)
  - paired evidence
    - aggregate `mean`: recurrent `97` / `256`, base `87` / `256`, delta `10`, W/L/T `21/11/224`, p `0.11018416518345475`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `4`, W/L `10/6`, mean margin delta `0.11294164114851292`, mean loops `1.7310235613245306`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.3360567331314087`, mean loops `1.7899339413642883`
    - `conceptual_reasoning_proxy`: n `72`, delta `3`, W/L `8/5`, mean margin delta `0.22984255850315094`, mean loops `1.873238415353828`
    - `deep_numeric_proxy`: n `36`, delta `3`, W/L `3/0`, mean margin delta `0.019907812277475994`, mean loops `1.7334681923190753`
