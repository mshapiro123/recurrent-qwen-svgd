# Stage 5 Benchmark Suite - stage5_depth_sweep_arc_loop1234_20260622_235932_loop4

- Status: `completed`
- Source summary: `outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `458.15`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-27`, accuracy delta `-0.1055` (base `186/256`, recurrent `159/256`)
  - paired evidence
    - aggregate `mean`: recurrent `159` / `256`, base `186` / `256`, delta `-27`, W/L/T `13/40/203`, p `0.0002685401188191605`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `2`, W/L `9/7`, mean margin delta `0.6583670822437853`, mean loops `3.066044107079506`
    - `base_confident_direct_proxy`: n `156`, delta `-29`, W/L `0/29`, mean margin delta `-3.101633828060212`, mean loops `3.066173988886369`
    - `conceptual_reasoning_proxy`: n `17`, delta `1`, W/L `2/1`, mean margin delta `0.21650679672465606`, mean loops `3.0780674008762134`
    - `deep_numeric_proxy`: n `19`, delta `-1`, W/L `2/3`, mean margin delta `1.115692333171242`, mean loops `3.069000244140625`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-19`, accuracy delta `-0.0742` (base `148/256`, recurrent `129/256`)
  - paired evidence
    - aggregate `mean`: recurrent `129` / `256`, base `148` / `256`, delta `-19`, W/L/T `30/49/177`, p `0.042165443994951105`
  - routing buckets
    - `ambiguous_proxy`: n `75`, delta `7`, W/L `18/11`, mean margin delta `0.7963637683788936`, mean loops `3.071057378715939`
    - `base_confident_direct_proxy`: n `111`, delta `-28`, W/L `0/28`, mean margin delta `-2.1330062631383404`, mean loops `3.0735748322160394`
    - `conceptual_reasoning_proxy`: n `47`, delta `1`, W/L `8/7`, mean margin delta `0.6876128258064707`, mean loops `3.074156602646442`
    - `deep_numeric_proxy`: n `23`, delta `1`, W/L `4/3`, mean margin delta `0.6668601065226223`, mean loops `3.0705149761144668`
