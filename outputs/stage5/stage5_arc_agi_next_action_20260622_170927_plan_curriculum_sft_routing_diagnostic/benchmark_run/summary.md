# Stage 5 Benchmark Suite - stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft_routing_diagnostic_arc_easy64_challenge64

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `176.07`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-11`, accuracy delta `-0.1719` (base `49/64`, recurrent `38/64`)
  - paired evidence
    - aggregate `mean`: recurrent `38` / `64`, base `49` / `64`, delta `-11`, W/L/T `3/14/47`, p `0.012725830078125`
  - routing buckets
    - `ambiguous_proxy`: n `13`, delta `1`, W/L `2/1`, mean margin delta `0.7570813366999993`, mean loops `3.060586392879486`
    - `base_confident_direct_proxy`: n `41`, delta `-12`, W/L `0/12`, mean margin delta `-2.994490166955696`, mean loops `3.0650519795534086`
    - `conceptual_reasoning_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.38864484429359436`, mean loops `3.082410764694214`
    - `deep_numeric_proxy`: n `5`, delta `0`, W/L `1/1`, mean margin delta `-0.03665660619735718`, mean loops `3.0699172377586366`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.1094` (base `36/64`, recurrent `29/64`)
  - paired evidence
    - aggregate `mean`: recurrent `29` / `64`, base `36` / `64`, delta `-7`, W/L/T `7/14/43`, p `0.18924713134765625`
  - routing buckets
    - `ambiguous_proxy`: n `17`, delta `4`, W/L `6/2`, mean margin delta `1.1284498338971067`, mean loops `3.0632144399717744`
    - `base_confident_direct_proxy`: n `27`, delta `-9`, W/L `0/9`, mean margin delta `-2.2728092433125884`, mean loops `3.0782544127217046`
    - `conceptual_reasoning_proxy`: n `12`, delta `-1`, W/L `1/2`, mean margin delta `0.2215343713760376`, mean loops `3.078439479072889`
    - `deep_numeric_proxy`: n `8`, delta `-1`, W/L `0/1`, mean margin delta `0.4861076604574919`, mean loops `3.074424721300602`
