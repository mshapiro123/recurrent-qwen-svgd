# Stage 5 Benchmark Suite - stage5_direct_preservation_confirm_loop1_20260622_235024

- Status: `completed`
- Source summary: `outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `247.62`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0195` (base `186/256`, recurrent `191/256`)
  - paired evidence
    - aggregate `mean`: recurrent `191` / `256`, base `186` / `256`, delta `5`, W/L/T `6/1/249`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `1`, W/L `2/1`, mean margin delta `-0.07812500355066732`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `156`, delta `0`, W/L `0/0`, mean margin delta `0.3377403892533412`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `17`, delta `4`, W/L `4/0`, mean margin delta `0.022058826597297892`, mean loops `1.0`
    - `deep_numeric_proxy`: n `19`, delta `0`, W/L `0/0`, mean margin delta `-0.14473681583216316`, mean loops `1.0`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0078` (base `148/256`, recurrent `150/256`)
  - paired evidence
    - aggregate `mean`: recurrent `150` / `256`, base `148` / `256`, delta `2`, W/L/T `6/4/246`, p `0.75390625`
  - routing buckets
    - `ambiguous_proxy`: n `75`, delta `-1`, W/L `2/3`, mean margin delta `-0.111666670118769`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `111`, delta `0`, W/L `0/0`, mean margin delta `0.3079954893873619`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `47`, delta `2`, W/L `3/1`, mean margin delta `-0.13430851915890865`, mean loops `1.0`
    - `deep_numeric_proxy`: n `23`, delta `1`, W/L `1/0`, mean margin delta `-0.1250000074505806`, mean loops `1.0`
