# Stage 5 Benchmark Suite - stage5_depth_sweep_arc_loop1234_20260622_235932_loop3

- Status: `completed`
- Source summary: `outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `377.34`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-18`, accuracy delta `-0.0703` (base `186/256`, recurrent `168/256`)
  - paired evidence
    - aggregate `mean`: recurrent `168` / `256`, base `186` / `256`, delta `-18`, W/L/T `14/32/210`, p `0.011351591436778108`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `-1`, W/L `7/8`, mean margin delta `0.6587018852587789`, mean loops `2.5113055547699332`
    - `base_confident_direct_proxy`: n `156`, delta `-20`, W/L `0/20`, mean margin delta `-3.0216390928238606`, mean loops `2.511593785347083`
    - `conceptual_reasoning_proxy`: n `17`, delta `1`, W/L `3/2`, mean margin delta `0.3363480532870573`, mean loops `2.518200688502368`
    - `deep_numeric_proxy`: n `19`, delta `2`, W/L `4/2`, mean margin delta `1.137844101378792`, mean loops `2.5132908664251628`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-15`, accuracy delta `-0.0586` (base `148/256`, recurrent `133/256`)
  - paired evidence
    - aggregate `mean`: recurrent `133` / `256`, base `148` / `256`, delta `-15`, W/L/T `30/45/181`, p `0.10534226887460488`
  - routing buckets
    - `ambiguous_proxy`: n `75`, delta `3`, W/L `16/13`, mean margin delta `0.8096644677718481`, mean loops `2.5141235979398093`
    - `base_confident_direct_proxy`: n `111`, delta `-24`, W/L `0/24`, mean margin delta `-2.107486632744874`, mean loops `2.5155092879458594`
    - `conceptual_reasoning_proxy`: n `47`, delta `5`, W/L `11/6`, mean margin delta `0.7050843028787602`, mean loops `2.5159354539627725`
    - `deep_numeric_proxy`: n `23`, delta `1`, W/L `3/2`, mean margin delta `0.6606204435229301`, mean loops `2.5141381958256597`
