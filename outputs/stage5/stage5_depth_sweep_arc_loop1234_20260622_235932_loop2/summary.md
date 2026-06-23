# Stage 5 Benchmark Suite - stage5_depth_sweep_arc_loop1234_20260622_235932_loop2

- Status: `completed`
- Source summary: `outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `298.45`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0195` (base `186/256`, recurrent `181/256`)
  - paired evidence
    - aggregate `mean`: recurrent `181` / `256`, base `186` / `256`, delta `-5`, W/L/T `15/20/221`, p `0.49955983320251107`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `-4`, W/L `6/10`, mean margin delta `0.47831026487983763`, mean loops `1.8299812050536275`
    - `base_confident_direct_proxy`: n `156`, delta `-8`, W/L `0/8`, mean margin delta `-2.8010324639229416`, mean loops `1.8301563423031415`
    - `conceptual_reasoning_proxy`: n `17`, delta `3`, W/L `5/2`, mean margin delta `0.35823262789670157`, mean loops `1.8325419163002687`
    - `deep_numeric_proxy`: n `19`, delta `4`, W/L `4/0`, mean margin delta `1.0045474516718012`, mean loops `1.830786794424057`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.0273` (base `148/256`, recurrent `141/256`)
  - paired evidence
    - aggregate `mean`: recurrent `141` / `256`, base `148` / `256`, delta `-7`, W/L/T `17/24/215`, p `0.34888887944907765`
  - routing buckets
    - `ambiguous_proxy`: n `75`, delta `-2`, W/L `8/10`, mean margin delta `0.6395153894027075`, mean loops `1.8310030187500848`
    - `base_confident_direct_proxy`: n `111`, delta `-9`, W/L `0/9`, mean margin delta `-1.932945881861153`, mean loops `1.831482760004095`
    - `conceptual_reasoning_proxy`: n `47`, delta `3`, W/L `7/4`, mean margin delta `0.571992656572702`, mean loops `1.8316889715955613`
    - `deep_numeric_proxy`: n `23`, delta `1`, W/L `2/1`, mean margin delta `0.4575685965626136`, mean loops `1.831098511599112`
