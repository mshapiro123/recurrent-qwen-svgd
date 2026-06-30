# Stage 5 Benchmark Suite - stage5_prelude_forced_depth_heldout_arc_loop1248_loop1

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_prelude_path_development/summary.json`
- Checkpoint: `outputs/stage5/stage5_prelude_path_development/unfrozen/unfrozen_recurrent_step_300.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `450.46`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `4`, accuracy delta `0.0930` (base `11/43`, recurrent `15/43`)
  - paired evidence
    - aggregate `mean`: recurrent `15` / `43`, base `11` / `43`, delta `4`, W/L/T `5/1/37`, p `0.21875`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `3`, W/L `4/1`, mean margin delta `-0.11008601103510175`, mean loops `6.111061892339161`
    - `conceptual_reasoning_proxy`: n `6`, delta `1`, W/L `1/0`, mean margin delta `0.3418159484863281`, mean loops `6.9468362132708235`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `-0.0552455120616489`, mean loops `6.142223742273119`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0465` (base `23/43`, recurrent `21/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `21` / `43`, base `23` / `43`, delta `-2`, W/L/T `1/3/39`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `1/2`, mean margin delta `-0.5642361162645102`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `1.5442707820911892`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `-0.8385416441985095`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `-0.7656249970729861`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `4` (base `11/43`, recurrent `15/43`)
