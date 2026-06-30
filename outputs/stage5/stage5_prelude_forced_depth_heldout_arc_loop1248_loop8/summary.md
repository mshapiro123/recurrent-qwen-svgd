# Stage 5 Benchmark Suite - stage5_prelude_forced_depth_heldout_arc_loop1248_loop8

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_prelude_path_development/summary.json`
- Checkpoint: `outputs/stage5/stage5_prelude_path_development/unfrozen/unfrozen_recurrent_step_300.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `8`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `436.18`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-4`, accuracy delta `-0.0930` (base `11/43`, recurrent `7/43`)
  - paired evidence
    - aggregate `mean`: recurrent `7` / `43`, base `11` / `43`, delta `-4`, W/L/T `3/7/33`, p `0.34375`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `-1`, W/L `2/3`, mean margin delta `-0.5377521727766309`, mean loops `6.111061892339161`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `1/1`, mean margin delta `0.19811741511027017`, mean loops `6.9468362132708235`
    - `deep_numeric_proxy`: n `9`, delta `-3`, W/L `0/3`, mean margin delta `-1.2448680731985304`, mean loops `6.142223742273119`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-4`, accuracy delta `-0.0930` (base `23/43`, recurrent `19/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `19` / `43`, base `23` / `43`, delta `-4`, W/L/T `5/9/29`, p `0.4239501953125`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `4/4`, mean margin delta `0.5399305673393732`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `-3`, W/L `0/3`, mean margin delta `-2.3710937468373836`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `1/1`, mean margin delta `0.5494791300346454`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `0.5200893099286726`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-4` (base `11/43`, recurrent `7/43`)
