# Stage 5 Benchmark Suite - stage5_prelude_forced_depth_heldout_arc_loop1248_loop2

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_prelude_path_development/summary.json`
- Checkpoint: `outputs/stage5/stage5_prelude_path_development/unfrozen/unfrozen_recurrent_step_300.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `2`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `436.46`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0233` (base `11/43`, recurrent `10/43`)
  - paired evidence
    - aggregate `mean`: recurrent `10` / `43`, base `11` / `43`, delta `-1`, W/L/T `4/5/34`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `3/2`, mean margin delta `-0.35696873920304434`, mean loops `6.111061892339161`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `1/1`, mean margin delta `0.44230814774831134`, mean loops `6.9468362132708235`
    - `deep_numeric_proxy`: n `9`, delta `-2`, W/L `0/2`, mean margin delta `-0.15660005807876587`, mean loops `6.142223742273119`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-3`, accuracy delta `-0.0698` (base `23/43`, recurrent `20/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `20` / `43`, base `23` / `43`, delta `-3`, W/L/T `1/4/38`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `1/2`, mean margin delta `0.1614583314836232`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-1.2578124931120935`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.18749998634060225`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `0.31919643628810135`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-1` (base `11/43`, recurrent `10/43`)
