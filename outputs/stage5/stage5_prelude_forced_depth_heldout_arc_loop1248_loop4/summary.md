# Stage 5 Benchmark Suite - stage5_prelude_forced_depth_heldout_arc_loop1248_loop4

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_prelude_path_development/summary.json`
- Checkpoint: `outputs/stage5/stage5_prelude_path_development/unfrozen/unfrozen_recurrent_step_300.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `4`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `436.55`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-4`, accuracy delta `-0.0930` (base `11/43`, recurrent `7/43`)
  - paired evidence
    - aggregate `mean`: recurrent `7` / `43`, base `11` / `43`, delta `-4`, W/L/T `4/8/31`, p `0.3876953125`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `4/3`, mean margin delta `-0.4237359975065504`, mean loops `6.111061892339161`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.35724782943725586`, mean loops `6.9468362132708235`
    - `deep_numeric_proxy`: n `9`, delta `-4`, W/L `0/4`, mean margin delta `-0.5480945706367493`, mean loops `6.142223742273119`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0233` (base `23/43`, recurrent `22/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `22` / `43`, base `23` / `43`, delta `-1`, W/L/T `4/5/34`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `2`, W/L `4/2`, mean margin delta `0.4375000040227961`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `-2`, W/L `0/2`, mean margin delta `-2.097656259643069`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.4791666579743226`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `0.4508928707135575`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-4` (base `11/43`, recurrent `7/43`)
