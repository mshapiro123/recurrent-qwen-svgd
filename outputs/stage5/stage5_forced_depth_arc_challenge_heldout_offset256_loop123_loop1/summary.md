# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop1

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `237.28`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `11/43`, recurrent `11/43`)
  - paired evidence
    - aggregate `mean`: recurrent `11` / `43`, base `11` / `43`, delta `0`, W/L/T `0/0/43`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `0`, W/L `0/0`, mean margin delta `-0.054729725633348734`, mean loops `1.3030755051544733`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.10192575057347615`, mean loops `1.5904409090677898`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.01958947049246894`, mean loops `1.413278533352746`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0233` (base `23/43`, recurrent `24/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `24` / `43`, base `23` / `43`, delta `1`, W/L/T `1/0/42`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `0/0`, mean margin delta `-0.11458332548823415`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `0.2929687470314093`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `-0.052083349165817104`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `1`, W/L `1/0`, mean margin delta `-0.004464286379516125`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `0` (base `11/43`, recurrent `11/43`)
