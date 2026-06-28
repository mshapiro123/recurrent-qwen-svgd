# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop3

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `3`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `234.71`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0233` (base `11/43`, recurrent `10/43`)
  - paired evidence
    - aggregate `mean`: recurrent `10` / `43`, base `11` / `43`, delta `-1`, W/L/T `5/6/32`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `4/3`, mean margin delta `-0.1592446608202798`, mean loops `1.3030755051544733`
    - `conceptual_reasoning_proxy`: n `6`, delta `1`, W/L `1/0`, mean margin delta `0.35139918327331543`, mean loops `1.5904409090677898`
    - `deep_numeric_proxy`: n `9`, delta `-3`, W/L `0/3`, mean margin delta `-0.05124417278501722`, mean loops `1.413278533352746`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-6`, accuracy delta `-0.1395` (base `23/43`, recurrent `17/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `17` / `43`, base `23` / `43`, delta `-6`, W/L/T `0/6/37`, p `0.03125`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-2`, W/L `0/2`, mean margin delta `0.4895833323890757`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `-1`, W/L `0/1`, mean margin delta `-2.3203124891539724`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `-2`, W/L `0/2`, mean margin delta `0.5364583246409893`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `0.6406250135707003`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-1` (base `11/43`, recurrent `10/43`)
