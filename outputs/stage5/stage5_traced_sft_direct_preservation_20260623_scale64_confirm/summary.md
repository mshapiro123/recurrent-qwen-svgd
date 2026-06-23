# Stage 5 Benchmark Suite - stage5_traced_sft_direct_preservation_20260623_scale64_confirm

- Status: `completed`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/summary.json`
- Checkpoint: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/phase1_direct_preserve/phase1_step_75.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `257.49`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-8`, accuracy delta `-0.0312` (base `148/256`, recurrent `140/256`)
  - paired evidence
    - aggregate `mean`: recurrent `140` / `256`, base `148` / `256`, delta `-8`, W/L/T `8/16/232`, p `0.15158963203430176`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-4`, W/L `7/11`, mean margin delta `0.040444588257094564`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `69`, delta `-1`, W/L `0/1`, mean margin delta `0.06081729198711506`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `38`, delta `-1`, W/L `0/1`, mean margin delta `-0.08024169112506666`, mean loops `1.0`
    - `deep_numeric_proxy`: n `31`, delta `-2`, W/L `1/3`, mean margin delta `0.07222952573530135`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0078` (base `201/256`, recurrent `203/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `203` / `256`, base `201` / `256`, delta `2`, W/L/T `4/2/250`, p `0.6875`
  - routing buckets
    - `ambiguous_proxy`: n `61`, delta `0`, W/L `1/1`, mean margin delta `-0.12858606198397998`, mean loops `None`
    - `base_confident_direct_proxy`: n `161`, delta `0`, W/L `0/0`, mean margin delta `0.8674495333398047`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `2`, W/L `3/1`, mean margin delta `0.022321429635797228`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.17812499474966897`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0039` (base `87/256`, recurrent `86/256`)
  - paired evidence
    - aggregate `mean`: recurrent `86` / `256`, base `87` / `256`, delta `-1`, W/L/T `7/8/241`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `-4`, W/L `2/6`, mean margin delta `-0.16147422984363588`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.027710793018341066`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `72`, delta `2`, W/L `4/2`, mean margin delta `-0.04744171765115526`, mean loops `1.0`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `0.010629963543679979`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-5`, accuracy delta `-0.0195` (base `156/256`, recurrent `151/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `151` / `256`, base `156` / `256`, delta `-5`, W/L/T `3/8/245`, p `0.2265625`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-3`, W/L `1/4`, mean margin delta `-0.24131943849145354`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `0.7175347198075743`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `-0.4447916662294625`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `0/2`, mean margin delta `-0.1875000046989457`, mean loops `None`
