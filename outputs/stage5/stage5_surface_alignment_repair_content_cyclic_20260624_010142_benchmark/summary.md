# Stage 5 Benchmark Suite - stage5_surface_alignment_repair_content_cyclic_20260624_010142_benchmark

- Status: `completed`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Checkpoint: `outputs/stage5/stage5_surface_alignment_repair_content_cyclic_20260624_010142/phase1_surface_align/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `562.32`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-8`, accuracy delta `-0.0312` (base `148/256`, recurrent `140/256`)
  - paired evidence
    - aggregate `mean`: recurrent `140` / `256`, base `148` / `256`, delta `-8`, W/L/T `6/14/236`, p `0.11531829833984375`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-3`, W/L `6/9`, mean margin delta `0.09218429104756501`, mean loops `1.1578159269134878`
    - `base_confident_direct_proxy`: n `69`, delta `-1`, W/L `0/1`, mean margin delta `0.026486848035584324`, mean loops `1.145136033538459`
    - `conceptual_reasoning_proxy`: n `38`, delta `-1`, W/L `0/1`, mean margin delta `-0.039613695521103706`, mean loops `1.3165785497740696`
    - `deep_numeric_proxy`: n `31`, delta `-3`, W/L `0/3`, mean margin delta `0.013663882209408668`, mean loops `1.327690270639235`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `5`, accuracy delta `0.0195` (base `201/256`, recurrent `206/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `206` / `256`, base `201` / `256`, delta `5`, W/L/T `6/1/249`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `61`, delta `2`, W/L `3/1`, mean margin delta `0.023657416191990258`, mean loops `None`
    - `base_confident_direct_proxy`: n `161`, delta `0`, W/L `0/0`, mean margin delta `-0.1155961045965347`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `3`, W/L `3/0`, mean margin delta `0.14809101460767643`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.08248527827672661`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0117` (base `87/256`, recurrent `90/256`)
  - paired evidence
    - aggregate `mean`: recurrent `90` / `256`, base `87` / `256`, delta `3`, W/L/T `9/6/241`, p `0.60723876953125`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `4/3`, mean margin delta `-0.11206717801287891`, mean loops `1.2091623540331677`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.14799223184585572`, mean loops `1.2401747035980224`
    - `conceptual_reasoning_proxy`: n `72`, delta `1`, W/L `4/3`, mean margin delta `-2.147671249177721e-05`, mean loops `1.3316727893220053`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `0.009582948353555467`, mean loops `1.2980305607672091`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-5`, accuracy delta `-0.0195` (base `156/256`, recurrent `151/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `151` / `256`, base `156` / `256`, delta `-5`, W/L/T `3/8/245`, p `0.2265625`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `-4`, W/L `1/5`, mean margin delta `-0.0072850814460941535`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.2592774666744474`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `0`, W/L `2/2`, mean margin delta `0.024972533300105067`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-1`, W/L `0/1`, mean margin delta `0.04942022317625357`, mean loops `None`
