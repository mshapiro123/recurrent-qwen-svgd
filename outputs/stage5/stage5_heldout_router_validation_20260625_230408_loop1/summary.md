# Stage 5 Benchmark Suite - stage5_heldout_router_validation_20260625_230408_loop1

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'open_hard_arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1099.18`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0391` (base `71/128`, recurrent `76/128`)
  - paired evidence
    - aggregate `mean`: recurrent `76` / `128`, base `71` / `128`, delta `5`, W/L/T `8/3/117`, p `0.2265625`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `2`, W/L `5/3`, mean margin delta `-0.010016236305236816`, mean loops `1.1489684653282166`
    - `base_confident_direct_proxy`: n `42`, delta `0`, W/L `0/0`, mean margin delta `-0.0025824719951266333`, mean loops `1.132487420051817`
    - `conceptual_reasoning_proxy`: n `24`, delta `3`, W/L `3/0`, mean margin delta `0.02905149261156718`, mean loops `1.2210755720734596`
    - `deep_numeric_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.12991270422935486`, mean loops `1.252320056160291`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `103/128`, recurrent `103/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `103` / `128`, base `103` / `128`, delta `0`, W/L/T `1/1/126`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `-0.06875000331550836`, mean loops `None`
    - `base_confident_direct_proxy`: n `86`, delta `0`, W/L `0/0`, mean margin delta `0.6560683066306601`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `13`, delta `1`, W/L `1/0`, mean margin delta `-0.13701924968224305`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `-0.3906249850988388`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `11/43`, recurrent `11/43`)
  - paired evidence
    - aggregate `mean`: recurrent `11` / `43`, base `11` / `43`, delta `0`, W/L/T `1/1/41`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `1/0`, mean margin delta `-0.003147942679268973`, mean loops `1.14441137441567`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.045731425285339355`, mean loops `1.3369924773772557`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.08490880330403645`, mean loops `1.2178230384985607`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `23/43`, recurrent `23/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `23` / `43`, base `23` / `43`, delta `0`, W/L/T `1/1/41`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `0/0`, mean margin delta `-0.17361108410275644`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `0.55078123374066`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `-0.3645833313154678`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `0`, W/L `1/1`, mean margin delta `-0.18749998084136418`, mean loops `None`
### open_hard_arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0078` (base `39/128`, recurrent `40/128`)
  - paired evidence
    - aggregate `mean`: recurrent `40` / `128`, base `39` / `128`, delta `1`, W/L/T `2/1/125`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `1`, W/L `1/0`, mean margin delta `-0.0043204352259635925`, mean loops `1.1505627655424178`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `0.025172765056292217`, mean loops `1.1765334034959476`
    - `conceptual_reasoning_proxy`: n `32`, delta `0`, W/L `1/1`, mean margin delta `-0.1362035870552063`, mean loops `1.2704503564164042`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.13428900241851807`, mean loops `1.3554657042026519`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-6`, accuracy delta `-0.0469` (base `75/128`, recurrent `69/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `69` / `128`, base `75` / `128`, delta `-6`, W/L/T `1/7/120`, p `0.0703125`
  - routing buckets
    - `ambiguous_proxy`: n `40`, delta `-4`, W/L `0/4`, mean margin delta `-0.3617187400115654`, mean loops `None`
    - `base_confident_direct_proxy`: n `56`, delta `0`, W/L `0/0`, mean margin delta `0.6869419662793267`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `-0.23671872501727192`, mean loops `None`
    - `deep_numeric_proxy`: n `12`, delta `-3`, W/L `0/3`, mean margin delta `-0.14322915316248933`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `0` (base `11/43`, recurrent `11/43`)
- `open_hard_arc_challenge` `content_question_only`: delta `1` (base `39/128`, recurrent `40/128`)
