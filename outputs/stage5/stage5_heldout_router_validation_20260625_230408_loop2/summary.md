# Stage 5 Benchmark Suite - stage5_heldout_router_validation_20260625_230408_loop2

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'open_hard_arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `3`
- Forced loop count: `2`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1104.94`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `4`, accuracy delta `0.0312` (base `71/128`, recurrent `75/128`)
  - paired evidence
    - aggregate `mean`: recurrent `75` / `128`, base `71` / `128`, delta `4`, W/L/T `10/6/112`, p `0.454498291015625`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `3`, W/L `4/1`, mean margin delta `-0.047661097049713136`, mean loops `1.1489684653282166`
    - `base_confident_direct_proxy`: n `42`, delta `-1`, W/L `0/1`, mean margin delta `-1.319625498283477`, mean loops `1.132487420051817`
    - `conceptual_reasoning_proxy`: n `24`, delta `3`, W/L `5/2`, mean margin delta `0.08096881707509358`, mean loops `1.2210755720734596`
    - `deep_numeric_proxy`: n `12`, delta `-1`, W/L `1/2`, mean margin delta `-0.01915992299715678`, mean loops `1.252320056160291`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0078` (base `103/128`, recurrent `104/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `104` / `128`, base `103` / `128`, delta `1`, W/L/T `3/2/123`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `0`, W/L `2/2`, mean margin delta `0.29250000301748513`, mean loops `None`
    - `base_confident_direct_proxy`: n `86`, delta `0`, W/L `0/0`, mean margin delta `-3.030644383354793`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `13`, delta `1`, W/L `1/0`, mean margin delta `0.4110576782662135`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.21093751769512892`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0698` (base `11/43`, recurrent `8/43`)
  - paired evidence
    - aggregate `mean`: recurrent `8` / `43`, base `11` / `43`, delta `-3`, W/L/T `2/5/36`, p `0.453125`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `-1`, W/L `2/3`, mean margin delta `-0.06565820106438228`, mean loops `1.14441137441567`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.10973616441090901`, mean loops `1.3369924773772557`
    - `deep_numeric_proxy`: n `9`, delta `-2`, W/L `0/2`, mean margin delta `-0.05655450291103787`, mean loops `1.2178230384985607`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-3`, accuracy delta `-0.0698` (base `23/43`, recurrent `20/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `20` / `43`, base `23` / `43`, delta `-3`, W/L/T `0/3/40`, p `0.25`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `0/1`, mean margin delta `0.39583333996900666`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-1.7669270876407002`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `-1`, W/L `0/1`, mean margin delta `0.33333334466442466`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `-1`, W/L `0/1`, mean margin delta `0.5379464295027512`, mean loops `None`
### open_hard_arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0156` (base `39/128`, recurrent `37/128`)
  - paired evidence
    - aggregate `mean`: recurrent `37` / `128`, base `39` / `128`, delta `-2`, W/L/T `9/11/108`, p `0.8238029479980469`
  - routing buckets
    - `ambiguous_proxy`: n `64`, delta `1`, W/L `4/3`, mean margin delta `0.04421524237841368`, mean loops `1.1505627655424178`
    - `base_confident_direct_proxy`: n `12`, delta `-3`, W/L `0/3`, mean margin delta `-0.9459260900815328`, mean loops `1.1765334034959476`
    - `conceptual_reasoning_proxy`: n `32`, delta `2`, W/L `4/2`, mean margin delta `0.029644418507814407`, mean loops `1.2704503564164042`
    - `deep_numeric_proxy`: n `20`, delta `-2`, W/L `1/3`, mean margin delta `-0.007185646891593933`, mean loops `1.3554657042026519`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `75/128`, recurrent `75/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `75` / `128`, base `75` / `128`, delta `0`, W/L/T `2/2/124`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `40`, delta `0`, W/L `1/1`, mean margin delta `0.7125000042608007`, mean loops `None`
    - `base_confident_direct_proxy`: n `56`, delta `0`, W/L `0/0`, mean margin delta `-2.4056919581510425`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.7546875159954652`, mean loops `None`
    - `deep_numeric_proxy`: n `12`, delta `-1`, W/L `0/1`, mean margin delta `0.1432291722546021`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-3` (base `11/43`, recurrent `8/43`)
- `open_hard_arc_challenge` `content_question_only`: delta `-2` (base `39/128`, recurrent `37/128`)
