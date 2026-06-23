# Stage 5 Benchmark Suite - stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe_debiased_gate

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `777.11`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `7`, accuracy delta `0.0547` (base `71/128`, recurrent `78/128`)
  - paired evidence
    - aggregate `mean`: recurrent `78` / `128`, base `71` / `128`, delta `7`, W/L/T `11/4/113`, p `0.11846923828125`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `7`, W/L `7/0`, mean margin delta `0.36007857978343966`, mean loops `1.6632148027420044`
    - `base_confident_direct_proxy`: n `42`, delta `0`, W/L `0/0`, mean margin delta `-0.524293045202891`, mean loops `1.6813113143046696`
    - `conceptual_reasoning_proxy`: n `24`, delta `2`, W/L `4/2`, mean margin delta `0.16657855610052744`, mean loops `1.7636001221835613`
    - `deep_numeric_proxy`: n `12`, delta `-2`, W/L `0/2`, mean margin delta `0.16870533426602682`, mean loops `1.7354888245463371`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `103/128`, recurrent `103/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `103` / `128`, base `103` / `128`, delta `0`, W/L/T `1/1/126`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `0.08992770347744226`, mean loops `None`
    - `base_confident_direct_proxy`: n `86`, delta `0`, W/L `0/0`, mean margin delta `-1.4056450073571618`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `13`, delta `1`, W/L `1/0`, mean margin delta `0.2523921292561751`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.05633188504725695`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `11/43`, recurrent `11/43`)
  - paired evidence
    - aggregate `mean`: recurrent `11` / `43`, base `11` / `43`, delta `0`, W/L/T `2/2/39`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `2/1`, mean margin delta `0.05360151827335358`, mean loops `1.675967806151935`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.029336750507354736`, mean loops `1.87147556245327`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.04802913798226251`, mean loops `1.6482947402530246`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0233` (base `23/43`, recurrent `24/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `24` / `43`, base `23` / `43`, delta `1`, W/L/T `1/0/42`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `0/0`, mean margin delta `0.1647793465009373`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.6246910246554762`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.19162410326922932`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `1`, W/L `1/0`, mean margin delta `0.29689507226326634`, mean loops `None`
