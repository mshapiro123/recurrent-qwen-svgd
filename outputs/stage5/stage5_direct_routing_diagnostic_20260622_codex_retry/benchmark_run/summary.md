# Stage 5 Benchmark Suite - stage5_direct_routing_diagnostic_20260622_codex_retry_arc_easy64_challenge64

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_145746_plan_arc_mix_probe/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_145746_plan_arc_mix_probe/arc_mix_response_w01_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `164.30`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-8`, accuracy delta `-0.1250` (base `49/64`, recurrent `41/64`)
  - paired evidence
    - aggregate `mean`: recurrent `41` / `64`, base `49` / `64`, delta `-8`, W/L/T `1/9/54`, p `0.021484375`
  - routing buckets
    - `ambiguous_proxy`: n `13`, delta `-1`, W/L `0/1`, mean margin delta `0.7533253179146693`, mean loops `3.0468215208787184`
    - `base_confident_direct_proxy`: n `41`, delta `-6`, W/L `0/6`, mean margin delta `-2.880388898786339`, mean loops `3.0519139545719796`
    - `conceptual_reasoning_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.4061653435230255`, mean loops `3.0712791204452516`
    - `deep_numeric_proxy`: n `5`, delta `-1`, W/L `1/2`, mean margin delta `0.04832202196121216`, mean loops `3.0587233543395995`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0312` (base `36/64`, recurrent `34/64`)
  - paired evidence
    - aggregate `mean`: recurrent `34` / `64`, base `36` / `64`, delta `-2`, W/L/T `7/9/48`, p `0.803619384765625`
  - routing buckets
    - `ambiguous_proxy`: n `17`, delta `5`, W/L `6/1`, mean margin delta `1.134664421024568`, mean loops `3.0500607478852366`
    - `base_confident_direct_proxy`: n `27`, delta `-6`, W/L `0/6`, mean margin delta `-2.167671926043652`, mean loops `3.0667937420032643`
    - `conceptual_reasoning_proxy`: n `12`, delta `0`, W/L `1/1`, mean margin delta `0.2755796213944753`, mean loops `3.0668024818102517`
    - `deep_numeric_proxy`: n `8`, delta `-1`, W/L `0/1`, mean margin delta `0.5736455898731947`, mean loops `3.062690019607544`
