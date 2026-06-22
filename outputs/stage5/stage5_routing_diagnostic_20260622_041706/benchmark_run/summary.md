# Stage 5 Benchmark Suite - stage5_routing_diagnostic_20260622_041706_arc_easy64_challenge64

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `160.23`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0156` (base `49/64`, recurrent `48/64`)
  - paired evidence
    - aggregate `mean`: recurrent `48` / `64`, base `49` / `64`, delta `-1`, W/L/T `1/2/61`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `13`, delta `1`, W/L `1/0`, mean margin delta `0.7734979276473706`, mean loops `2.568508900128878`
    - `base_confident_direct_proxy`: n `41`, delta `-2`, W/L `0/2`, mean margin delta `-2.4873468239519108`, mean loops `2.5804901690017887`
    - `conceptual_reasoning_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.5295848071575164`, mean loops `2.602752709388733`
    - `deep_numeric_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.19665260314941407`, mean loops `2.6235814332962035`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0312` (base `36/64`, recurrent `38/64`)
  - paired evidence
    - aggregate `mean`: recurrent `38` / `64`, base `36` / `64`, delta `2`, W/L/T `7/5/52`, p `0.7744140625`
  - routing buckets
    - `ambiguous_proxy`: n `17`, delta `3`, W/L `4/1`, mean margin delta `0.9119988376384273`, mean loops `2.5651468739790073`
    - `base_confident_direct_proxy`: n `27`, delta `-3`, W/L `0/3`, mean margin delta `-2.016581752234035`, mean loops `2.61728823405725`
    - `conceptual_reasoning_proxy`: n `12`, delta `2`, W/L `3/1`, mean margin delta `0.30147182444731396`, mean loops `2.603570267558098`
    - `deep_numeric_proxy`: n `8`, delta `0`, W/L `0/0`, mean margin delta `0.6717620473355055`, mean loops `2.5764594773451486`
