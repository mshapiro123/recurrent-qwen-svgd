# Stage 5 Benchmark Suite - stage5_recovered_phase1_arc256_20260621_172908

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `218.18`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0078` (base `148/256`, recurrent `146/256`)
  - paired evidence
    - aggregate `mean`: recurrent `146` / `256`, base `148` / `256`, delta `-2`, W/L/T `24/26/206`, p `0.887724827340783`
