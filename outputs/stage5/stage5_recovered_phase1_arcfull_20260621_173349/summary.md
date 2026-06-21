# Stage 5 Benchmark Suite - stage5_recovered_phase1_arcfull_20260621_173349

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `249.87`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `1`, accuracy delta `0.0033` (base `167/299`, recurrent `168/299`)
  - paired evidence
    - aggregate `mean`: recurrent `168` / `299`, base `167` / `299`, delta `1`, W/L/T `29/28/242`, p `1.0`
