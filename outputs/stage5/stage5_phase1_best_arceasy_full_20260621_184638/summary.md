# Stage 5 Benchmark Suite - stage5_phase1_best_arceasy_full_20260621_184638

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt`
- Benchmarks: `['arc_easy']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `459.29`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-12`, accuracy delta `-0.0211` (base `421/570`, recurrent `409/570`)
  - paired evidence
    - aggregate `mean`: recurrent `409` / `570`, base `421` / `570`, delta `-12`, W/L/T `15/27/528`, p `0.08842954698775429`
