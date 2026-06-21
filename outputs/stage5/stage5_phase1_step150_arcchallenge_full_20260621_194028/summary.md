# Stage 5 Benchmark Suite - stage5_phase1_step150_arcchallenge_full_20260621_194028

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arceasy_sweep_full_20260621_185841/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `252.95`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0067` (base `167/299`, recurrent `169/299`)
  - paired evidence
    - aggregate `mean`: recurrent `169` / `299`, base `167` / `299`, delta `2`, W/L/T `24/22/253`, p `0.8829959121223965`
