# Stage 5 Benchmark Suite - stage5_recovered_phase1_particles_arc_20260621_174231_rep2_k4_arc256

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `230.52`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-8`, accuracy delta `-0.0312` (base `148/256`, recurrent `140/256`)
  - aggregate `mean`: correct delta `-4`, accuracy delta `-0.0156` (base `148/256`, recurrent `144/256`)
  - aggregate `vote`: correct delta `-4`, accuracy delta `-0.0156` (base `148/256`, recurrent `144/256`)
  - paired evidence
    - aggregate `max`: recurrent `140` / `256`, base `148` / `256`, delta `-8`, W/L/T `12/20/224`, p `0.21532714972272515`
    - aggregate `mean`: recurrent `144` / `256`, base `148` / `256`, delta `-4`, W/L/T `12/16/228`, p `0.5715881884098053`
    - aggregate `vote`: recurrent `144` / `256`, base `148` / `256`, delta `-4`, W/L/T `13/17/226`, p `0.584664711728692`
