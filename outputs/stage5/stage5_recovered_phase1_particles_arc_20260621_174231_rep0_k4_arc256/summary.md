# Stage 5 Benchmark Suite - stage5_recovered_phase1_particles_arc_20260621_174231_rep0_k4_arc256

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `231.11`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-9`, accuracy delta `-0.0352` (base `148/256`, recurrent `139/256`)
  - aggregate `mean`: correct delta `-12`, accuracy delta `-0.0469` (base `148/256`, recurrent `136/256`)
  - aggregate `vote`: correct delta `-13`, accuracy delta `-0.0508` (base `148/256`, recurrent `135/256`)
  - paired evidence
    - aggregate `max`: recurrent `139` / `256`, base `148` / `256`, delta `-9`, W/L/T `13/22/221`, p `0.17546524899080396`
    - aggregate `mean`: recurrent `136` / `256`, base `148` / `256`, delta `-12`, W/L/T `12/24/220`, p `0.06524533522315323`
    - aggregate `vote`: recurrent `135` / `256`, base `148` / `256`, delta `-13`, W/L/T `12/25/219`, p `0.04703102743951604`
