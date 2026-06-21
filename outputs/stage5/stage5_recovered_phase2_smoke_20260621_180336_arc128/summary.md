# Stage 5 Benchmark Suite - stage5_recovered_phase2_smoke_20260621_180336_arc128

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_recovered_phase2_smoke_20260621_180336/phase2/phase2_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `129.18`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-6`, accuracy delta `-0.0469` (base `72/128`, recurrent `66/128`)
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0234` (base `72/128`, recurrent `69/128`)
  - aggregate `vote`: correct delta `-4`, accuracy delta `-0.0312` (base `72/128`, recurrent `68/128`)
  - paired evidence
    - aggregate `max`: recurrent `66` / `128`, base `72` / `128`, delta `-6`, W/L/T `5/11/112`, p `0.210113525390625`
    - aggregate `mean`: recurrent `69` / `128`, base `72` / `128`, delta `-3`, W/L/T `6/9/113`, p `0.60723876953125`
    - aggregate `vote`: recurrent `68` / `128`, base `72` / `128`, delta `-4`, W/L/T `6/10/112`, p `0.454498291015625`
