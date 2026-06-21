# Stage 5 Benchmark Suite - stage5_recovered_phase2_smoke_20260621_175538_arc128

- Status: `completed`
- Source summary: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/summary.json`
- Checkpoint: `outputs/stage5/stage5_recovered_phase2_smoke_20260621_175538/phase2/phase2_step_25.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `129.58`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-4`, accuracy delta `-0.0312` (base `72/128`, recurrent `68/128`)
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0156` (base `72/128`, recurrent `70/128`)
  - aggregate `vote`: correct delta `-5`, accuracy delta `-0.0391` (base `72/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `max`: recurrent `68` / `128`, base `72` / `128`, delta `-4`, W/L/T `7/11/110`, p `0.480682373046875`
    - aggregate `mean`: recurrent `70` / `128`, base `72` / `128`, delta `-2`, W/L/T `7/9/112`, p `0.803619384765625`
    - aggregate `vote`: recurrent `67` / `128`, base `72` / `128`, delta `-5`, W/L/T `6/11/111`, p `0.332305908203125`
