# Stage 5 Benchmark Suite - stage5_arc64_phase2_k4_no_svgd_20260621_160716

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc64_phase2_svgd_k4_20260621_160421/summary.json`
- Checkpoint: `outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `76.89`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-4`, accuracy delta `-0.0625` (base `36/64`, recurrent `32/64`)
  - aggregate `mean`: correct delta `-4`, accuracy delta `-0.0625` (base `36/64`, recurrent `32/64`)
  - aggregate `vote`: correct delta `-4`, accuracy delta `-0.0625` (base `36/64`, recurrent `32/64`)
  - paired evidence
    - aggregate `max`: recurrent `32` / `64`, base `36` / `64`, delta `-4`, W/L/T `7/11/46`, p `0.480682373046875`
    - aggregate `mean`: recurrent `32` / `64`, base `36` / `64`, delta `-4`, W/L/T `7/11/46`, p `0.480682373046875`
    - aggregate `vote`: recurrent `32` / `64`, base `36` / `64`, delta `-4`, W/L/T `7/11/46`, p `0.480682373046875`
