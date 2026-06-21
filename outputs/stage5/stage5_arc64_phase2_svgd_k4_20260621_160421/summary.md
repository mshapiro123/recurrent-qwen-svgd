# Stage 5 Benchmark Suite - stage5_arc64_phase2_svgd_k4_20260621_160421

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_challenge_128_20260621_155723/summary.json`
- Checkpoint: `outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `78.27`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-2`, accuracy delta `-0.0312` (base `36/64`, recurrent `34/64`)
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0156` (base `36/64`, recurrent `35/64`)
  - aggregate `vote`: correct delta `0`, accuracy delta `0.0000` (base `36/64`, recurrent `36/64`)
  - paired evidence
    - aggregate `max`: recurrent `34` / `64`, base `36` / `64`, delta `-2`, W/L/T `1/3/60`, p `0.625`
    - aggregate `mean`: recurrent `35` / `64`, base `36` / `64`, delta `-1`, W/L/T `1/2/61`, p `1.0`
    - aggregate `vote`: recurrent `36` / `64`, base `36` / `64`, delta `0`, W/L/T `2/2/60`, p `1.0`
