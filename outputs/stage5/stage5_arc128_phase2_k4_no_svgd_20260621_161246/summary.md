# Stage 5 Benchmark Suite - stage5_arc128_phase2_k4_no_svgd_20260621_161246

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc128_phase2_svgd_k4_20260621_160916/summary.json`
- Checkpoint: `outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `126.11`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-9`, accuracy delta `-0.0703` (base `72/128`, recurrent `63/128`)
  - aggregate `mean`: correct delta `-9`, accuracy delta `-0.0703` (base `72/128`, recurrent `63/128`)
  - aggregate `vote`: correct delta `-9`, accuracy delta `-0.0703` (base `72/128`, recurrent `63/128`)
  - paired evidence
    - aggregate `max`: recurrent `63` / `128`, base `72` / `128`, delta `-9`, W/L/T `16/25/87`, p `0.21102359760880063`
    - aggregate `mean`: recurrent `63` / `128`, base `72` / `128`, delta `-9`, W/L/T `16/25/87`, p `0.21102359760880063`
    - aggregate `vote`: recurrent `63` / `128`, base `72` / `128`, delta `-9`, W/L/T `16/25/87`, p `0.21102359760880063`
