# Stage 5 Benchmark Suite - stage5_arc128_phase2_svgd_k4_20260621_160916

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc64_phase2_k4_no_svgd_20260621_160716/summary.json`
- Checkpoint: `outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase2`
- Recurrent trajectories: `4`
- Elapsed seconds: `127.45`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `max`: correct delta `-5`, accuracy delta `-0.0391` (base `72/128`, recurrent `67/128`)
  - aggregate `mean`: correct delta `-6`, accuracy delta `-0.0469` (base `72/128`, recurrent `66/128`)
  - aggregate `vote`: correct delta `-5`, accuracy delta `-0.0391` (base `72/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `max`: recurrent `67` / `128`, base `72` / `128`, delta `-5`, W/L/T `1/6/121`, p `0.125`
    - aggregate `mean`: recurrent `66` / `128`, base `72` / `128`, delta `-6`, W/L/T `1/7/120`, p `0.0703125`
    - aggregate `vote`: recurrent `67` / `128`, base `72` / `128`, delta `-5`, W/L/T `2/7/119`, p `0.1796875`
