# Stage 5 Benchmark Suite - stage5_arc_challenge_128_20260621_155723

- Status: `completed`
- Source summary: `outputs/stage5/stage5_mcq_gate_20260621_155430/summary.json`
- Checkpoint: `outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `124.74`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0156` (base `72/128`, recurrent `70/128`)
  - paired evidence
    - aggregate `mean`: recurrent `70` / `128`, base `72` / `128`, delta `-2`, W/L/T `17/19/92`, p `0.8679394004284404`
