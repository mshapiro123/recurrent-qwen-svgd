# Stage 5 Benchmark Suite - stage5_full_assessment_once_20260622_005522_balanced_full

- Status: `completed`
- Source summary: `outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/arc_mix_response_w005_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `712.63`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-6`, accuracy delta `-0.0105` (base `421/570`, recurrent `415/570`)
  - paired evidence
    - aggregate `mean`: recurrent `415` / `570`, base `421` / `570`, delta `-6`, W/L/T `18/24/528`, p `0.44079906734259566`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0100` (base `167/299`, recurrent `164/299`)
  - paired evidence
    - aggregate `mean`: recurrent `164` / `299`, base `167` / `299`, delta `-3`, W/L/T `11/14/274`, p `0.6900379657745361`
