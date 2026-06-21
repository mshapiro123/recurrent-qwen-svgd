# Stage 5 Benchmark Suite - stage5_recovery_full_assessment_current_balanced_full

- Status: `completed`
- Source summary: `outputs/stage5/stage5_balanced_recovery_autopilot_current/summary.json`
- Checkpoint: `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `700.09`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `-9`, accuracy delta `-0.0158` (base `421/570`, recurrent `412/570`)
  - paired evidence
    - aggregate `mean`: recurrent `412` / `570`, base `421` / `570`, delta `-9`, W/L/T `20/29/521`, p `0.2528697301676033`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0067` (base `167/299`, recurrent `169/299`)
  - paired evidence
    - aggregate `mean`: recurrent `169` / `299`, base `167` / `299`, delta `2`, W/L/T `13/11/275`, p `0.8388197422027588`
