# Stage 5 Balanced ARC-Mix Gate - stage5_balanced_recovery_autopilot_current_arc_mix

- Status: `proxy_lift`
- Passed: `True`
- Source summary: `outputs/stage5/stage5_balanced_mcq_current/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt`
- Mixed rows: `9380`
- Next step: Run full ARC-Easy and ARC-Challenge balanced benchmark on the best ARC-mix checkpoint.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `arc_mix_nodistill_lr3e6` | 66/128 | 65/128 | 68/128 | 1 | -2 | `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt` |
