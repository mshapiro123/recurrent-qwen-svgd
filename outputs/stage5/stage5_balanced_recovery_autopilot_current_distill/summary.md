# Stage 5 Balanced Distillation Gate - stage5_balanced_recovery_autopilot_current_distill

- Status: `no_proxy_lift`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_balanced_mcq_current/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt`
- ARC proxy limit: `128`
- Next step: Do not extend this distillation setting. Try a mixed ARC-train supervision gate or revisit training data before particle/SVGD work.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `response_w005_lr3e6` | 71/128 | 71/128 | 72/128 | 0 | -1 | `outputs/stage5/stage5_balanced_recovery_autopilot_current_distill_response_w005_lr3e6/phase1/phase1_step_100.pt` |
