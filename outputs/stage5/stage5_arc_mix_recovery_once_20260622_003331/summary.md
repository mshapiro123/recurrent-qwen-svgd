# Stage 5 Balanced ARC-Mix Gate - stage5_arc_mix_recovery_once_20260622_003331

- Status: `proxy_lift`
- Passed: `True`
- Source summary: `outputs/stage5/stage5_recovery_full_assessment_current/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt`
- Mixed rows: `13227`
- Next step: Run full ARC-Easy and ARC-Challenge balanced benchmark on the best ARC-mix checkpoint.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `arc_mix_response_w005_lr2e6` | 68/128 | 66/128 | 68/128 | 2 | 0 | `outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/arc_mix_response_w005_lr2e6/phase1/phase1_step_50.pt` |
