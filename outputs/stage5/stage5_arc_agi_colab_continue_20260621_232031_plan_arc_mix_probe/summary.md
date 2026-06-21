# Stage 5 Balanced ARC-Mix Gate - stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe

- Status: `proxy_lift`
- Passed: `True`
- Source summary: `outputs/stage5/stage5_benchmark_assessment_20260621_183952/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt`
- Mixed rows: `13227`
- Next step: Run full ARC-Easy and ARC-Challenge balanced benchmark on the best ARC-mix checkpoint.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `arc_mix_response_w005_lr2e6` | 67/128 | 66/128 | 68/128 | 1 | -1 | `outputs/stage5/stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe/arc_mix_response_w005_lr2e6/phase1/phase1_step_100.pt` |
