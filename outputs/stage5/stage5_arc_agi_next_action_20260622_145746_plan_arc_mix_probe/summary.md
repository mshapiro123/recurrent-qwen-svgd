# Stage 5 Balanced ARC-Mix Gate - stage5_arc_agi_next_action_20260622_145746_plan_arc_mix_probe

- Status: `proxy_lift_calibration_warning`
- Passed: `False`
- Decision: `stop_for_calibration_repair`
- Blocked reason: Proxy lifted accuracy but failed the calibration-preservation threshold.
- Source summary: `outputs/stage5/stage5_arc_agi_next_plan_20260622_101547_benchmark_assessment/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Mixed rows: `13227`
- Proxy eval config: `ARC-Challenge`
- Calibration thresholds: mean margin delta >= `-0.05`, max prediction-count shift <= `16`
- Next step: Do not run the full paid assessment yet; the proxy lifted accuracy but degraded base-comparison calibration. Increase preservation/distillation or inspect answer-prior drift.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use label-only completions, so response-only distillation is concentrated on the option label decision rather than the prompt text.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w01_lr2e6` | 66/128 | 57/128 | 68/128 | 9 | -2 | -0.4325 | 32 | `warning` | `outputs/stage5/stage5_arc_agi_next_action_20260622_145746_plan_arc_mix_probe/arc_mix_response_w01_lr2e6/phase1/phase1_step_50.pt` |
