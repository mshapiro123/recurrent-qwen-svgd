# Stage 5 Balanced ARC-Mix Gate - stage5_arc_agi_next_action_20260622_173328_plan_routing_repair_direct_halting_arc_mix

- Status: `proxy_lift_calibration_warning`
- Passed: `False`
- Decision: `stop_for_calibration_repair`
- Blocked reason: Proxy lifted accuracy but failed the calibration-preservation threshold.
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft_routing_diagnostic/benchmark_run/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Mixed rows: `20125`
- Proxy eval config: `ARC-Easy`
- Calibration thresholds: mean margin delta >= `0.0`, max prediction-count shift <= `8`
- Next step: Do not run the full paid assessment yet; the proxy lifted accuracy but degraded base-comparison calibration. Increase preservation/distillation or inspect answer-prior drift.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use label-only completions, so response-only distillation is concentrated on the option label decision rather than the prompt text.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w02_lr2e6` | 84/128 | 82/128 | 87/128 | 2 | -3 | -1.5560 | 23 | `warning` | `outputs/stage5/stage5_arc_agi_next_action_20260622_173328_plan_routing_repair_direct_halting_arc_mix/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt` |
