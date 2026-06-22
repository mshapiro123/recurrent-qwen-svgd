# Stage 5 Balanced ARC-Mix Gate - stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation

- Status: `no_proxy_lift`
- Passed: `False`
- Decision: `stop_and_revise_objective`
- Blocked reason: Best ARC-mix checkpoint did not improve over the recurrent start or close the base gap.
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft_routing_diagnostic/benchmark_run/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Mixed rows: `11121`
- Proxy eval config: `ARC-Easy`
- Calibration thresholds: mean margin delta >= `0.0`, max prediction-count shift <= `8`
- Next step: Do not extend this ARC-mix setting; inspect failures or revise supervision mix.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use label-only completions, so response-only distillation is concentrated on the option label decision rather than the prompt text.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w05_lr1e6` | 81/128 | 82/128 | 87/128 | -1 | -6 | -1.5245 | 20 | `warning` | `outputs/stage5/stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/arc_mix_response_w05_lr1e6/phase1/phase1_step_100.pt` |
