# Stage 5 Balanced ARC-Mix Gate - stage5_competence_recovery_from_reentry_benchmark_arc_mix

- Status: `proxy_lift_calibration_warning`
- Passed: `False`
- Decision: `stop_for_calibration_repair`
- Blocked reason: Proxy lifted accuracy but failed the calibration-preservation threshold.
- Source summary: `outputs/stage5/stage5_debiased_benchmark_assessment_20260625_121302/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_114836_curriculum_sft/phase1/phase1_step_75.pt`
- Mixed rows: `13882`
- Proxy eval config: `ARC-Challenge`
- Loop control: `{'use_target_loop_control': False, 'use_learned_loop_control': False, 'eval_use_learned_loop_control': False, 'loop_control_ce_weight': 0.0, 'halt_target_nll_weight': 0.0, 'optimizer_modules': 'all'}`
- Calibration thresholds: mean margin delta >= `-0.05`, max prediction-count shift <= `16`
- Next step: Do not run the full paid assessment yet; the proxy lifted accuracy but degraded base-comparison calibration. Increase preservation/distillation or inspect answer-prior drift.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use short answer-surface completions, so response-only distillation is concentrated on the evaluated choice surface rather than the prompt text.
- Depth routing reason: When target_loop_count metadata is present, optional learned loop-control CE and halting target NLL let the same ARC-mix runner test whether direct/easy rows can stay shallow while harder rows receive more recurrent computation.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w02_lr2e6` | 72/128 | 67/128 | 69/128 | 5 | 3 | -0.0694 | 6 | `warning` | `outputs/stage5/stage5_competence_recovery_from_reentry_benchmark_arc_mix/arc_mix_response_w02_lr2e6/phase1/phase1_step_100.pt` |
| `arc_mix_response_w01_lr2e6` | 71/128 | 67/128 | 69/128 | 4 | 2 | -0.0574 | 6 | `warning` | `outputs/stage5/stage5_competence_recovery_from_reentry_benchmark_arc_mix/arc_mix_response_w01_lr2e6/phase1/phase1_step_150.pt` |
