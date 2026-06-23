# Stage 5 Balanced ARC-Mix Gate - stage5_content_arcmix_qonly_optiontext_20260623_121707

- Status: `proxy_lift`
- Passed: `True`
- Decision: `run_full_balanced_assessment`
- Blocked reason: none
- Source summary: `outputs/stage5/stage5_content_direct_preserve_m05_arc256_check_20260623_113424/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_content_direct_preserve_m05_20260623_1122/phase1_direct_preserve/phase1_step_150.pt`
- Mixed rows: `6144`
- Proxy eval config: `ARC-Easy`
- Calibration thresholds: mean margin delta >= `-0.2`, max prediction-count shift <= `20`
- Next step: Run full ARC-Easy and ARC-Challenge balanced benchmark on the best ARC-mix checkpoint.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use short answer-surface completions, so response-only distillation is concentrated on the evaluated choice surface rather than the prompt text.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w02_lr2e6` | 78/128 | 70/128 | 74/128 | 8 | 4 | 0.1078 | 4 | `ok` | `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt` |
| `arc_mix_response_w05_lr1e6` | 73/128 | 70/128 | 74/128 | 3 | -1 | 0.0008 | 4 | `ok` | `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w05_lr1e6/phase1/phase1_step_100.pt` |
