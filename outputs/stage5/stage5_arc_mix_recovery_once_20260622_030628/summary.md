# Stage 5 Balanced ARC-Mix Gate - stage5_arc_mix_recovery_once_20260622_030628

- Status: `no_proxy_lift`
- Passed: `False`
- Decision: `stop_and_revise_objective`
- Blocked reason: Best ARC-mix checkpoint did not improve over the recurrent start or close the base gap.
- Source summary: `outputs/stage5/stage5_full_assessment_once_20260622_005522/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/arc_mix_response_w005_lr2e6/phase1/phase1_step_50.pt`
- Mixed rows: `13227`
- Calibration thresholds: mean margin delta >= `-0.05`, max prediction-count shift <= `16`
- Next step: Do not extend this ARC-mix setting; inspect failures or revise supervision mix.
- Drive backup: `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5_arc_mix_recovery_once_20260622_030628`
- Colab push note: the runner committed locally in Colab but `git push` was rejected because a notebook autosave commit reached `main` first; this compact summary preserves the planner-relevant result.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use label-only completions, so response-only distillation is concentrated on the option label decision rather than the prompt text.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w01_lr2e6` | 66/128 | 68/128 | 68/128 | -2 | -2 | -0.3082 | 10 | `warning` | `outputs/stage5/stage5_arc_mix_recovery_once_20260622_030628/arc_mix_response_w01_lr2e6/phase1/phase1_step_50.pt` |

## Review

Next A100 spend: **NO: stop A100 work and repair locally.**
