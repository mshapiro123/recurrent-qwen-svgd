# Stage 5 Balanced ARC-Mix Gate - stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe

- Status: `proxy_matches_base`
- Passed: `True`
- Decision: `run_full_balanced_assessment`
- Blocked reason: none
- Source summary: `outputs/stage5/stage5_arc_mix_offset_then_depth_chain_20260623_135452_offset256_confirm/summary.json`
- Resume checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- Mixed rows: `15744`
- Proxy eval config: `ARC-Challenge`
- Loop control: `{'use_target_loop_control': False, 'use_learned_loop_control': True, 'eval_use_learned_loop_control': True, 'loop_control_ce_weight': 0.05, 'halt_target_nll_weight': 0.03, 'optimizer_modules': 'all'}`
- Calibration thresholds: mean margin delta >= `-0.05`, max prediction-count shift <= `16`
- Next step: Run full balanced benchmark on the best ARC-mix checkpoint; proxy no longer trails base.

## Objective Rationale

- Failure mode: The previous proxy-selected recurrent checkpoint lost full balanced ARC points through answer-calibration drift: lower correct-answer margins and answer-prior shift.
- Proxy hypothesis: Mix Opus reasoning traces with ARC-style MCQ label supervision and use response-only frozen-base KL distillation to preserve the base model's answer-token distribution.
- Response distillation reason: ARC MCQ SFT rows use short answer-surface completions, so response-only distillation is concentrated on the evaluated choice surface rather than the prompt text.
- Depth routing reason: When target_loop_count metadata is present, optional learned loop-control CE and halting target NLL let the same ARC-mix runner test whether direct/easy rows can stay shallow while harder rows receive more recurrent computation.

## Arms

| arm | best proxy | start proxy | base proxy | lift vs start | gap vs base | margin vs base | max pred shift | calibration | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `arc_mix_response_w02_lr2e6` | 54/128 | 55/128 | 43/128 | -1 | 11 | 0.1244 | 3 | `ok` | `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt` |
