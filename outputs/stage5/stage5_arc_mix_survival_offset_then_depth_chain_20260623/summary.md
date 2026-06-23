# Stage 5 ARC-Mix Offset-Then-Depth Chain - stage5_arc_mix_survival_offset_then_depth_chain_20260623

- Status: `offset_not_confirmed`
- Passed offset: `False`
- Launched depth routing: `False`
- Offset summary: `outputs/stage5/stage5_arc_mix_survival_offset_then_depth_chain_20260623_offset256_confirm/summary.json`
- Depth summary: `not_run`
- Post-depth debiased summary: `not_run`
- Checkpoint: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Next step: Do not launch depth-routing SFT yet; diagnose the regressed offset readouts.

## Offset Evidence

- `arc_easy` `content_question_only` / `mean`: paired `256`, delta `11`, W/L/T `23/12/221`, passed `True`
- `arc_easy` `cyclic_label_aggregated` / `permutation_mean`: paired `256`, delta `-2`, W/L/T `1/3/252`, passed `False`
- `arc_challenge` `content_question_only` / `mean`: paired `43`, delta `-3`, W/L/T `2/5/36`, passed `False`
- `arc_challenge` `cyclic_label_aggregated` / `permutation_mean`: paired `43`, delta `1`, W/L/T `1/0/42`, passed `True`
