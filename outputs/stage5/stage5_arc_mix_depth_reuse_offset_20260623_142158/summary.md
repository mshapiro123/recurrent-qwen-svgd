# Stage 5 ARC-Mix Offset-Then-Depth Chain - stage5_arc_mix_depth_reuse_offset_20260623_142158

- Status: `depth_completed`
- Passed offset: `True`
- Launched depth routing: `True`
- Offset summary: `outputs/stage5/stage5_arc_mix_offset_then_depth_chain_20260623_135452_offset256_confirm/summary.json`
- Depth summary: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/summary.json`
- Post-depth debiased summary: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe_debiased_gate/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Next step: Review post-depth content and cyclic-debiased gate before extending training.

## Offset Evidence

- `arc_easy` `content_question_only` / `mean`: paired `256`, delta `10`, W/L/T `21/11/224`, passed `True`
- `arc_easy` `cyclic_label_aggregated` / `permutation_mean`: paired `256`, delta `-2`, W/L/T `2/4/250`, passed `True`
- `arc_challenge` `content_question_only` / `mean`: paired `43`, delta `0`, W/L/T `2/2/39`, passed `True`
- `arc_challenge` `cyclic_label_aggregated` / `permutation_mean`: paired `43`, delta `-1`, W/L/T `1/2/40`, passed `True`

## Post-Depth Debiased Evidence

Cyclic-debiased scoring is the primary survival gate here; content-question scoring is a leading indicator.

- `arc_easy` `content_question_only` / `mean`: paired `128`, delta `7`, W/L/T `11/4/113`, passed `True`
- `arc_easy` `cyclic_label_aggregated` / `permutation_mean`: paired `128`, delta `0`, W/L/T `1/1/126`, passed `True`
- `arc_challenge` `content_question_only` / `mean`: paired `43`, delta `0`, W/L/T `2/2/39`, passed `True`
- `arc_challenge` `cyclic_label_aggregated` / `permutation_mean`: paired `43`, delta `1`, W/L/T `1/0/42`, passed `True`
