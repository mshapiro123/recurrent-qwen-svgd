# Stage 5 ARC-Mix Offset-Then-Depth Chain - stage5_arc_mix_offset_then_depth_chain_20260623_135452

- Status: `offset_not_confirmed`
- Passed offset: `False`
- Launched depth routing: `False`
- Offset summary: `outputs/stage5/stage5_arc_mix_offset_then_depth_chain_20260623_135452_offset256_confirm/summary.json`
- Depth summary: `not_run`
- Post-depth debiased summary: `not_run`
- Checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- Next step: Do not launch depth-routing SFT yet; diagnose the regressed offset readouts.

## Offset Evidence

- `arc_easy` `content_question_only` / `mean`: paired `256`, delta `10`, W/L/T `21/11/224`, passed `True`
- `arc_easy` `cyclic_label_aggregated` / `permutation_mean`: paired `256`, delta `-2`, W/L/T `2/4/250`, passed `False`
- `arc_challenge` `content_question_only` / `mean`: paired `43`, delta `0`, W/L/T `2/2/39`, passed `False`
- `arc_challenge` `cyclic_label_aggregated` / `permutation_mean`: paired `43`, delta `-1`, W/L/T `1/2/40`, passed `False`
