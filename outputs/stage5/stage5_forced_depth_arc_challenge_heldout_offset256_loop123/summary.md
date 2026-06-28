# Stage 5 Forced Depth Diagnostic - stage5_forced_depth_arc_challenge_heldout_offset256_loop123

- Cell version: `forced_depth_arc_v1`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Loops: `[1, 2, 3]`
- Forward max loops: `3`
- Benchmarks: `arc_challenge`
- Score targets: `content_question_only,cyclic_label_aggregated`

## Loop Runs

### stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop1
- Forced loop count: `1`
- Status: `completed`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `11/43`, delta `0`, W/L/T `0/0/43`, p `None`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `24/43`, delta `1`, W/L/T `1/0/42`, p `1.0`

### stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop2
- Forced loop count: `2`
- Status: `completed`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `8/43`, delta `-3`, W/L/T `1/4/38`, p `0.375`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `21/43`, delta `-2`, W/L/T `0/2/41`, p `0.5`

### stage5_forced_depth_arc_challenge_heldout_offset256_loop123_loop3
- Forced loop count: `3`
- Status: `completed`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `10/43`, delta `-1`, W/L/T `5/6/32`, p `1.0`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `17/43`, delta `-6`, W/L/T `0/6/37`, p `0.03125`
