# Stage 5 Held-Out Router Validation Sweep - stage5_heldout_router_validation_20260625_230408

- Cell version: `heldout_router_validation_v1`
- Discovery sweep: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Loops: `[1, 2, 3]`
- Forward max loops: `3`
- Benchmarks: `arc_easy,arc_challenge,open_hard_arc_challenge`
- Score targets: `content_question_only,cyclic_label_aggregated`

## stage5_heldout_router_validation_20260625_230408_loop1
- Forced loop count: `1`
- Status: `completed`
- arc_easy `content_question_only/mean`: base `71/128`, recurrent `76/128`, delta `5`, W/L/T `8/3/117`, p `0.2265625`
- arc_easy `cyclic_label_aggregated/permutation_mean`: base `103/128`, recurrent `103/128`, delta `0`, W/L/T `1/1/126`, p `1.0`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `11/43`, delta `0`, W/L/T `1/1/41`, p `1.0`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `23/43`, delta `0`, W/L/T `1/1/41`, p `1.0`
- open_hard_arc_challenge `content_question_only/mean`: base `39/128`, recurrent `40/128`, delta `1`, W/L/T `2/1/125`, p `1.0`
- open_hard_arc_challenge `cyclic_label_aggregated/permutation_mean`: base `75/128`, recurrent `69/128`, delta `-6`, W/L/T `1/7/120`, p `0.0703125`

## stage5_heldout_router_validation_20260625_230408_loop2
- Forced loop count: `2`
- Status: `completed`
- arc_easy `content_question_only/mean`: base `71/128`, recurrent `75/128`, delta `4`, W/L/T `10/6/112`, p `0.454498291015625`
- arc_easy `cyclic_label_aggregated/permutation_mean`: base `103/128`, recurrent `104/128`, delta `1`, W/L/T `3/2/123`, p `1.0`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `8/43`, delta `-3`, W/L/T `2/5/36`, p `0.453125`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `20/43`, delta `-3`, W/L/T `0/3/40`, p `0.25`
- open_hard_arc_challenge `content_question_only/mean`: base `39/128`, recurrent `37/128`, delta `-2`, W/L/T `9/11/108`, p `0.8238029479980469`
- open_hard_arc_challenge `cyclic_label_aggregated/permutation_mean`: base `75/128`, recurrent `75/128`, delta `0`, W/L/T `2/2/124`, p `1.0`

## stage5_heldout_router_validation_20260625_230408_loop3
- Forced loop count: `3`
- Status: `completed`
- arc_easy `content_question_only/mean`: base `71/128`, recurrent `62/128`, delta `-9`, W/L/T `6/15/107`, p `0.0783538818359375`
- arc_easy `cyclic_label_aggregated/permutation_mean`: base `103/128`, recurrent `97/128`, delta `-6`, W/L/T `0/6/122`, p `0.03125`
- arc_challenge `content_question_only/mean`: base `11/43`, recurrent `8/43`, delta `-3`, W/L/T `4/7/32`, p `0.548828125`
- arc_challenge `cyclic_label_aggregated/permutation_mean`: base `23/43`, recurrent `21/43`, delta `-2`, W/L/T `1/3/39`, p `0.625`
- open_hard_arc_challenge `content_question_only/mean`: base `39/128`, recurrent `33/128`, delta `-6`, W/L/T `10/16/102`, p `0.32693958282470703`
- open_hard_arc_challenge `cyclic_label_aggregated/permutation_mean`: base `75/128`, recurrent `73/128`, delta `-2`, W/L/T `3/5/120`, p `0.7265625`
