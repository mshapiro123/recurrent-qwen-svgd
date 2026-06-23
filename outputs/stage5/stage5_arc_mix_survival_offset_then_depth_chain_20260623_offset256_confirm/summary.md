# Stage 5 Benchmark Suite - stage5_arc_mix_survival_offset_then_depth_chain_20260623_offset256_confirm

- Status: `completed`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260623_145438/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_mix_depth_reuse_offset_20260623_142158_depth_routing_probe/arc_mix_response_w02_lr2e6/phase1/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1282.65`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `11`, accuracy delta `0.0430` (base `152/256`, recurrent `163/256`)
  - paired evidence
    - aggregate `mean`: recurrent `163` / `256`, base `152` / `256`, delta `11`, W/L/T `23/12/221`, p `0.0895310789346695`
  - routing buckets
    - `ambiguous_proxy`: n `105`, delta `11`, W/L `15/4`, mean margin delta `0.30515742216791425`, mean loops `1.805055424712953`
    - `base_confident_direct_proxy`: n `77`, delta `0`, W/L `0/0`, mean margin delta `-0.5169869174140614`, mean loops `1.8104730835466676`
    - `conceptual_reasoning_proxy`: n `46`, delta `3`, W/L `7/4`, mean margin delta `0.1999226212501526`, mean loops `1.952845234585845`
    - `deep_numeric_proxy`: n `28`, delta `-3`, W/L `1/4`, mean margin delta `0.1621913260647229`, mean loops `1.8404698293123924`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0078` (base `204/256`, recurrent `202/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `202` / `256`, base `204` / `256`, delta `-2`, W/L/T `1/3/252`, p `0.625`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `-2`, W/L `0/2`, mean margin delta `0.1500643510837108`, mean loops `None`
    - `base_confident_direct_proxy`: n `164`, delta `0`, W/L `0/0`, mean margin delta `-1.495690769049797`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `29`, delta `0`, W/L `1/1`, mean margin delta `0.38847061465012617`, mean loops `None`
    - `deep_numeric_proxy`: n `13`, delta `0`, W/L `0/0`, mean margin delta `0.0032097626764040488`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0698` (base `11/43`, recurrent `8/43`)
  - paired evidence
    - aggregate `mean`: recurrent `8` / `43`, base `11` / `43`, delta `-3`, W/L/T `2/5/36`, p `0.453125`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `-1`, W/L `2/3`, mean margin delta `0.032399271215711324`, mean loops `1.804346932896546`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.04467610518137614`, mean loops `2.0319868475198746`
    - `deep_numeric_proxy`: n `9`, delta `-2`, W/L `0/2`, mean margin delta `-0.043232738971710205`, mean loops `1.762629015578164`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0233` (base `23/43`, recurrent `24/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `24` / `43`, base `23` / `43`, delta `1`, W/L/T `1/0/42`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `0`, W/L `0/0`, mean margin delta `0.21752738643489364`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.6546304856504624`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.24828396877273917`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `1`, W/L `1/0`, mean margin delta `0.3326092691027692`, mean loops `None`
