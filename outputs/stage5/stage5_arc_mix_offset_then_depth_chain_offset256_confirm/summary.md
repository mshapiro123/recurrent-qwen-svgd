# Stage 5 Benchmark Suite - stage5_arc_mix_offset_then_depth_chain_offset256_confirm

- Status: `completed`
- Source summary: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json`
- Checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1313.48`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `8`, accuracy delta `0.0312` (base `152/256`, recurrent `160/256`)
  - paired evidence
    - aggregate `mean`: recurrent `160` / `256`, base `152` / `256`, delta `8`, W/L/T `20/12/224`, p `0.21532714972272515`
  - routing buckets
    - `ambiguous_proxy`: n `105`, delta `9`, W/L `13/4`, mean margin delta `0.2568628495647794`, mean loops `1.821383322704406`
    - `base_confident_direct_proxy`: n `77`, delta `0`, W/L `0/0`, mean margin delta `-0.5510501041524596`, mean loops `1.8261790364593655`
    - `conceptual_reasoning_proxy`: n `46`, delta `2`, W/L `6/4`, mean margin delta `0.1768091761547586`, mean loops `1.968374285361041`
    - `deep_numeric_proxy`: n `28`, delta `-3`, W/L `1/4`, mean margin delta `0.12896737243447984`, mean loops `1.8550933737839972`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0078` (base `204/256`, recurrent `202/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `202` / `256`, base `204` / `256`, delta `-2`, W/L/T `2/4/250`, p `0.6875`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `-2`, W/L `0/2`, mean margin delta `0.15243457249365747`, mean loops `None`
    - `base_confident_direct_proxy`: n `164`, delta `0`, W/L `0/0`, mean margin delta `-1.516107774377475`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `29`, delta `0`, W/L `1/1`, mean margin delta `0.38787787574632415`, mean loops `None`
    - `deep_numeric_proxy`: n `13`, delta `0`, W/L `1/1`, mean margin delta `-0.012110168830706524`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-2`, accuracy delta `-0.0465` (base `11/43`, recurrent `9/43`)
  - paired evidence
    - aggregate `mean`: recurrent `9` / `43`, base `11` / `43`, delta `-2`, W/L/T `2/4/37`, p `0.6875`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `-1`, W/L `2/3`, mean margin delta `0.03168714897973197`, mean loops `1.8195909740669387`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.05220818519592285`, mean loops `2.0497077653805413`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.03863463799158732`, mean loops `1.7768282790978749`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `23/43`, recurrent `23/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `23` / `43`, base `23` / `43`, delta `0`, W/L/T `1/1/41`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `0/1`, mean margin delta `0.2296748987217951`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.6699592432317635`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.2352482882949213`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `1`, W/L `1/0`, mean margin delta `0.3410711111500859`, mean loops `None`
