# Stage 5 Benchmark Suite - stage5_arc_mix_offset_then_depth_chain_20260623_135452_offset256_confirm

- Status: `completed`
- Source summary: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json`
- Checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1243.88`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `10`, accuracy delta `0.0391` (base `152/256`, recurrent `162/256`)
  - paired evidence
    - aggregate `mean`: recurrent `162` / `256`, base `152` / `256`, delta `10`, W/L/T `21/11/224`, p `0.11018416518345475`
  - routing buckets
    - `ambiguous_proxy`: n `105`, delta `10`, W/L `14/4`, mean margin delta `0.28667037174815224`, mean loops `1.6801477701891037`
    - `base_confident_direct_proxy`: n `77`, delta `0`, W/L `0/0`, mean margin delta `-0.435253694895413`, mean loops `1.6884340227166295`
    - `conceptual_reasoning_proxy`: n `46`, delta `3`, W/L `6/3`, mean margin delta `0.17933268909868988`, mean loops `1.8140716688788456`
    - `deep_numeric_proxy`: n `28`, delta `-3`, W/L `1/4`, mean margin delta `0.13087828883102962`, mean loops `1.7304426510419166`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0078` (base `204/256`, recurrent `202/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `202` / `256`, base `204` / `256`, delta `-2`, W/L/T `2/4/250`, p `0.6875`
  - routing buckets
    - `ambiguous_proxy`: n `50`, delta `-2`, W/L `0/2`, mean margin delta `0.1229046489391476`, mean loops `None`
    - `base_confident_direct_proxy`: n `164`, delta `0`, W/L `0/0`, mean margin delta `-1.3465042818066517`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `29`, delta `0`, W/L `1/1`, mean margin delta `0.34053668476127347`, mean loops `None`
    - `deep_numeric_proxy`: n `13`, delta `0`, W/L `1/1`, mean margin delta `-0.023843143479182184`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `11/43`, recurrent `11/43`)
  - paired evidence
    - aggregate `mean`: recurrent `11` / `43`, base `11` / `43`, delta `0`, W/L/T `2/2/39`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `28`, delta `1`, W/L `2/1`, mean margin delta `0.05226846890790122`, mean loops `1.6875733371291841`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.03273582458496094`, mean loops `1.884537676970164`
    - `deep_numeric_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.042093786928388804`, mean loops `1.6593599021434784`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0233` (base `23/43`, recurrent `22/43`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `22` / `43`, base `23` / `43`, delta `-1`, W/L/T `1/2/40`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `0/1`, mean margin delta `0.17595994929110426`, mean loops `None`
    - `base_confident_direct_proxy`: n `12`, delta `0`, W/L `0/0`, mean margin delta `-0.6412007008912042`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `6`, delta `0`, W/L `0/0`, mean margin delta `0.18068041543786725`, mean loops `None`
    - `deep_numeric_proxy`: n `7`, delta `0`, W/L `1/1`, mean margin delta `0.3058151782357267`, mean loops `None`
