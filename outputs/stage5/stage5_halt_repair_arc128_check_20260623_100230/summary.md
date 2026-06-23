# Stage 5 Benchmark Suite - stage5_halt_repair_arc128_check_20260623_100230

- Status: `completed`
- Source summary: `outputs/stage5/stage5_ce8_balanced_arc256_depth_curve_summary_20260623/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_challenge', 'arc_easy']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `1072.41`

## Recurrent vs Base

### arc_challenge
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `2`, accuracy delta `0.0156` (base `68/128`, recurrent `70/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `70` / `128`, base `68` / `128`, delta `2`, W/L/T `4/2/122`, p `0.6875`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `2`, W/L `2/0`, mean margin delta `0.21134569095231073`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-1.0979010837061494`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `1`, W/L `2/1`, mean margin delta `0.3262960982322693`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-1`, W/L `0/1`, mean margin delta `0.17916396570702395`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `4`, accuracy delta `0.0312` (base `43/128`, recurrent `47/128`)
  - paired evidence
    - aggregate `mean`: recurrent `47` / `128`, base `43` / `128`, delta `4`, W/L/T `10/6/112`, p `0.454498291015625`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `1`, W/L `3/2`, mean margin delta `-0.27458740411133603`, mean loops `1.7936781663661714`
    - `base_confident_direct_proxy`: n `11`, delta `-2`, W/L `0/2`, mean margin delta `-0.3774478218772195`, mean loops `1.8154593137177555`
    - `conceptual_reasoning_proxy`: n `39`, delta `3`, W/L `5/2`, mean margin delta `0.008808019833687024`, mean loops `1.9169164513930297`
    - `deep_numeric_proxy`: n `20`, delta `2`, W/L `2/0`, mean margin delta `0.04320268929004669`, mean loops `1.7468413457274437`
### arc_easy
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0312` (base `96/128`, recurrent `100/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `100` / `128`, base `96` / `128`, delta `4`, W/L/T `4/0/124`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `2`, W/L `2/0`, mean margin delta `0.19957721842718976`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `-1.2714809748410196`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `2`, W/L `2/0`, mean margin delta `0.19952561166137456`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.19019494764506817`, mean loops `None`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.0547` (base `74/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `mean`: recurrent `67` / `128`, base `74` / `128`, delta `-7`, W/L/T `7/14/107`, p `0.18924713134765625`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `-5`, W/L `5/10`, mean margin delta `-0.00597620341512892`, mean loops `1.7498133447435167`
    - `base_confident_direct_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.4606754848280469`, mean loops `1.7742438263184317`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `2/2`, mean margin delta `-0.26404800415039065`, mean loops `1.8413195505738258`
    - `deep_numeric_proxy`: n `17`, delta `-1`, W/L `0/1`, mean margin delta `-0.3034467381589553`, mean loops `1.9041535608908708`
