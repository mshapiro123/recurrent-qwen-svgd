# Stage 5 Benchmark Suite - stage5_lineage_regression_battery_current_1_stage5_reentry_recovery_20260625_154210

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260625_154210/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/phase1/phase1_step_75.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `1`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `12254.93`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `30`, accuracy delta `0.0058` (base `3115/5197`, recurrent `3145/5197`)
  - paired evidence
    - aggregate `mean`: recurrent `3145` / `5197`, base `3115` / `5197`, delta `30`, W/L/T `181/151/4865`, p `0.11134099001135897`
  - routing buckets
    - `ambiguous_proxy`: n `2211`, delta `35`, W/L `116/81`, mean margin delta `0.04367782641918289`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `1519`, delta `-4`, W/L `0/4`, mean margin delta `0.09892271590309366`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `864`, delta `0`, W/L `39/39`, mean margin delta `0.01613804501377874`, mean loops `1.0`
    - `deep_numeric_proxy`: n `603`, delta `-1`, W/L `26/27`, mean margin delta `0.005889524521914683`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `8`, accuracy delta `0.0015` (base `3976/5197`, recurrent `3984/5197`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `3984` / `5197`, base `3976` / `5197`, delta `8`, W/L/T `61/53/5083`, p `0.5122548722298448`
  - routing buckets
    - `ambiguous_proxy`: n `1131`, delta `6`, W/L `41/35`, mean margin delta `-0.16160293092952274`, mean loops `None`
    - `base_confident_direct_proxy`: n `3246`, delta `0`, W/L `0/0`, mean margin delta `0.6983332061219655`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `453`, delta `3`, W/L `9/6`, mean margin delta `-0.19370861097952613`, mean loops `None`
    - `deep_numeric_proxy`: n `367`, delta `-1`, W/L `11/12`, mean margin delta `-0.12465655852911095`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `8`, accuracy delta `0.0031` (base `886/2590`, recurrent `894/2590`)
  - paired evidence
    - aggregate `mean`: recurrent `894` / `2590`, base `886` / `2590`, delta `8`, W/L/T `69/61/2460`, p `0.539419552281719`
  - routing buckets
    - `ambiguous_proxy`: n `1216`, delta `7`, W/L `38/31`, mean margin delta `-0.008197101801143665`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `230`, delta `-3`, W/L `0/3`, mean margin delta `0.022573881272388543`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `631`, delta `0`, W/L `17/17`, mean margin delta `-0.025121587864381574`, mean loops `1.0`
    - `deep_numeric_proxy`: n `513`, delta `4`, W/L `14/10`, mean margin delta `0.00635480781977172`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0015` (base `1522/2590`, recurrent `1526/2590`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `1526` / `2590`, base `1522` / `2590`, delta `4`, W/L/T `45/41/2504`, p `0.7465343822890573`
  - routing buckets
    - `ambiguous_proxy`: n `798`, delta `4`, W/L `23/19`, mean margin delta `-0.24019031974884492`, mean loops `None`
    - `base_confident_direct_proxy`: n `996`, delta `0`, W/L `0/0`, mean margin delta `0.636279491996354`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `417`, delta `-6`, W/L `7/13`, mean margin delta `-0.25099920027163086`, mean loops `None`
    - `deep_numeric_proxy`: n `379`, delta `6`, W/L `15/9`, mean margin delta `-0.14041886393224703`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `8` (base `886/2590`, recurrent `894/2590`)
