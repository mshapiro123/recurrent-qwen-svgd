# Stage 5 Benchmark Suite - stage5_lineage_regression_battery_current_4_stage5_n24_support12_rung_20260707_140139

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `1`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `7822.72`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0006` (base `3115/5197`, recurrent `3118/5197`)
  - paired evidence
    - aggregate `mean`: recurrent `3118` / `5197`, base `3115` / `5197`, delta `3`, W/L/T `24/21/5152`, p `0.765991824244793`
  - routing buckets
    - `ambiguous_proxy`: n `2211`, delta `2`, W/L `18/16`, mean margin delta `0.0013101576734376352`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `1519`, delta `0`, W/L `0/0`, mean margin delta `-0.0012378184157935627`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `864`, delta `2`, W/L `4/2`, mean margin delta `3.967848089006212e-06`, mean loops `1.0`
    - `deep_numeric_proxy`: n `603`, delta `-1`, W/L `2/3`, mean margin delta `-0.0011558506520431037`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-14`, accuracy delta `-0.0027` (base `3976/5197`, recurrent `3962/5197`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `3962` / `5197`, base `3976` / `5197`, delta `-14`, W/L/T `13/27/5157`, p `0.03847730828420026`
  - routing buckets
    - `ambiguous_proxy`: n `1131`, delta `-11`, W/L `8/19`, mean margin delta `-0.0008233870616300161`, mean loops `None`
    - `base_confident_direct_proxy`: n `3246`, delta `0`, W/L `0/0`, mean margin delta `0.0011337659517257205`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `453`, delta `-1`, W/L `2/3`, mean margin delta `-0.0012072311956116777`, mean loops `None`
    - `deep_numeric_proxy`: n `367`, delta `-2`, W/L `3/5`, mean margin delta `0.0038147148406801304`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0019` (base `886/2590`, recurrent `891/2590`)
  - paired evidence
    - aggregate `mean`: recurrent `891` / `2590`, base `886` / `2590`, delta `5`, W/L/T `12/7/2571`, p `0.359283447265625`
  - routing buckets
    - `ambiguous_proxy`: n `1216`, delta `5`, W/L `7/2`, mean margin delta `0.0016825616506761626`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `230`, delta `0`, W/L `0/0`, mean margin delta `0.001938013732433319`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `631`, delta `1`, W/L `3/2`, mean margin delta `-0.001432526697258564`, mean loops `1.0`
    - `deep_numeric_proxy`: n `513`, delta `-1`, W/L `2/3`, mean margin delta `-0.0015137518590886225`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0008` (base `1522/2590`, recurrent `1520/2590`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `1520` / `2590`, base `1522` / `2590`, delta `-2`, W/L/T `11/13/2566`, p `0.8388197422027588`
  - routing buckets
    - `ambiguous_proxy`: n `798`, delta `-4`, W/L `3/7`, mean margin delta `0.0030519009812201283`, mean loops `None`
    - `base_confident_direct_proxy`: n `996`, delta `0`, W/L `0/0`, mean margin delta `-0.00017570400863968255`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `417`, delta `1`, W/L `4/3`, mean margin delta `-0.0030600565049587357`, mean loops `None`
    - `deep_numeric_proxy`: n `379`, delta `1`, W/L `4/3`, mean margin delta `0.0032706695835891617`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `5` (base `886/2590`, recurrent `891/2590`)
