# Stage 5 Benchmark Suite - stage5_regression_battery_loop1_current_1_stage5_chain_continuation_attribution_20260704_163056

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_chain_continuation_attribution_20260704_163056/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `1`
- Forced loop count: `1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `12246.45`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `2`, accuracy delta `0.0004` (base `3115/5197`, recurrent `3117/5197`)
  - paired evidence
    - aggregate `mean`: recurrent `3117` / `5197`, base `3115` / `5197`, delta `2`, W/L/T `23/21/5153`, p `0.880395821280672`
  - routing buckets
    - `ambiguous_proxy`: n `2211`, delta `0`, W/L `15/15`, mean margin delta `0.0018819159967551021`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `1519`, delta `0`, W/L `0/0`, mean margin delta `-3.182480431764513e-05`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `864`, delta `3`, W/L `5/2`, mean margin delta `5.107521320934649e-05`, mean loops `1.0`
    - `deep_numeric_proxy`: n `603`, delta `-1`, W/L `3/4`, mean margin delta `0.0006391307706658916`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-12`, accuracy delta `-0.0023` (base `3976/5197`, recurrent `3964/5197`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `3964` / `5197`, base `3976` / `5197`, delta `-12`, W/L/T `13/25/5159`, p `0.07295138851623051`
  - routing buckets
    - `ambiguous_proxy`: n `1131`, delta `-11`, W/L `6/17`, mean margin delta `-0.0012304760865862196`, mean loops `None`
    - `base_confident_direct_proxy`: n `3246`, delta `0`, W/L `0/0`, mean margin delta `-0.0004778345829862594`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `453`, delta `2`, W/L `5/3`, mean margin delta `0.0054152863407014125`, mean loops `None`
    - `deep_numeric_proxy`: n `367`, delta `-3`, W/L `2/5`, mean margin delta `-0.0012744099882132723`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0012` (base `886/2590`, recurrent `883/2590`)
  - paired evidence
    - aggregate `mean`: recurrent `883` / `2590`, base `886` / `2590`, delta `-3`, W/L/T `10/13/2567`, p `0.6776394844055176`
  - routing buckets
    - `ambiguous_proxy`: n `1216`, delta `-3`, W/L `5/8`, mean margin delta `0.00181695532151743`, mean loops `1.0`
    - `base_confident_direct_proxy`: n `230`, delta `0`, W/L `0/0`, mean margin delta `-0.0023379593439724136`, mean loops `1.0`
    - `conceptual_reasoning_proxy`: n `631`, delta `1`, W/L `3/2`, mean margin delta `-0.0015740826277271882`, mean loops `1.0`
    - `deep_numeric_proxy`: n `513`, delta `-1`, W/L `2/3`, mean margin delta `-0.0010174532034243756`, mean loops `1.0`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0004` (base `1522/2590`, recurrent `1521/2590`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `1521` / `2590`, base `1522` / `2590`, delta `-1`, W/L/T `11/12/2567`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `798`, delta `-1`, W/L `4/5`, mean margin delta `0.0003968255087377706`, mean loops `None`
    - `base_confident_direct_proxy`: n `996`, delta `0`, W/L `0/0`, mean margin delta `0.000100398962360237`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `417`, delta `-1`, W/L `4/5`, mean margin delta `0.0004871102195303087`, mean loops `None`
    - `deep_numeric_proxy`: n `379`, delta `1`, W/L `3/2`, mean margin delta `0.0027759466098093988`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-3` (base `886/2590`, recurrent `883/2590`)
