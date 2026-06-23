# Stage 5 Benchmark Suite - stage5_benchmark_mcq_ladder_sft_stream_20260623_055706

- Status: `completed`
- Source summary: `outputs/stage5/stage5_mcq_ladder_sft_nodrive_20260623_052428/summary.json`
- Checkpoint: `outputs/stage5/stage5_mcq_ladder_sft_nodrive_20260623_052428/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `599.93`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-9`, accuracy delta `-0.1406` (base `41/64`, recurrent `32/64`)
  - paired evidence
    - aggregate `mean`: recurrent `32` / `64`, base `41` / `64`, delta `-9`, W/L/T `4/13/47`, p `0.049041748046875`
  - routing buckets
    - `ambiguous_proxy`: n `21`, delta `-3`, W/L `3/6`, mean margin delta `-0.05051233087267194`, mean loops `3.128186745303018`
    - `base_confident_direct_proxy`: n `23`, delta `-4`, W/L `0/4`, mean margin delta `-1.239420984102332`, mean loops `3.1267386213592863`
    - `conceptual_reasoning_proxy`: n `10`, delta `-1`, W/L `1/2`, mean margin delta `-0.3114979863166809`, mean loops `3.1322439134120943`
    - `deep_numeric_proxy`: n `10`, delta `-1`, W/L `0/1`, mean margin delta `-0.5192054033279419`, mean loops `3.1352606892585753`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `52/64`, recurrent `52/64`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `52` / `64`, base `52` / `64`, delta `0`, W/L/T `0/0/64`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `15`, delta `0`, W/L `0/0`, mean margin delta `0.37264368987331786`, mean loops `None`
    - `base_confident_direct_proxy`: n `40`, delta `0`, W/L `0/0`, mean margin delta `-2.731413603803376`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `5`, delta `0`, W/L `0/0`, mean margin delta `0.29977105371654034`, mean loops `None`
    - `deep_numeric_proxy`: n `4`, delta `0`, W/L `0/0`, mean margin delta `0.1941769616678357`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-5`, accuracy delta `-0.0781` (base `24/64`, recurrent `19/64`)
  - paired evidence
    - aggregate `mean`: recurrent `19` / `64`, base `24` / `64`, delta `-5`, W/L/T `3/8/53`, p `0.2265625`
  - routing buckets
    - `ambiguous_proxy`: n `25`, delta `1`, W/L `2/1`, mean margin delta `-0.2626764678955078`, mean loops `3.13183491786321`
    - `base_confident_direct_proxy`: n `8`, delta `-2`, W/L `0/2`, mean margin delta `-0.8389098048210144`, mean loops `3.128977820277214`
    - `conceptual_reasoning_proxy`: n `20`, delta `-2`, W/L `1/3`, mean margin delta `-0.11019906997680665`, mean loops `3.134504535794258`
    - `deep_numeric_proxy`: n `11`, delta `-2`, W/L `0/2`, mean margin delta `-0.15208315307443793`, mean loops `3.1320499362367573`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-2`, accuracy delta `-0.0312` (base `37/64`, recurrent `35/64`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `35` / `64`, base `37` / `64`, delta `-2`, W/L/T `0/2/62`, p `0.5`
  - routing buckets
    - `ambiguous_proxy`: n `18`, delta `-1`, W/L `0/1`, mean margin delta `0.6889273424253419`, mean loops `None`
    - `base_confident_direct_proxy`: n `27`, delta `0`, W/L `0/0`, mean margin delta `-2.0474980574756585`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.3490377872843634`, mean loops `None`
    - `deep_numeric_proxy`: n `8`, delta `-1`, W/L `0/1`, mean margin delta `0.41540269305308664`, mean loops `None`
