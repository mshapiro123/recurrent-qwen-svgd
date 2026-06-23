# Stage 5 Benchmark Suite - stage5_local_hf_traced_sft_benchmark_20260623_194626

- Status: `completed`
- Source summary: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/summary.json`
- Checkpoint: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/phase1/phase1_step_200.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `304.03`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-7`, accuracy delta `-0.0547` (base `75/128`, recurrent `68/128`)
  - paired evidence
    - aggregate `mean`: recurrent `68` / `128`, base `75` / `128`, delta `-7`, W/L/T `3/10/115`, p `0.09228515625`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `-6`, W/L `2/8`, mean margin delta `-0.11208828842198407`, mean loops `1.0962840605665136`
    - `base_confident_direct_proxy`: n `36`, delta `-1`, W/L `0/1`, mean margin delta `-0.03942965136633979`, mean loops `1.0685072301162615`
    - `conceptual_reasoning_proxy`: n `21`, delta `0`, W/L `0/0`, mean margin delta `-0.1710406712123326`, mean loops `1.1448466153371901`
    - `deep_numeric_proxy`: n `17`, delta `0`, W/L `1/1`, mean margin delta `-0.060047458199893725`, mean loops `1.2180602427791147`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `5`, accuracy delta `0.0391` (base `95/128`, recurrent `100/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `100` / `128`, base `95` / `128`, delta `5`, W/L/T `6/1/121`, p `0.125`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `2`, W/L `3/1`, mean margin delta `-0.08245974946767091`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `0.28238088257057825`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `3`, W/L `3/0`, mean margin delta `0.17558022700250148`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.15241325926035643`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0234` (base `43/128`, recurrent `46/128`)
  - paired evidence
    - aggregate `mean`: recurrent `46` / `128`, base `43` / `128`, delta `3`, W/L/T `4/1/123`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `-1`, W/L `0/1`, mean margin delta `-0.2853967382990081`, mean loops `1.1397401192064942`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.15604164383628152`, mean loops `1.122332667762583`
    - `conceptual_reasoning_proxy`: n `39`, delta `3`, W/L `3/0`, mean margin delta `-0.11346059273450802`, mean loops `1.2101069383132153`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.0933020532131195`, mean loops `1.15111583173275`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0078` (base `68/128`, recurrent `69/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `69` / `128`, base `68` / `128`, delta `1`, W/L/T `4/3/121`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `2`, W/L `2/0`, mean margin delta `-0.05734412099935056`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-0.18820638763946076`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `1`, W/L `2/1`, mean margin delta `-0.05072419602423906`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-2`, W/L `0/2`, mean margin delta `0.0007250625640153885`, mean loops `None`
