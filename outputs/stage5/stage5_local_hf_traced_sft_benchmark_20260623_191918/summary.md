# Stage 5 Benchmark Suite - stage5_local_hf_traced_sft_benchmark_20260623_191918

- Status: `completed`
- Source summary: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_191843/summary.json`
- Checkpoint: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_191843/phase1/phase1_step_150.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `121.31`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0104` (base `34/96`, recurrent `33/96`)
  - paired evidence
    - aggregate `mean`: recurrent `33` / `96`, base `34` / `96`, delta `-1`, W/L/T `1/2/93`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `0`, W/L `0/0`, mean margin delta `-0.2249445582545081`, mean loops `1.37461119728495`
    - `base_confident_direct_proxy`: n `9`, delta `-1`, W/L `0/1`, mean margin delta `-0.10379637612236871`, mean loops `1.3980343904760149`
    - `conceptual_reasoning_proxy`: n `28`, delta `0`, W/L `1/1`, mean margin delta `-0.07848548889160156`, mean loops `1.5251695364713669`
    - `deep_numeric_proxy`: n `16`, delta `0`, W/L `0/0`, mean margin delta `0.08883471414446831`, mean loops `1.3452776322762172`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `1`, accuracy delta `0.0104` (base `51/96`, recurrent `52/96`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `52` / `96`, base `51` / `96`, delta `1`, W/L/T `3/2/91`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `31`, delta `1`, W/L `1/0`, mean margin delta `0.08110739425405539`, mean loops `None`
    - `base_confident_direct_proxy`: n `37`, delta `0`, W/L `0/0`, mean margin delta `-0.6636096996607611`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `17`, delta `2`, W/L `2/0`, mean margin delta `0.1502281080054886`, mean loops `None`
    - `deep_numeric_proxy`: n `11`, delta `-2`, W/L `0/2`, mean margin delta `0.08103510308446306`, mean loops `None`
