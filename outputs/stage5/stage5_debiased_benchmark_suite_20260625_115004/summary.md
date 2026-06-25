# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260625_115004

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260625_114836/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_114836_curriculum_sft/phase1/phase1_step_75.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1372.11`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `6`, accuracy delta `0.0469` (base `86/128`, recurrent `92/128`)
  - paired evidence
    - aggregate `mean`: recurrent `92` / `128`, base `86` / `128`, delta `6`, W/L/T `6/0/122`, p `0.03125`
  - routing buckets
    - `ambiguous_proxy`: n `34`, delta `2`, W/L `2/0`, mean margin delta `-0.02005758651477449`, mean loops `1.133056229528259`
    - `base_confident_direct_proxy`: n `73`, delta `0`, W/L `0/0`, mean margin delta `-0.04536611998249611`, mean loops `1.1992731816148106`
    - `conceptual_reasoning_proxy`: n `12`, delta `2`, W/L `2/0`, mean margin delta `0.3535536589721839`, mean loops `1.3362978498140972`
    - `deep_numeric_proxy`: n `9`, delta `2`, W/L `2/0`, mean margin delta `0.4370366326636738`, mean loops `1.32592092288865`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0234` (base `74/128`, recurrent `71/128`)
  - paired evidence
    - aggregate `mean`: recurrent `71` / `128`, base `74` / `128`, delta `-3`, W/L/T `3/6/119`, p `0.5078125`
  - routing buckets
    - `ambiguous_proxy`: n `54`, delta `-1`, W/L `3/4`, mean margin delta `0.06754757077605636`, mean loops `1.0902322497632768`
    - `base_confident_direct_proxy`: n `37`, delta `-1`, W/L `0/1`, mean margin delta `-0.010753608434586911`, mean loops `1.0644621576811817`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `-0.023731154203414918`, mean loops `1.1387526527047158`
    - `deep_numeric_proxy`: n `17`, delta `-1`, W/L `0/1`, mean margin delta `0.01489038677776561`, mean loops `1.2045654917464537`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `3`, accuracy delta `0.0234` (base `96/128`, recurrent `99/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `99` / `128`, base `96` / `128`, delta `3`, W/L/T `4/1/123`, p `0.375`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `1`, W/L `2/1`, mean margin delta `-0.04890134446322918`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `0.09050282231481696`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `2`, W/L `2/0`, mean margin delta `0.12042871173471212`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.1408505957159731`, mean loops `None`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0078` (base `72/128`, recurrent `71/128`)
  - paired evidence
    - aggregate `mean`: recurrent `71` / `128`, base `72` / `128`, delta `-1`, W/L/T `2/3/123`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `36`, delta `-2`, W/L `0/2`, mean margin delta `0.057205404775838055`, mean loops `1.2233282683624163`
    - `base_confident_direct_proxy`: n `52`, delta `0`, W/L `0/0`, mean margin delta `-0.15462194834477627`, mean loops `1.3078470304608345`
    - `conceptual_reasoning_proxy`: n `26`, delta `1`, W/L `2/1`, mean margin delta `-0.032498312970766656`, mean loops `1.3416132307969606`
    - `deep_numeric_proxy`: n `14`, delta `0`, W/L `0/0`, mean margin delta `0.05285451029028211`, mean loops `1.2466794919399988`
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0234` (base `43/128`, recurrent `46/128`)
  - paired evidence
    - aggregate `mean`: recurrent `46` / `128`, base `43` / `128`, delta `3`, W/L/T `5/2/121`, p `0.453125`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `-2`, W/L `0/2`, mean margin delta `-0.1407059410522724`, mean loops `1.1324094816528518`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `0.13105730576948685`, mean loops `1.1142663522200151`
    - `conceptual_reasoning_proxy`: n `39`, delta `4`, W/L `4/0`, mean margin delta `0.02928741467304719`, mean loops `1.1992818040725512`
    - `deep_numeric_proxy`: n `20`, delta `1`, W/L `1/0`, mean margin delta `0.08218550980091095`, mean loops `1.14096989184618`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-1`, accuracy delta `-0.0078` (base `68/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `67` / `128`, base `68` / `128`, delta `-1`, W/L/T `1/2/125`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `43`, delta `1`, W/L `1/0`, mean margin delta `-0.009133200110622153`, mean loops `None`
    - `base_confident_direct_proxy`: n `46`, delta `0`, W/L `0/0`, mean margin delta `-0.3298559974241273`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `-1`, W/L `0/1`, mean margin delta `0.0012450920045375825`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `-1`, W/L `0/1`, mean margin delta `0.002964724830928308`, mean loops `None`
### gpqa_lite
- score target `label`
- score target `content_question_only`
- score target `cyclic_label_aggregated`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_debiased_benchmark_suite_20260625_115004/gpqa_diamond_16.jsonl
