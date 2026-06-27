# Stage 5 Benchmark Suite - stage5_debiased_benchmark_suite_20260627_191753

- Status: `completed_with_failures`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Benchmarks: `['arc_easy', 'arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent max loops: `4`
- Forced loop count: `None`
- Recurrent trajectories: `1`
- Learned loop control: `True`
- Elapsed seconds: `1857.45`

## Recurrent vs Base

### arc_easy
- score target `label`
  - aggregate `mean`: correct delta `5`, accuracy delta `0.0391` (base `85/128`, recurrent `90/128`)
  - paired evidence
    - aggregate `mean`: recurrent `90` / `128`, base `85` / `128`, delta `5`, W/L/T `7/2/119`, p `0.1796875`
  - routing buckets
    - `ambiguous_proxy`: n `33`, delta `2`, W/L `3/1`, mean margin delta `0.1806915978139097`, mean loops `1.1450671056906383`
    - `base_confident_direct_proxy`: n `74`, delta `-1`, W/L `0/1`, mean margin delta `-0.5886600573949918`, mean loops `1.2108747733605874`
    - `conceptual_reasoning_proxy`: n `12`, delta `3`, W/L `3/0`, mean margin delta `0.2897544888158639`, mean loops `1.3523175567388535`
    - `deep_numeric_proxy`: n `9`, delta `1`, W/L `1/0`, mean margin delta `0.48084947135713363`, mean loops `1.3425591223769717`
- score target `content_question_only`
  - aggregate `mean`: correct delta `-1`, accuracy delta `-0.0078` (base `74/128`, recurrent `73/128`)
  - paired evidence
    - aggregate `mean`: recurrent `73` / `128`, base `74` / `128`, delta `-1`, W/L/T `0/1/127`, p `1.0`
  - routing buckets
    - `ambiguous_proxy`: n `53`, delta `0`, W/L `0/0`, mean margin delta `0.026456807019575587`, mean loops `1.097152370889232`
    - `base_confident_direct_proxy`: n `38`, delta `0`, W/L `0/0`, mean margin delta `-0.02662316278407448`, mean loops `1.0652836622376192`
    - `conceptual_reasoning_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.026442426443099975`, mean loops `1.1471116155385972`
    - `deep_numeric_proxy`: n `17`, delta `-1`, W/L `0/1`, mean margin delta `-0.07644021160462323`, mean loops `1.213456788483788`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `96/128`, recurrent `96/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `96` / `128`, base `96` / `128`, delta `0`, W/L/T `0/0/128`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `35`, delta `0`, W/L `0/0`, mean margin delta `0.10027757079473563`, mean loops `None`
    - `base_confident_direct_proxy`: n `74`, delta `0`, W/L `0/0`, mean margin delta `-0.5343854795557735`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `10`, delta `0`, W/L `0/0`, mean margin delta `0.15078411996364594`, mean loops `None`
    - `deep_numeric_proxy`: n `9`, delta `0`, W/L `0/0`, mean margin delta `0.13666903268959787`, mean loops `None`
### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `-3`, accuracy delta `-0.0234` (base `71/128`, recurrent `68/128`)
  - paired evidence
    - aggregate `mean`: recurrent `68` / `128`, base `71` / `128`, delta `-3`, W/L/T `0/3/125`, p `0.25`
  - routing buckets
    - `ambiguous_proxy`: n `36`, delta `-2`, W/L `0/2`, mean margin delta `0.24766534846276045`, mean loops `1.236193808140578`
    - `base_confident_direct_proxy`: n `52`, delta `-1`, W/L `0/1`, mean margin delta `-0.7029803317183485`, mean loops `1.325618980022577`
    - `conceptual_reasoning_proxy`: n `26`, delta `0`, W/L `0/0`, mean margin delta `0.20436478635439506`, mean loops `1.3586958004878118`
    - `deep_numeric_proxy`: n `14`, delta `0`, W/L `0/0`, mean margin delta `0.16173926208700454`, mean loops `1.2609592328468957`
- score target `content_question_only`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `44/128`, recurrent `44/128`)
  - paired evidence
    - aggregate `mean`: recurrent `44` / `128`, base `44` / `128`, delta `0`, W/L/T `0/0/128`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `58`, delta `0`, W/L `0/0`, mean margin delta `0.01837112677508387`, mean loops `1.1384823312019479`
    - `base_confident_direct_proxy`: n `11`, delta `0`, W/L `0/0`, mean margin delta `-0.0781148997220126`, mean loops `1.1181328675963662`
    - `conceptual_reasoning_proxy`: n `39`, delta `0`, W/L `0/0`, mean margin delta `0.020154497562310636`, mean loops `1.2067335125727532`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.016109704971313477`, mean loops `1.1482102011640867`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `0`, accuracy delta `0.0000` (base `67/128`, recurrent `67/128`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `67` / `128`, base `67` / `128`, delta `0`, W/L/T `0/0/128`, p `None`
  - routing buckets
    - `ambiguous_proxy`: n `44`, delta `0`, W/L `0/0`, mean margin delta `0.12850384087527567`, mean loops `None`
    - `base_confident_direct_proxy`: n `45`, delta `0`, W/L `0/0`, mean margin delta `-0.8857105446151561`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `0.3456833005696535`, mean loops `None`
    - `deep_numeric_proxy`: n `14`, delta `0`, W/L `0/0`, mean margin delta `0.10857356561436539`, mean loops `None`
### gpqa_lite
- score target `label`
- score target `content_question_only`
- score target `cyclic_label_aggregated`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_debiased_benchmark_suite_20260627_191753/gpqa_diamond_16.jsonl

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `0` (base `44/128`, recurrent `44/128`)
