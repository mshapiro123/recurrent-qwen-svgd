# Stage 5 Benchmark Suite - stage5_surface_alignment_repair_content_cyclic_20260624_005040_benchmark

- Status: `completed`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Checkpoint: `outputs/stage5/stage5_surface_alignment_repair_content_cyclic_20260624_005040/phase1_surface_align/phase1_step_50.pt`
- Benchmarks: `['arc_easy', 'arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `561.81`

## Recurrent vs Base

### arc_easy
- score target `content_question_only`
  - aggregate `mean`: correct delta `-10`, accuracy delta `-0.0391` (base `148/256`, recurrent `138/256`)
  - paired evidence
    - aggregate `mean`: recurrent `138` / `256`, base `148` / `256`, delta `-10`, W/L/T `5/15/236`, p `0.04138946533203125`
  - routing buckets
    - `ambiguous_proxy`: n `118`, delta `-5`, W/L `5/10`, mean margin delta `0.087857088800204`, mean loops `1.1578960163613496`
    - `base_confident_direct_proxy`: n `69`, delta `-1`, W/L `0/1`, mean margin delta `0.019585873553718346`, mean loops `1.145192318463671`
    - `conceptual_reasoning_proxy`: n `38`, delta `-1`, W/L `0/1`, mean margin delta `-0.03338003629132321`, mean loops `1.316674893623904`
    - `deep_numeric_proxy`: n `31`, delta `-3`, W/L `0/3`, mean margin delta `0.01214560770219372`, mean loops `1.3277757619657824`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `4`, accuracy delta `0.0156` (base `201/256`, recurrent `205/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `205` / `256`, base `201` / `256`, delta `4`, W/L/T `5/1/250`, p `0.21875`
  - routing buckets
    - `ambiguous_proxy`: n `61`, delta `1`, W/L `2/1`, mean margin delta `0.011724023682782884`, mean loops `None`
    - `base_confident_direct_proxy`: n `161`, delta `0`, W/L `0/0`, mean margin delta `-0.12172272773112476`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `14`, delta `3`, W/L `3/0`, mean margin delta `0.13730700992579972`, mean loops `None`
    - `deep_numeric_proxy`: n `20`, delta `0`, W/L `0/0`, mean margin delta `0.09401862635277211`, mean loops `None`
### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `3`, accuracy delta `0.0117` (base `87/256`, recurrent `90/256`)
  - paired evidence
    - aggregate `mean`: recurrent `90` / `256`, base `87` / `256`, delta `3`, W/L/T `9/6/241`, p `0.60723876953125`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `1`, W/L `4/3`, mean margin delta `-0.1120382426230888`, mean loops `1.209210036892878`
    - `base_confident_direct_proxy`: n `25`, delta `0`, W/L `0/0`, mean margin delta `-0.14841754198074342`, mean loops `1.2400123536586762`
    - `conceptual_reasoning_proxy`: n `72`, delta `1`, W/L `4/3`, mean margin delta `0.00039469036791059707`, mean loops `1.3317039728992515`
    - `deep_numeric_proxy`: n `36`, delta `1`, W/L `1/0`, mean margin delta `0.009843248460027907`, mean loops `1.2981098647470828`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-3`, accuracy delta `-0.0117` (base `156/256`, recurrent `153/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `153` / `256`, base `156` / `256`, delta `-3`, W/L/T `4/7/245`, p `0.548828125`
  - routing buckets
    - `ambiguous_proxy`: n `81`, delta `0`, W/L `3/3`, mean margin delta `-0.0013125163671625722`, mean loops `None`
    - `base_confident_direct_proxy`: n `108`, delta `0`, W/L `0/0`, mean margin delta `-0.26326438086533155`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `-1`, W/L `1/2`, mean margin delta `0.022749569872394203`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-2`, W/L `0/2`, mean margin delta `0.054946042357407736`, mean loops `None`
