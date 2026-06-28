# Stage 5 Benchmark Suite - stage5_forced_depth_arc_challenge_loop123_20260628_031842_loop8

- Status: `completed`
- Suite profile: `default`
- Source summary: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/summary.json`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Benchmarks: `['arc_challenge']`
- Recurrent mode: `phase1`
- Recurrent max loops: `8`
- Forced loop count: `8`
- Recurrent trajectories: `1`
- Learned loop control: `False`
- Elapsed seconds: `2416.54`

## Recurrent vs Base

### arc_challenge
- score target `content_question_only`
  - aggregate `mean`: correct delta `-15`, accuracy delta `-0.0586` (base `88/256`, recurrent `73/256`)
  - paired evidence
    - aggregate `mean`: recurrent `73` / `256`, base `88` / `256`, delta `-15`, W/L/T `29/44/183`, p `0.1006436775202357`
  - routing buckets
    - `ambiguous_proxy`: n `123`, delta `-3`, W/L `12/15`, mean margin delta `-0.09269931209765798`, mean loops `1.390775893261116`
    - `base_confident_direct_proxy`: n `25`, delta `-6`, W/L `0/6`, mean margin delta `-0.9592268800735474`, mean loops `1.4155764365196228`
    - `conceptual_reasoning_proxy`: n `72`, delta `-4`, W/L `12/16`, mean margin delta `-0.06278719007968903`, mean loops `1.5561795851422682`
    - `deep_numeric_proxy`: n `36`, delta `-2`, W/L `5/7`, mean margin delta `-0.22190550135241616`, mean loops `1.4876426955064137`
- score target `cyclic_label_aggregated`
  - aggregate `permutation_mean`: correct delta `-84`, accuracy delta `-0.3281` (base `155/256`, recurrent `71/256`)
  - paired evidence
    - aggregate `permutation_mean`: recurrent `71` / `256`, base `155` / `256`, delta `-84`, W/L/T `23/107/126`, p `3.7937066234622226e-14`
  - routing buckets
    - `ambiguous_proxy`: n `82`, delta `-4`, W/L `16/20`, mean margin delta `0.6521214405945464`, mean loops `None`
    - `base_confident_direct_proxy`: n `107`, delta `-71`, W/L `0/71`, mean margin delta `-2.6395443828130207`, mean loops `None`
    - `conceptual_reasoning_proxy`: n `45`, delta `-5`, W/L `4/9`, mean margin delta `0.870138894000815`, mean loops `None`
    - `deep_numeric_proxy`: n `22`, delta `-4`, W/L `3/7`, mean margin delta `0.5362216092109905`, mean loops `None`

## Hard Content Signal

- `arc_challenge` `content_question_only`: delta `-15` (base `88/256`, recurrent `73/256`)
