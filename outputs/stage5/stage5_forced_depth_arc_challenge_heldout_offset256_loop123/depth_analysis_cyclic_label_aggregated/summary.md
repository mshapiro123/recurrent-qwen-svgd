# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_heldout_offset256_loop123

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_heldout_offset256_loop123/summary.json`
- Score target: `cyclic_label_aggregated`
- Aggregate: `permutation_mean`
- Loops: `[1, 2, 3]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `24/43`, base `23/43`, delta `1`, W/L/T `1/0/42`, p `1.0`
- loop `2`: recurrent `21/43`, base `23/43`, delta `-2`, W/L/T `0/2/41`, p `0.5`
- loop `3`: recurrent `17/43`, base `23/43`, delta `-6`, W/L/T `0/6/37`, p `0.03125`

### Depth Interaction

- loop1 correct: `24/43`
- any recurrent depth correct: `24/43` (oracle gain vs loop1 `0`)
- base or any recurrent correct: `24/43` (oracle gain vs base `1`)
- deeper unique over loop1: `0`
- deeper unique over base+loop1: `0`
- loop1 harmed by at least one deeper loop: `8`
- depth hit patterns: `{'000': 19, '111': 16, '100': 2, '101': 1, '110': 5}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `24/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `24/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `24/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `24/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `2`: correct `23/43`, delta vs loop1 `-1`, routed deep `10`, W/L `0/1`
- `base` margin < `0.5` -> loop `2`: correct `22/43`, delta vs loop1 `-2`, routed deep `15`, W/L `0/2`
- `loop1` margin < `0.25` -> loop `2`: correct `22/43`, delta vs loop1 `-2`, routed deep `10`, W/L `0/2`
- `base` margin < `0.75` -> loop `2`: correct `21/43`, delta vs loop1 `-3`, routed deep `19`, W/L `0/3`
- `base` margin < `1.0` -> loop `2`: correct `21/43`, delta vs loop1 `-3`, routed deep `25`, W/L `0/3`
- `base` margin < `1.5` -> loop `2`: correct `21/43`, delta vs loop1 `-3`, routed deep `30`, W/L `0/3`

### Best Score Selectors

- subset `[1]` `mean`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `mean`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `max`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `max`: correct `24/43`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `8`: base `5`, by_loop `{1: 5, 2: 5, 3: 5}`, any recurrent `5`
- `low` n `15`: base `4`, by_loop `{1: 5, 2: 3, 3: 1}`, any recurrent `5`
- `thin` n `10`: base `7`, by_loop `{1: 7, 2: 6, 3: 5}`, any recurrent `7`
- `very_confident` n `10`: base `7`, by_loop `{1: 7, 2: 7, 3: 6}`, any recurrent `7`
