# Depth Sweep Analysis - stage5_prelude_forced_depth_heldout_arc_loop1248

- Source: `outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json`
- Score target: `cyclic_label_aggregated`
- Aggregate: `permutation_mean`
- Loops: `[1, 2, 4, 8]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `21/43`, base `23/43`, delta `-2`, W/L/T `1/3/39`, p `0.625`
- loop `2`: recurrent `20/43`, base `23/43`, delta `-3`, W/L/T `1/4/38`, p `0.375`
- loop `4`: recurrent `22/43`, base `23/43`, delta `-1`, W/L/T `4/5/34`, p `1.0`
- loop `8`: recurrent `19/43`, base `23/43`, delta `-4`, W/L/T `5/9/29`, p `0.4239501953125`

### Depth Interaction

- loop1 correct: `21/43`
- any recurrent depth correct: `28/43` (oracle gain vs loop1 `7`)
- base or any recurrent correct: `29/43` (oracle gain vs base `6`)
- deeper unique over loop1: `7`
- deeper unique over base+loop1: `5`
- loop1 harmed by at least one deeper loop: `11`
- depth hit patterns: `{'0000': 15, '1111': 10, '1110': 7, '1101': 1, '1100': 1, '0001': 3, '1011': 1, '0011': 3, '1000': 1, '0111': 1}`

### Best Simple Threshold Routers

- `loop1` margin < `0.75` -> loop `8`: correct `26/43`, delta vs loop1 `5`, routed deep `15`, W/L `6/1`
- `loop1` margin < `0.5` -> loop `8`: correct `25/43`, delta vs loop1 `4`, routed deep `13`, W/L `5/1`
- `base` margin < `1.0` -> loop `4`: correct `24/43`, delta vs loop1 `3`, routed deep `25`, W/L `4/1`
- `base` margin < `0.5` -> loop `8`: correct `24/43`, delta vs loop1 `3`, routed deep `15`, W/L `5/2`
- `loop1` margin < `1.0` -> loop `8`: correct `24/43`, delta vs loop1 `3`, routed deep `22`, W/L `7/4`
- `base` margin < `0.25` -> loop `8`: correct `23/43`, delta vs loop1 `2`, routed deep `10`, W/L `3/1`
- `loop1` margin < `0.5` -> loop `4`: correct `23/43`, delta vs loop1 `2`, routed deep `13`, W/L `3/1`
- `loop1` margin < `0.75` -> loop `4`: correct `23/43`, delta vs loop1 `2`, routed deep `15`, W/L `3/1`
- `loop1` margin < `1.0` -> loop `4`: correct `23/43`, delta vs loop1 `2`, routed deep `22`, W/L `4/2`
- `loop1` margin < `1.5` -> loop `4`: correct `23/43`, delta vs loop1 `2`, routed deep `25`, W/L `4/2`

### Best Score Selectors

- subset `[1, 2, 4, 8]` `mean`: correct `24/43`, delta vs loop1 `3`, W/L `3/0`, p `0.25`
- subset `[1, 2]` `max`: correct `22/43`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 4]` `max`: correct `22/43`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 4, 8]` `max`: correct `22/43`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 4, 8]` `loop1_plus_weighted_deeper:1.0`: correct `22/43`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1]` `mean`: correct `21/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `21/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `mean`: correct `21/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `21/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `21/43`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `8`: base `5`, by_loop `{1: 5, 2: 5, 4: 3, 8: 3}`, any recurrent `5`
- `low` n `15`: base `4`, by_loop `{1: 3, 2: 2, 4: 4, 8: 6}`, any recurrent `8`
- `thin` n `10`: base `7`, by_loop `{1: 6, 2: 6, 4: 8, 8: 4}`, any recurrent `8`
- `very_confident` n `10`: base `7`, by_loop `{1: 7, 2: 7, 4: 7, 8: 6}`, any recurrent `7`
