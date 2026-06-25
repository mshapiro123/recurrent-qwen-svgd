# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_loop123_20260625_194738

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Score target: `cyclic_label_aggregated`
- Aggregate: `permutation_mean`
- Loops: `[1, 2, 3]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `148/256`, base `154/256`, delta `-6`, W/L/T `3/9/244`, p `0.14599609375`
- loop `2`: recurrent `156/256`, base `154/256`, delta `2`, W/L/T `9/7/240`, p `0.803619384765625`
- loop `3`: recurrent `154/256`, base `154/256`, delta `0`, W/L/T `12/12/232`, p `1.0`

### Depth Interaction

- loop1 correct: `148/256`
- any recurrent depth correct: `164/256` (oracle gain vs loop1 `16`)
- base or any recurrent correct: `168/256` (oracle gain vs base `14`)
- deeper unique over loop1: `16`
- deeper unique over base+loop1: `11`
- loop1 harmed by at least one deeper loop: `11`
- depth hit patterns: `{'011': 9, '000': 92, '110': 9, '111': 137, '101': 2, '001': 6, '010': 1}`

### Best Simple Threshold Routers

- `base` margin < `1.5` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `158`, W/L `10/2`
- `base` margin < `2.0` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `183`, W/L `10/2`
- `base` margin < `0.75` -> loop `3`: correct `156/256`, delta vs loop1 `8`, routed deep `94`, W/L `15/7`
- `loop1` margin < `0.75` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `86`, W/L `10/2`
- `loop1` margin < `1.0` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `101`, W/L `10/2`
- `loop1` margin < `1.5` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `131`, W/L `10/2`
- `loop1` margin < `2.0` -> loop `2`: correct `156/256`, delta vs loop1 `8`, routed deep `161`, W/L `10/2`
- `loop1` margin < `0.25` -> loop `3`: correct `156/256`, delta vs loop1 `8`, routed deep `41`, W/L `12/4`
- `loop1` margin < `0.5` -> loop `3`: correct `156/256`, delta vs loop1 `8`, routed deep `59`, W/L `13/5`
- `base` margin < `0.5` -> loop `2`: correct `155/256`, delta vs loop1 `7`, routed deep `68`, W/L `9/2`

### Best Score Selectors

- subset `[1, 2, 3]` `mean`: correct `154/256`, delta vs loop1 `6`, W/L `6/0`, p `0.03125`
- subset `[1, 2]` `mean`: correct `151/256`, delta vs loop1 `3`, W/L `3/0`, p `0.25`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `151/256`, delta vs loop1 `3`, W/L `3/0`, p `0.25`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:1.0`: correct `151/256`, delta vs loop1 `3`, W/L `3/0`, p `0.25`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `150/256`, delta vs loop1 `2`, W/L `2/0`, p `0.5`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.75`: correct `150/256`, delta vs loop1 `2`, W/L `2/0`, p `0.5`
- subset `[1, 2]` `max`: correct `149/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `149/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `149/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 3]` `max`: correct `149/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`

### Base Confidence Buckets

- `confident` n `71`: base `42`, by_loop `{1: 42, 2: 43, 3: 41}`, any recurrent `43`
- `low` n `68`: base `26`, by_loop `{1: 21, 2: 28, 3: 28}`, any recurrent `34`
- `thin` n `44`: base `20`, by_loop `{1: 19, 2: 19, 3: 19}`, any recurrent `21`
- `very_confident` n `73`: base `66`, by_loop `{1: 66, 2: 66, 3: 66}`, any recurrent `66`
