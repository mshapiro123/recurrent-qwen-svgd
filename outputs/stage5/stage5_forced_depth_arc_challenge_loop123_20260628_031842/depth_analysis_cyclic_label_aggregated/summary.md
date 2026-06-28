# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_loop123_20260628_031842

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260628_031842/summary.json`
- Score target: `cyclic_label_aggregated`
- Aggregate: `permutation_mean`
- Loops: `[1, 2, 3, 4, 8]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `155/256`, base `155/256`, delta `0`, W/L/T `4/4/248`, p `1.0`
- loop `2`: recurrent `154/256`, base `155/256`, delta `-1`, W/L/T `9/10/237`, p `1.0`
- loop `3`: recurrent `148/256`, base `155/256`, delta `-7`, W/L/T `15/22/219`, p `0.3240086000878364`
- loop `4`: recurrent `141/256`, base `155/256`, delta `-14`, W/L/T `17/31/208`, p `0.05946337525377032`
- loop `8`: recurrent `71/256`, base `155/256`, delta `-84`, W/L/T `23/107/126`, p `3.7937066234622226e-14`

### Depth Interaction

- loop1 correct: `155/256`
- any recurrent depth correct: `197/256` (oracle gain vs loop1 `42`)
- base or any recurrent correct: `199/256` (oracle gain vs base `44`)
- deeper unique over loop1: `42`
- deeper unique over base+loop1: `40`
- loop1 harmed by at least one deeper loop: `111`
- depth hit patterns: `{'11111': 44, '00000': 59, '11000': 12, '01110': 5, '00111': 1, '11110': 75, '11100': 10, '10000': 6, '00010': 6, '00110': 4, '00001': 17, '11101': 1, '01111': 1, '10110': 1, '00100': 4, '11010': 1, '00011': 1, '00101': 1, '01101': 1, '11001': 3, '01000': 1, '10011': 1, '10010': 1}`

### Best Simple Threshold Routers

- `base` margin < `0.25` -> loop `2`: correct `156/256`, delta vs loop1 `1`, routed deep `43`, W/L `5/4`
- `base` margin < `0.0` -> loop `2`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.5` -> loop `2`: correct `155/256`, delta vs loop1 `0`, routed deep `69`, W/L `6/6`
- `base` margin < `0.0` -> loop `3`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `4`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `8`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `4`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `8`: correct `155/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`

### Best Score Selectors

- subset `[1, 2]` `max`: correct `156/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 3]` `max`: correct `156/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 3, 4]` `max`: correct `156/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 3, 4, 8]` `max`: correct `156/256`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1]` `mean`: correct `155/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `155/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `154/256`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `154/256`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `154/256`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `154/256`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`

### Base Confidence Buckets

- `confident` n `67`: base `41`, by_loop `{1: 41, 2: 41, 3: 40, 4: 37, 8: 23}`, any recurrent `52`
- `low` n `69`: base `28`, by_loop `{1: 28, 2: 28, 3: 23, 4: 24, 8: 10}`, any recurrent `44`
- `thin` n `45`: base `20`, by_loop `{1: 20, 2: 19, 3: 19, 4: 17, 8: 14}`, any recurrent `32`
- `very_confident` n `75`: base `66`, by_loop `{1: 66, 2: 66, 3: 66, 4: 63, 8: 24}`, any recurrent `69`
