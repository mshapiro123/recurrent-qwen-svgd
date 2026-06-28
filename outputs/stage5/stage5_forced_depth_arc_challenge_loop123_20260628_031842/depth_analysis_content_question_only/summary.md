# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_loop123_20260628_031842

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260628_031842/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 3, 4, 8]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `89/256`, base `88/256`, delta `1`, W/L/T `3/2/251`, p `1.0`
- loop `2`: recurrent `87/256`, base `88/256`, delta `-1`, W/L/T `18/19/219`, p `1.0`
- loop `3`: recurrent `83/256`, base `88/256`, delta `-5`, W/L/T `26/31/199`, p `0.5966417603730826`
- loop `4`: recurrent `75/256`, base `88/256`, delta `-13`, W/L/T `26/39/191`, p `0.13603239487789212`
- loop `8`: recurrent `73/256`, base `88/256`, delta `-15`, W/L/T `29/44/183`, p `0.1006436775202357`

### Depth Interaction

- loop1 correct: `89/256`
- any recurrent depth correct: `133/256` (oracle gain vs loop1 `44`)
- base or any recurrent correct: `133/256` (oracle gain vs base `45`)
- deeper unique over loop1: `44`
- deeper unique over base+loop1: `42`
- loop1 harmed by at least one deeper loop: `51`
- depth hit patterns: `{'00000': 123, '11100': 8, '01111': 5, '00001': 12, '11000': 13, '00110': 2, '11110': 9, '10100': 2, '11111': 38, '11001': 2, '00111': 8, '01000': 2, '00010': 3, '10001': 3, '01110': 7, '10000': 12, '10011': 1, '00011': 1, '01001': 1, '01100': 2, '00101': 1, '10111': 1}`

### Best Simple Threshold Routers

- `base` margin < `0.5` -> loop `3`: correct `94/256`, delta vs loop1 `5`, routed deep `134`, W/L `21/16`
- `loop1` margin < `0.25` -> loop `3`: correct `94/256`, delta vs loop1 `5`, routed deep `76`, W/L `13/8`
- `base` margin < `0.25` -> loop `3`: correct `93/256`, delta vs loop1 `4`, routed deep `81`, W/L `14/10`
- `loop1` margin < `0.5` -> loop `3`: correct `92/256`, delta vs loop1 `3`, routed deep `129`, W/L `20/17`
- `loop1` margin < `0.25` -> loop `4`: correct `92/256`, delta vs loop1 `3`, routed deep `76`, W/L `14/11`
- `loop1` margin < `0.25` -> loop `8`: correct `91/256`, delta vs loop1 `2`, routed deep `76`, W/L `12/10`
- `base` margin < `0.25` -> loop `2`: correct `90/256`, delta vs loop1 `1`, routed deep `81`, W/L `9/8`
- `base` margin < `0.25` -> loop `4`: correct `90/256`, delta vs loop1 `1`, routed deep `81`, W/L `15/14`
- `loop1` margin < `0.25` -> loop `2`: correct `90/256`, delta vs loop1 `1`, routed deep `76`, W/L `8/7`
- `base` margin < `0.0` -> loop `2`: correct `89/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`

### Best Score Selectors

- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `90/256`, delta vs loop1 `1`, W/L `2/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.25`: correct `90/256`, delta vs loop1 `1`, W/L `4/3`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `90/256`, delta vs loop1 `1`, W/L `7/6`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.75`: correct `90/256`, delta vs loop1 `1`, W/L `9/8`, p `1.0`
- subset `[1, 2, 3, 4]` `loop1_plus_weighted_deeper:0.1`: correct `90/256`, delta vs loop1 `1`, W/L `2/1`, p `1.0`
- subset `[1, 2, 3, 4, 8]` `loop1_plus_weighted_deeper:0.1`: correct `90/256`, delta vs loop1 `1`, W/L `2/1`, p `1.0`
- subset `[1]` `mean`: correct `89/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `89/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `89/256`, delta vs loop1 `0`, W/L `3/3`, p `1.0`
- subset `[1, 2, 3]` `mean`: correct `89/256`, delta vs loop1 `0`, W/L `14/14`, p `1.0`

### Base Confidence Buckets

- `confident` n `40`: base `20`, by_loop `{1: 20, 2: 21, 3: 20, 4: 19, 8: 17}`, any recurrent `22`
- `low` n `134`: base `39`, by_loop `{1: 40, 2: 40, 3: 45, 4: 37, 8: 38}`, any recurrent `74`
- `thin` n `67`: base `24`, by_loop `{1: 24, 2: 22, 3: 14, 4: 15, 8: 15}`, any recurrent `32`
- `very_confident` n `15`: base `5`, by_loop `{1: 5, 2: 4, 3: 4, 4: 4, 8: 3}`, any recurrent `5`
