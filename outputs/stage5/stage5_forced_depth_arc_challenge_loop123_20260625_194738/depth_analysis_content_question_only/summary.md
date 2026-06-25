# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_loop123_20260625_194738

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 3]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `89/256`, base `87/256`, delta `2`, W/L/T `7/5/244`, p `0.7744140625`
- loop `2`: recurrent `85/256`, base `87/256`, delta `-2`, W/L/T `20/22/214`, p `0.8776143287523155`
- loop `3`: recurrent `87/256`, base `87/256`, delta `0`, W/L/T `29/29/198`, p `1.0`

### Depth Interaction

- loop1 correct: `89/256`
- any recurrent depth correct: `117/256` (oracle gain vs loop1 `28`)
- base or any recurrent correct: `119/256` (oracle gain vs base `32`)
- deeper unique over loop1: `28`
- deeper unique over base+loop1: `25`
- loop1 harmed by at least one deeper loop: `31`
- depth hit patterns: `{'000': 139, '111': 58, '011': 14, '100': 17, '101': 5, '110': 9, '001': 10, '010': 4}`

### Best Simple Threshold Routers

- `base` margin < `0.5` -> loop `3`: correct `96/256`, delta vs loop1 `7`, routed deep `135`, W/L `20/13`
- `base` margin < `0.25` -> loop `3`: correct `95/256`, delta vs loop1 `6`, routed deep `76`, W/L `14/8`
- `loop1` margin < `0.25` -> loop `3`: correct `95/256`, delta vs loop1 `6`, routed deep `72`, W/L `12/6`
- `base` margin < `0.5` -> loop `2`: correct `94/256`, delta vs loop1 `5`, routed deep `135`, W/L `17/12`
- `loop1` margin < `0.75` -> loop `3`: correct `94/256`, delta vs loop1 `5`, routed deep `163`, W/L `22/17`
- `loop1` margin < `0.25` -> loop `2`: correct `93/256`, delta vs loop1 `4`, routed deep `72`, W/L `11/7`
- `loop1` margin < `0.5` -> loop `3`: correct `93/256`, delta vs loop1 `4`, routed deep `119`, W/L `16/12`
- `base` margin < `0.25` -> loop `2`: correct `91/256`, delta vs loop1 `2`, routed deep `76`, W/L `10/8`
- `base` margin < `0.75` -> loop `3`: correct `91/256`, delta vs loop1 `2`, routed deep `180`, W/L `23/21`
- `loop1` margin < `0.75` -> loop `2`: correct `91/256`, delta vs loop1 `2`, routed deep `163`, W/L `18/16`

### Best Score Selectors

- subset `[1, 2]` `max`: correct `93/256`, delta vs loop1 `4`, W/L `11/7`, p `0.480682373046875`
- subset `[1, 2, 3]` `max`: correct `92/256`, delta vs loop1 `3`, W/L `10/7`, p `0.629058837890625`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `91/256`, delta vs loop1 `2`, W/L `7/5`, p `0.7744140625`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.75`: correct `91/256`, delta vs loop1 `2`, W/L `8/6`, p `0.79052734375`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:1.0`: correct `91/256`, delta vs loop1 `2`, W/L `10/8`, p `0.8145294189453125`
- subset `[1, 2]` `mean`: correct `90/256`, delta vs loop1 `1`, W/L `7/6`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `90/256`, delta vs loop1 `1`, W/L `7/6`, p `1.0`
- subset `[1]` `mean`: correct `89/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `89/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `89/256`, delta vs loop1 `0`, W/L `1/1`, p `1.0`

### Base Confidence Buckets

- `confident` n `41`: base `21`, by_loop `{1: 21, 2: 19, 3: 20}`, any recurrent `22`
- `low` n `135`: base `37`, by_loop `{1: 40, 2: 45, 3: 47}`, any recurrent `63`
- `thin` n `66`: base `25`, by_loop `{1: 24, 2: 18, 3: 17}`, any recurrent `28`
- `very_confident` n `14`: base `4`, by_loop `{1: 4, 2: 3, 3: 3}`, any recurrent `4`
