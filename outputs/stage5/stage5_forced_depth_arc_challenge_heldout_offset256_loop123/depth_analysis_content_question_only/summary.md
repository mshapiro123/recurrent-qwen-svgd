# Depth Sweep Analysis - stage5_forced_depth_arc_challenge_heldout_offset256_loop123

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_heldout_offset256_loop123/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 3]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `11/43`, base `11/43`, delta `0`, W/L/T `0/0/43`, p `None`
- loop `2`: recurrent `8/43`, base `11/43`, delta `-3`, W/L/T `1/4/38`, p `0.375`
- loop `3`: recurrent `10/43`, base `11/43`, delta `-1`, W/L/T `5/6/32`, p `1.0`

### Depth Interaction

- loop1 correct: `11/43`
- any recurrent depth correct: `16/43` (oracle gain vs loop1 `5`)
- base or any recurrent correct: `16/43` (oracle gain vs base `5`)
- deeper unique over loop1: `5`
- deeper unique over base+loop1: `5`
- loop1 harmed by at least one deeper loop: `6`
- depth hit patterns: `{'000': 27, '100': 4, '110': 2, '111': 5, '001': 4, '011': 1}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `2`: correct `10/43`, delta vs loop1 `-1`, routed deep `15`, W/L `1/2`
- `base` margin < `1.5` -> loop `3`: correct `10/43`, delta vs loop1 `-1`, routed deep `36`, W/L `5/6`
- `base` margin < `2.0` -> loop `3`: correct `10/43`, delta vs loop1 `-1`, routed deep `41`, W/L `5/6`
- `loop1` margin < `0.5` -> loop `3`: correct `10/43`, delta vs loop1 `-1`, routed deep `27`, W/L `4/5`
- `loop1` margin < `0.75` -> loop `3`: correct `10/43`, delta vs loop1 `-1`, routed deep `30`, W/L `4/5`
- `loop1` margin < `1.5` -> loop `3`: correct `10/43`, delta vs loop1 `-1`, routed deep `36`, W/L `5/6`

### Best Score Selectors

- subset `[1]` `mean`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `mean`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `8`: base `0`, by_loop `{1: 0, 2: 0, 3: 1}`, any recurrent `1`
- `low` n `25`: base `9`, by_loop `{1: 9, 2: 7, 3: 7}`, any recurrent `12`
- `thin` n `8`: base `2`, by_loop `{1: 2, 2: 1, 3: 2}`, any recurrent `3`
- `very_confident` n `2`: base `0`, by_loop `{1: 0, 2: 0, 3: 0}`, any recurrent `0`
