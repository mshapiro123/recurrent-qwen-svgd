# Depth Sweep Analysis - stage5_prelude_forced_depth_heldout_arc_loop1248

- Source: `outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 4, 8]`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `15/43`, base `11/43`, delta `4`, W/L/T `5/1/37`, p `0.21875`
- loop `2`: recurrent `10/43`, base `11/43`, delta `-1`, W/L/T `4/5/34`, p `1.0`
- loop `4`: recurrent `7/43`, base `11/43`, delta `-4`, W/L/T `4/8/31`, p `0.3876953125`
- loop `8`: recurrent `7/43`, base `11/43`, delta `-4`, W/L/T `3/7/33`, p `0.34375`

### Depth Interaction

- loop1 correct: `15/43`
- any recurrent depth correct: `19/43` (oracle gain vs loop1 `4`)
- base or any recurrent correct: `20/43` (oracle gain vs base `9`)
- deeper unique over loop1: `4`
- deeper unique over base+loop1: `4`
- loop1 harmed by at least one deeper loop: `11`
- depth hit patterns: `{'1000': 6, '0000': 24, '1100': 3, '0001': 1, '1001': 1, '1111': 4, '0011': 1, '0100': 1, '0110': 1, '1110': 1}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `4`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `8`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `4`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `8`: correct `15/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `2`: correct `12/43`, delta vs loop1 `-3`, routed deep `15`, W/L `1/4`
- `base` margin < `0.5` -> loop `2`: correct `12/43`, delta vs loop1 `-3`, routed deep `25`, W/L `1/4`
- `base` margin < `1.5` -> loop `2`: correct `11/43`, delta vs loop1 `-4`, routed deep `36`, W/L `2/6`
- `base` margin < `2.0` -> loop `2`: correct `11/43`, delta vs loop1 `-4`, routed deep `41`, W/L `2/6`

### Best Score Selectors

- subset `[1]` `mean`: correct `15/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `15/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 4]` `loop1_plus_weighted_deeper:0.5`: correct `14/43`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2, 4]` `loop1_plus_weighted_deeper:0.75`: correct `14/43`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2, 4]` `loop1_plus_weighted_deeper:1.0`: correct `14/43`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `mean`: correct `13/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `13/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `13/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `13/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `13/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`

### Base Confidence Buckets

- `confident` n `8`: base `0`, by_loop `{1: 0, 2: 1, 4: 0, 8: 1}`, any recurrent `2`
- `low` n `25`: base `9`, by_loop `{1: 11, 2: 8, 4: 5, 8: 4}`, any recurrent `12`
- `thin` n `8`: base `2`, by_loop `{1: 3, 2: 1, 4: 2, 8: 2}`, any recurrent `4`
- `very_confident` n `2`: base `0`, by_loop `{1: 1, 2: 0, 4: 0, 8: 0}`, any recurrent `1`
