# Depth Sweep Analysis - stage5_heldout_router_validation_20260625_230408

- Source: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 3]`

## arc_easy

### Loop Summaries

- loop `1`: recurrent `76/128`, base `71/128`, delta `5`, W/L/T `8/3/117`, p `0.2265625`
- loop `2`: recurrent `75/128`, base `71/128`, delta `4`, W/L/T `10/6/112`, p `0.454498291015625`
- loop `3`: recurrent `62/128`, base `71/128`, delta `-9`, W/L/T `6/15/107`, p `0.0783538818359375`

### Depth Interaction

- loop1 correct: `76/128`
- any recurrent depth correct: `86/128` (oracle gain vs loop1 `10`)
- base or any recurrent correct: `86/128` (oracle gain vs base `15`)
- deeper unique over loop1: `10`
- deeper unique over base+loop1: `7`
- loop1 harmed by at least one deeper loop: `23`
- depth hit patterns: `{'111': 53, '000': 42, '110': 12, '011': 8, '100': 10, '010': 2, '101': 1}`

### Best Simple Threshold Routers

- `loop1` margin < `1.0` -> loop `2`: correct `77/128`, delta vs loop1 `1`, routed deep `76`, W/L `10/9`
- `base` margin < `0.0` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.5` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `50`, W/L `9/9`
- `base` margin < `0.75` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `61`, W/L `9/9`
- `base` margin < `0.0` -> loop `3`: correct `76/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.75` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `60`, W/L `9/9`
- `loop1` margin < `1.5` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `88`, W/L `10/10`
- `loop1` margin < `2.0` -> loop `2`: correct `76/128`, delta vs loop1 `0`, routed deep `102`, W/L `10/10`
- `loop1` margin < `0.0` -> loop `3`: correct `76/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`

### Best Score Selectors

- subset `[1, 2]` `max`: correct `78/128`, delta vs loop1 `2`, W/L `4/2`, p `0.6875`
- subset `[1, 2, 3]` `max`: correct `78/128`, delta vs loop1 `2`, W/L `4/2`, p `0.6875`
- subset `[1]` `mean`: correct `76/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `76/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `76/128`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `76/128`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `76/128`, delta vs loop1 `0`, W/L `3/3`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `75/128`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `75/128`, delta vs loop1 `-1`, W/L `2/3`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `75/128`, delta vs loop1 `-1`, W/L `3/4`, p `1.0`

### Base Confidence Buckets

- `confident` n `26`: base `18`, by_loop `{1: 19, 2: 19, 3: 14}`, any recurrent `19`
- `low` n `50`: base `18`, by_loop `{1: 22, 2: 22, 3: 19}`, any recurrent `31`
- `thin` n `24`: base `11`, by_loop `{1: 11, 2: 10, 3: 10}`, any recurrent `11`
- `very_confident` n `28`: base `24`, by_loop `{1: 24, 2: 24, 3: 19}`, any recurrent `25`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `11/43`, base `11/43`, delta `0`, W/L/T `1/1/41`, p `1.0`
- loop `2`: recurrent `8/43`, base `11/43`, delta `-3`, W/L/T `2/5/36`, p `0.453125`
- loop `3`: recurrent `8/43`, base `11/43`, delta `-3`, W/L/T `4/7/32`, p `0.548828125`

### Depth Interaction

- loop1 correct: `11/43`
- any recurrent depth correct: `15/43` (oracle gain vs loop1 `4`)
- base or any recurrent correct: `16/43` (oracle gain vs base `5`)
- deeper unique over loop1: `4`
- deeper unique over base+loop1: `4`
- loop1 harmed by at least one deeper loop: `7`
- depth hit patterns: `{'000': 28, '100': 5, '111': 4, '001': 2, '110': 2, '011': 2}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `11/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.5` -> loop `2`: correct `10/43`, delta vs loop1 `-1`, routed deep `23`, W/L `2/3`
- `loop1` margin < `0.25` -> loop `2`: correct `10/43`, delta vs loop1 `-1`, routed deep `14`, W/L `1/2`
- `base` margin < `0.25` -> loop `2`: correct `9/43`, delta vs loop1 `-2`, routed deep `15`, W/L `1/3`
- `loop1` margin < `0.5` -> loop `2`: correct `9/43`, delta vs loop1 `-2`, routed deep `24`, W/L `2/4`
- `base` margin < `0.75` -> loop `2`: correct `8/43`, delta vs loop1 `-3`, routed deep `32`, W/L `2/5`
- `base` margin < `1.0` -> loop `2`: correct `8/43`, delta vs loop1 `-3`, routed deep `33`, W/L `2/5`

### Best Score Selectors

- subset `[1]` `mean`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `max`: correct `11/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `11/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.25`: correct `10/43`, delta vs loop1 `-1`, W/L `0/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:1.0`: correct `10/43`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `mean`: correct `9/43`, delta vs loop1 `-2`, W/L `0/2`, p `0.5`

### Base Confidence Buckets

- `confident` n `8`: base `0`, by_loop `{1: 0, 2: 0, 3: 0}`, any recurrent `0`
- `low` n `23`: base `8`, by_loop `{1: 8, 2: 7, 3: 5}`, any recurrent `10`
- `thin` n `10`: base `3`, by_loop `{1: 3, 2: 1, 3: 3}`, any recurrent `5`
- `very_confident` n `2`: base `0`, by_loop `{1: 0, 2: 0, 3: 0}`, any recurrent `0`

## open_hard_arc_challenge

### Loop Summaries

- loop `1`: recurrent `40/128`, base `39/128`, delta `1`, W/L/T `2/1/125`, p `1.0`
- loop `2`: recurrent `37/128`, base `39/128`, delta `-2`, W/L/T `9/11/108`, p `0.8238029479980469`
- loop `3`: recurrent `33/128`, base `39/128`, delta `-6`, W/L/T `10/16/102`, p `0.32693958282470703`

### Depth Interaction

- loop1 correct: `40/128`
- any recurrent depth correct: `52/128` (oracle gain vs loop1 `12`)
- base or any recurrent correct: `52/128` (oracle gain vs base `13`)
- deeper unique over loop1: `12`
- deeper unique over base+loop1: `11`
- loop1 harmed by at least one deeper loop: `21`
- depth hit patterns: `{'000': 76, '111': 19, '100': 9, '011': 8, '101': 3, '110': 9, '001': 3, '010': 1}`

### Best Simple Threshold Routers

- `loop1` margin < `0.25` -> loop `2`: correct `43/128`, delta vs loop1 `3`, routed deep `37`, W/L `7/4`
- `loop1` margin < `0.25` -> loop `3`: correct `42/128`, delta vs loop1 `2`, routed deep `37`, W/L `6/4`
- `base` margin < `0.25` -> loop `2`: correct `41/128`, delta vs loop1 `1`, routed deep `41`, W/L `6/5`
- `base` margin < `0.25` -> loop `3`: correct `41/128`, delta vs loop1 `1`, routed deep `41`, W/L `5/4`
- `base` margin < `0.0` -> loop `2`: correct `40/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `40/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `40/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `40/128`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.5` -> loop `2`: correct `39/128`, delta vs loop1 `-1`, routed deep `65`, W/L `7/8`
- `base` margin < `0.75` -> loop `2`: correct `39/128`, delta vs loop1 `-1`, routed deep `82`, W/L `8/9`

### Best Score Selectors

- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `43/128`, delta vs loop1 `3`, W/L `5/2`, p `0.453125`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.25`: correct `43/128`, delta vs loop1 `3`, W/L `4/1`, p `0.375`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `42/128`, delta vs loop1 `2`, W/L `3/1`, p `0.625`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `42/128`, delta vs loop1 `2`, W/L `5/3`, p `0.7265625`
- subset `[1, 2]` `mean`: correct `41/128`, delta vs loop1 `1`, W/L `6/5`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `41/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `41/128`, delta vs loop1 `1`, W/L `6/5`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `41/128`, delta vs loop1 `1`, W/L `6/5`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `41/128`, delta vs loop1 `1`, W/L `2/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:1.0`: correct `41/128`, delta vs loop1 `1`, W/L `6/5`, p `1.0`

### Base Confidence Buckets

- `confident` n `26`: base `10`, by_loop `{1: 10, 2: 8, 3: 8}`, any recurrent `12`
- `low` n `65`: base `17`, by_loop `{1: 18, 2: 17, 3: 16}`, any recurrent `26`
- `thin` n `28`: base `10`, by_loop `{1: 10, 2: 10, 3: 7}`, any recurrent `11`
- `very_confident` n `9`: base `2`, by_loop `{1: 2, 2: 2, 3: 2}`, any recurrent `3`
