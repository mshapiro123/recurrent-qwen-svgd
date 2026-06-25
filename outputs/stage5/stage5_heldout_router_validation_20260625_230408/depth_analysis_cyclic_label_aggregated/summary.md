# Depth Sweep Analysis - stage5_heldout_router_validation_20260625_230408

- Source: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Score target: `cyclic_label_aggregated`
- Aggregate: `permutation_mean`
- Loops: `[1, 2, 3]`

## arc_easy

### Loop Summaries

- loop `1`: recurrent `103/128`, base `103/128`, delta `0`, W/L/T `1/1/126`, p `1.0`
- loop `2`: recurrent `104/128`, base `103/128`, delta `1`, W/L/T `3/2/123`, p `1.0`
- loop `3`: recurrent `97/128`, base `103/128`, delta `-6`, W/L/T `0/6/122`, p `0.03125`

### Depth Interaction

- loop1 correct: `103/128`
- any recurrent depth correct: `105/128` (oracle gain vs loop1 `2`)
- base or any recurrent correct: `106/128` (oracle gain vs base `3`)
- deeper unique over loop1: `2`
- deeper unique over base+loop1: `2`
- loop1 harmed by at least one deeper loop: `6`
- depth hit patterns: `{'111': 97, '110': 5, '010': 2, '000': 23, '100': 1}`

### Best Simple Threshold Routers

- `base` margin < `0.5` -> loop `2`: correct `105/128`, delta vs loop1 `2`, routed deep `20`, W/L `2/0`
- `base` margin < `0.75` -> loop `2`: correct `105/128`, delta vs loop1 `2`, routed deep `25`, W/L `2/0`
- `base` margin < `0.25` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `8`, W/L `1/0`
- `base` margin < `1.0` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `31`, W/L `2/1`
- `base` margin < `1.5` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `42`, W/L `2/1`
- `base` margin < `2.0` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `55`, W/L `2/1`
- `loop1` margin < `0.5` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `11`, W/L `2/1`
- `loop1` margin < `0.75` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `20`, W/L `2/1`
- `loop1` margin < `1.0` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `26`, W/L `2/1`
- `loop1` margin < `1.5` -> loop `2`: correct `104/128`, delta vs loop1 `1`, routed deep `38`, W/L `2/1`

### Best Score Selectors

- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `104/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `104/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `104/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `104/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1]` `mean`: correct `103/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `103/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `mean`: correct `103/128`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `max`: correct `103/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `103/128`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `max`: correct `103/128`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `24`: base `14`, by_loop `{1: 14, 2: 14, 3: 14}`, any recurrent `14`
- `low` n `20`: base `9`, by_loop `{1: 9, 2: 11, 3: 6}`, any recurrent `11`
- `thin` n `11`: base `8`, by_loop `{1: 8, 2: 7, 3: 5}`, any recurrent `8`
- `very_confident` n `73`: base `72`, by_loop `{1: 72, 2: 72, 3: 72}`, any recurrent `72`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `23/43`, base `23/43`, delta `0`, W/L/T `1/1/41`, p `1.0`
- loop `2`: recurrent `20/43`, base `23/43`, delta `-3`, W/L/T `0/3/40`, p `0.25`
- loop `3`: recurrent `21/43`, base `23/43`, delta `-2`, W/L/T `1/3/39`, p `0.625`

### Depth Interaction

- loop1 correct: `23/43`
- any recurrent depth correct: `25/43` (oracle gain vs loop1 `2`)
- base or any recurrent correct: `25/43` (oracle gain vs base `2`)
- deeper unique over loop1: `2`
- deeper unique over base+loop1: `1`
- loop1 harmed by at least one deeper loop: `4`
- depth hit patterns: `{'000': 18, '111': 19, '001': 2, '110': 1, '100': 3}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `23/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `23/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `23/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `23/43`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `9`, W/L `1/3`
- `base` margin < `0.5` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `15`, W/L `2/4`
- `base` margin < `0.75` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `21`, W/L `2/4`
- `base` margin < `1.0` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `25`, W/L `2/4`
- `base` margin < `1.5` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `31`, W/L `2/4`
- `base` margin < `2.0` -> loop `3`: correct `21/43`, delta vs loop1 `-2`, routed deep `33`, W/L `2/4`

### Best Score Selectors

- subset `[1]` `mean`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `mean`: correct `23/43`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.25`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `23/43`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `8`: base `5`, by_loop `{1: 5, 2: 5, 3: 5}`, any recurrent `5`
- `low` n `15`: base `4`, by_loop `{1: 4, 2: 1, 3: 2}`, any recurrent `6`
- `thin` n `10`: base `7`, by_loop `{1: 7, 2: 7, 3: 7}`, any recurrent `7`
- `very_confident` n `10`: base `7`, by_loop `{1: 7, 2: 7, 3: 7}`, any recurrent `7`

## open_hard_arc_challenge

### Loop Summaries

- loop `1`: recurrent `69/128`, base `75/128`, delta `-6`, W/L/T `1/7/120`, p `0.0703125`
- loop `2`: recurrent `75/128`, base `75/128`, delta `0`, W/L/T `2/2/124`, p `1.0`
- loop `3`: recurrent `73/128`, base `75/128`, delta `-2`, W/L/T `3/5/120`, p `0.7265625`

### Depth Interaction

- loop1 correct: `69/128`
- any recurrent depth correct: `76/128` (oracle gain vs loop1 `7`)
- base or any recurrent correct: `78/128` (oracle gain vs base `3`)
- deeper unique over loop1: `7`
- deeper unique over base+loop1: `2`
- loop1 harmed by at least one deeper loop: `1`
- depth hit patterns: `{'111': 68, '001': 1, '000': 52, '010': 2, '110': 1, '011': 4}`

### Best Simple Threshold Routers

- `base` margin < `0.75` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `45`, W/L `6/0`
- `base` margin < `1.0` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `54`, W/L `6/0`
- `base` margin < `1.5` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `68`, W/L `6/0`
- `base` margin < `2.0` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `84`, W/L `6/0`
- `loop1` margin < `0.25` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `19`, W/L `6/0`
- `loop1` margin < `0.5` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `26`, W/L `6/0`
- `loop1` margin < `0.75` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `34`, W/L `6/0`
- `loop1` margin < `1.0` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `45`, W/L `6/0`
- `loop1` margin < `1.5` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `61`, W/L `6/0`
- `loop1` margin < `2.0` -> loop `2`: correct `75/128`, delta vs loop1 `6`, routed deep `74`, W/L `6/0`

### Best Score Selectors

- subset `[1, 2, 3]` `mean`: correct `72/128`, delta vs loop1 `3`, W/L `3/0`, p `0.25`
- subset `[1, 2]` `mean`: correct `70/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:1.0`: correct `70/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:1.0`: correct `70/128`, delta vs loop1 `1`, W/L `1/0`, p `1.0`
- subset `[1]` `mean`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `max`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `69/128`, delta vs loop1 `0`, W/L `0/0`, p `None`

### Base Confidence Buckets

- `confident` n `30`: base `19`, by_loop `{1: 19, 2: 19, 3: 19}`, any recurrent `19`
- `low` n `33`: base `10`, by_loop `{1: 5, 2: 10, 3: 8}`, any recurrent `11`
- `thin` n `21`: base `9`, by_loop `{1: 8, 2: 9, 3: 9}`, any recurrent `9`
- `very_confident` n `44`: base `37`, by_loop `{1: 37, 2: 37, 3: 37}`, any recurrent `37`
