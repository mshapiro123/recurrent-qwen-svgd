# Depth Sweep Analysis - stage5_depth_sweep_arc_loop1234_20260622_235932

- Source: `outputs/stage5/stage5_depth_sweep_arc_loop1234_20260622_235932/summary.json`
- Loops: `[1, 2, 3, 4]`

## arc_easy

### Loop Summaries

- loop `1`: recurrent `191/256`, base `186/256`, delta `5`, W/L/T `6/1/249`, p `0.125`
- loop `2`: recurrent `181/256`, base `186/256`, delta `-5`, W/L/T `15/20/221`, p `0.49955983320251107`
- loop `3`: recurrent `168/256`, base `186/256`, delta `-18`, W/L/T `14/32/210`, p `0.011351591436778108`
- loop `4`: recurrent `159/256`, base `186/256`, delta `-27`, W/L/T `13/40/203`, p `0.0002685401188191605`

### Depth Interaction

- loop1 correct: `191/256`
- any recurrent depth correct: `212/256` (oracle gain vs loop1 `21`)
- base or any recurrent correct: `212/256` (oracle gain vs base `26`)
- deeper unique over loop1: `21`
- deeper unique over base+loop1: `20`
- loop1 harmed by at least one deeper loop: `56`
- depth hit patterns: `{'1111': 135, '0000': 44, '1110': 13, '1000': 14, '1100': 19, '0001': 2, '1101': 2, '0111': 5, '0110': 2, '1001': 2, '0100': 5, '1011': 6, '0011': 7}`

### Best Simple Threshold Routers

- `base` margin < `0.0` -> loop `2`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `4`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `4`: correct `191/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `2`: correct `190/256`, delta vs loop1 `-1`, routed deep `10`, W/L `2/3`
- `loop1` margin < `0.25` -> loop `2`: correct `188/256`, delta vs loop1 `-3`, routed deep `14`, W/L `3/6`
- `loop1` margin < `0.5` -> loop `2`: correct `188/256`, delta vs loop1 `-3`, routed deep `25`, W/L `5/8`
- `base` margin < `0.5` -> loop `2`: correct `187/256`, delta vs loop1 `-4`, routed deep `30`, W/L `4/8`

### Best Score Selectors

- subset `[1]` `mean`: correct `191/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1]` `max`: correct `191/256`, delta vs loop1 `0`, W/L `0/0`, p `None`
- subset `[1, 2]` `max`: correct `191/256`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3]` `max`: correct `191/256`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2, 3, 4]` `max`: correct `191/256`, delta vs loop1 `0`, W/L `1/1`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `190/256`, delta vs loop1 `-1`, W/L `1/2`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `190/256`, delta vs loop1 `-1`, W/L `3/4`, p `1.0`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `189/256`, delta vs loop1 `-2`, W/L `1/3`, p `0.625`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `189/256`, delta vs loop1 `-2`, W/L `3/5`, p `0.7265625`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `189/256`, delta vs loop1 `-2`, W/L `1/3`, p `0.625`

### Base Confidence Buckets

- `confident` n `48`: base `30`, by_loop `{1: 30, 2: 28, 3: 23, 4: 22}`, any recurrent `34`
- `low` n `30`: base `14`, by_loop `{1: 19, 2: 15, 3: 9, 4: 9}`, any recurrent `25`
- `thin` n `38`: base `16`, by_loop `{1: 16, 2: 15, 3: 18, 4: 19}`, any recurrent `26`
- `very_confident` n `140`: base `126`, by_loop `{1: 126, 2: 123, 3: 118, 4: 109}`, any recurrent `127`

## arc_challenge

### Loop Summaries

- loop `1`: recurrent `150/256`, base `148/256`, delta `2`, W/L/T `6/4/246`, p `0.75390625`
- loop `2`: recurrent `141/256`, base `148/256`, delta `-7`, W/L/T `17/24/215`, p `0.34888887944907765`
- loop `3`: recurrent `133/256`, base `148/256`, delta `-15`, W/L/T `30/45/181`, p `0.10534226887460488`
- loop `4`: recurrent `129/256`, base `148/256`, delta `-19`, W/L/T `30/49/177`, p `0.042165443994951105`

### Depth Interaction

- loop1 correct: `150/256`
- any recurrent depth correct: `191/256` (oracle gain vs loop1 `41`)
- base or any recurrent correct: `192/256` (oracle gain vs base `44`)
- deeper unique over loop1: `41`
- deeper unique over base+loop1: `38`
- loop1 harmed by at least one deeper loop: `58`
- depth hit patterns: `{'1000': 16, '0000': 65, '1100': 25, '1111': 92, '1011': 5, '0111': 6, '0011': 15, '0010': 4, '0100': 6, '0001': 8, '1110': 9, '1101': 1, '0110': 2, '1001': 2}`

### Best Simple Threshold Routers

- `loop1` margin < `0.25` -> loop `2`: correct `152/256`, delta vs loop1 `2`, routed deep `23`, W/L `6/4`
- `base` margin < `0.0` -> loop `2`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.0` -> loop `3`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `3`: correct `150/256`, delta vs loop1 `0`, routed deep `29`, W/L `5/5`
- `base` margin < `0.0` -> loop `4`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `2`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.0` -> loop `3`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `loop1` margin < `0.25` -> loop `3`: correct `150/256`, delta vs loop1 `0`, routed deep `23`, W/L `4/4`
- `loop1` margin < `0.0` -> loop `4`: correct `150/256`, delta vs loop1 `0`, routed deep `0`, W/L `0/0`
- `base` margin < `0.25` -> loop `2`: correct `149/256`, delta vs loop1 `-1`, routed deep `29`, W/L `4/5`

### Best Score Selectors

- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.5`: correct `155/256`, delta vs loop1 `5`, W/L `6/1`, p `0.125`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.5`: correct `153/256`, delta vs loop1 `3`, W/L `6/3`, p `0.5078125`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.1`: correct `153/256`, delta vs loop1 `3`, W/L `4/1`, p `0.375`
- subset `[1, 2, 3]` `loop1_plus_weighted_deeper:0.25`: correct `153/256`, delta vs loop1 `3`, W/L `4/1`, p `0.375`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.1`: correct `152/256`, delta vs loop1 `2`, W/L `4/2`, p `0.6875`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.25`: correct `152/256`, delta vs loop1 `2`, W/L `4/2`, p `0.6875`
- subset `[1, 2]` `loop1_plus_weighted_deeper:0.75`: correct `152/256`, delta vs loop1 `2`, W/L `6/4`, p `0.75390625`
- subset `[1, 2, 3]` `mean`: correct `152/256`, delta vs loop1 `2`, W/L `10/8`, p `0.8145294189453125`
- subset `[1, 2, 3, 4]` `loop1_plus_weighted_deeper:0.1`: correct `152/256`, delta vs loop1 `2`, W/L `3/1`, p `0.625`
- subset `[1, 2, 3, 4]` `loop1_plus_weighted_deeper:0.25`: correct `152/256`, delta vs loop1 `2`, W/L `3/1`, p `0.625`

### Base Confidence Buckets

- `confident` n `54`: base `38`, by_loop `{1: 38, 2: 30, 3: 30, 4: 26}`, any recurrent `46`
- `low` n `62`: base `24`, by_loop `{1: 26, 2: 21, 3: 20, 4: 21}`, any recurrent `40`
- `thin` n `50`: base `13`, by_loop `{1: 13, 2: 17, 3: 17, 4: 16}`, any recurrent `27`
- `very_confident` n `90`: base `73`, by_loop `{1: 73, 2: 73, 3: 66, 4: 66}`, any recurrent `78`
