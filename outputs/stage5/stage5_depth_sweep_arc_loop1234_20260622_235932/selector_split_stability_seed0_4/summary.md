# Depth Selector Split Stability - stage5_depth_sweep_arc_loop1234_20260622_235932

- Source: `outputs/stage5/stage5_depth_sweep_arc_loop1234_20260622_235932/summary.json`
- Seeds: `[0, 1, 2, 3, 4]`
- Train fraction: `0.5`
- Loops: `[1, 2, 3, 4]`

## arc_challenge

- splits: `5`
- test total per split: `128`
- mean base correct: `75.400`
- mean loop1 correct: `75.600`
- mean selected correct: `77.000`
- mean selected delta vs loop1: `1.400` (min `0`, max `3`)
- split signs: +`4` / 0`1` / -`0`
- mean any-depth oracle gain vs loop1: `21.800`
- mean W/L vs loop1: `1.600`/`0.200`

### Selector Choices

- `score:loop1_plus_weighted_deeper:0.5[1,2,3]`: `3`
- `score:loop1_plus_weighted_deeper:0.1[1,2,3]`: `1`
- `score:max[1,2]`: `1`

### Per Seed

- seed `0`: selected `74`, loop1 `73`, base `73`, delta `1`, W/L `1/0`, oracle gain `15`, selector `score:loop1_plus_weighted_deeper:0.5[1,2,3]`
- seed `1`: selected `77`, loop1 `75`, base `74`, delta `2`, W/L `2/0`, oracle gain `26`, selector `score:loop1_plus_weighted_deeper:0.1[1,2,3]`
- seed `2`: selected `81`, loop1 `80`, base `80`, delta `1`, W/L `2/1`, oracle gain `21`, selector `score:loop1_plus_weighted_deeper:0.5[1,2,3]`
- seed `3`: selected `80`, loop1 `77`, base `78`, delta `3`, W/L `3/0`, oracle gain `21`, selector `score:loop1_plus_weighted_deeper:0.5[1,2,3]`
- seed `4`: selected `73`, loop1 `73`, base `72`, delta `0`, W/L `0/0`, oracle gain `26`, selector `score:max[1,2]`

## arc_easy

- splits: `5`
- test total per split: `128`
- mean base correct: `93.400`
- mean loop1 correct: `95.600`
- mean selected correct: `93.800`
- mean selected delta vs loop1: `-1.800` (min `-6`, max `0`)
- split signs: +`0` / 0`1` / -`4`
- mean any-depth oracle gain vs loop1: `10.600`
- mean W/L vs loop1: `0.000`/`1.800`

### Selector Choices

- `score:max[1,2]`: `3`
- `score:loop1_plus_weighted_deeper:1.0[1,2,3,4]`: `1`
- `threshold:base<0.0->loop2`: `1`

### Per Seed

- seed `0`: selected `91`, loop1 `92`, base `90`, delta `-1`, W/L `0/1`, oracle gain `14`, selector `score:max[1,2]`
- seed `1`: selected `92`, loop1 `93`, base `92`, delta `-1`, W/L `0/1`, oracle gain `11`, selector `score:max[1,2]`
- seed `2`: selected `99`, loop1 `105`, base `100`, delta `-6`, W/L `0/6`, oracle gain `5`, selector `score:loop1_plus_weighted_deeper:1.0[1,2,3,4]`
- seed `3`: selected `91`, loop1 `91`, base `91`, delta `0`, W/L `0/0`, oracle gain `11`, selector `threshold:base<0.0->loop2`
- seed `4`: selected `96`, loop1 `97`, base `94`, delta `-1`, W/L `0/1`, oracle gain `12`, selector `score:max[1,2]`
