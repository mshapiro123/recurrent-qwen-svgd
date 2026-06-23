# Depth Selector Split - stage5_depth_sweep_arc_loop1234_20260622_235932

- Source: `outputs/stage5/stage5_depth_sweep_arc_loop1234_20260622_235932/summary.json`
- Seed: `0`
- Train fraction: `0.5`
- Loops: `[1, 2, 3, 4]`

## arc_easy

### Baselines

- train loop1: `99/128` (base `96/128`, delta `3`)
- test loop1: `92/128` (base `90/128`, delta `2`)
- train any-depth oracle: `106/128` (gain vs loop1 `7`)
- test any-depth oracle: `106/128` (gain vs loop1 `14`)

### Selector Chosen On Train

- family: `score_selector`
- train correct: `100/128` (delta vs loop1 `1`, W/L `1/0`, p `1.0`)
- test correct: `91/128` (delta vs loop1 `-1`, W/L `0/1`, p `1.0`)

- subset: `[1, 2]`
- method: `max`

### Test Prediction Bias

- top prediction: `B`
- top prediction fraction: `0.375`

## arc_challenge

### Baselines

- train loop1: `77/128` (base `75/128`, delta `2`)
- test loop1: `73/128` (base `73/128`, delta `0`)
- train any-depth oracle: `103/128` (gain vs loop1 `26`)
- test any-depth oracle: `88/128` (gain vs loop1 `15`)

### Selector Chosen On Train

- family: `score_selector`
- train correct: `81/128` (delta vs loop1 `4`, W/L `5/1`, p `0.21875`)
- test correct: `74/128` (delta vs loop1 `1`, W/L `1/0`, p `1.0`)

- subset: `[1, 2, 3]`
- method: `loop1_plus_weighted_deeper:0.5`

### Test Prediction Bias

- top prediction: `B`
- top prediction fraction: `0.297`
