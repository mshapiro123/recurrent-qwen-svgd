# Rescue Detectability Gate - stage5_rescue_detectability_unfreeze_20260628_content

- Source sweep: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260628_031842/summary.json`
- Benchmark: `arc_challenge`
- Score target: `content_question_only` / `mean`
- Loops: `[1, 2, 3, 4, 8]`
- Status: `passed`
- Category counts: `{'harmable': 51, 'rescuable': 44, 'stable_correct': 38, 'stable_wrong': 123}`

## Direction Agreement Gate

- Best shrinkage: `10.0`
- Observed agreement: `0.9594287568666126`
- Null mean agreement: `0.6731437928956148`
- Null p95 agreement: `0.8811519956216952`
- Observed minus null p95: `0.07827676124491734`
- Clears null p95: `True`

- shrinkage `0.1`: observed `0.8798661476366793`, null_p95 `0.864545099460084`, margin `0.015321048176595298`, clears `True`
- shrinkage `1.0`: observed `0.9369631968320298`, null_p95 `0.8631199886145046`, margin `0.07384320821752521`, clears `True`
- shrinkage `10.0`: observed `0.9594287568666126`, null_p95 `0.8811519956216952`, margin `0.07827676124491734`, clears `True`

## Supervised Probe Discovery Curves

### whitened_rescue_score_shrinkage_0.1
- harm_budget_1: correct `90`, delta `1`, rescue `2`, harm `1`, routed `14`
- harm_budget_2: correct `91`, delta `2`, rescue `4`, harm `2`, routed `14`
- max_net: correct `95`, delta `6`, rescue `25`, harm `19`, routed `172`

### whitened_rescue_score_shrinkage_1
- harm_budget_2: correct `92`, delta `3`, rescue `5`, harm `2`, routed `14`
- max_net: correct `96`, delta `7`, rescue `17`, harm `10`, routed `103`

### whitened_rescue_score_shrinkage_10
- harm_budget_2: correct `91`, delta `2`, rescue `4`, harm `2`, routed `14`
- max_net: correct `96`, delta `7`, rescue `13`, harm `6`, routed `128`
