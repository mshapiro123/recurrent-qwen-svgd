# Rescue Selector Transfer - stage5_rescue_selector_transfer_20260627_185546

- Discovery: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Held-out: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Discovery benchmark: `arc_challenge`
- Score target / aggregate: `content_question_only` / `mean`

## Discovery Spectral Diagnostics

- `rescuable_vs_harmable`: dim `8`, diff norm `1.048953313111271`, top energy `{'1': 0.7357220691292224, '2': 0.9837237209101612, '4': 0.9970257677028462, '8': 0.9999999999999999}`, tail-half energy `0.0029742322971535624`
- `rescuable_vs_rest`: dim `8`, diff norm `1.0615184233176327`, top energy `{'1': 0.24508290600438104, '2': 0.93437283036038, '4': 0.9987667449293857, '8': 1.0}`, tail-half energy `0.0012332550706143956`

## Diverse-Probe Detectability

- Available: `True`
- Observed alignment: `0.8812367490001952`
- Null p95 alignment: `0.8865604701957049`
- Clears null p95: `False`

## Discovery Supervised Probe Candidates

- `whitened_rescue_score_shrinkage_0.1`: discovery max-net delta `7`, rescue/harm `24/17`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_1`: discovery max-net delta `8`, rescue/harm `21/13`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_10`: discovery max-net delta `7`, rescue/harm `13/6`; zero-harm delta `None`, rescue/harm `None/None`

## Discovery Policies

- `max_net`: `base_predicted_margin` low `0.3647139072418213` -> loop `3`; discovery delta `8`, rescue/harm `17/9`, routed `103`
- `zero_harm`: `loop1_mean_expected_loops` high `1.6781634986400604` -> loop `2`; discovery delta `1`, rescue/harm `1/0`, routed `14`
- `harm_budget_1`: `loop1_mean_expected_loops` high `1.5409627258777618` -> loop `3`; discovery delta `3`, rescue/harm `4/1`, routed `39`
- `harm_budget_2`: `loop1_mean_halt_entropy` high `0.6798350960016251` -> loop `2`; discovery delta `3`, rescue/harm `5/2`, routed `39`
- `manual_base_margin_low_0.196586`: `base_predicted_margin` low `0.19658637046813965` -> loop `3`; discovery delta `None`, rescue/harm `None/None`, routed `None`
- `manual_base_margin_low_0.364714`: `base_predicted_margin` low `0.3647139072418213` -> loop `3`; discovery delta `None`, rescue/harm `None/None`, routed `None`
- `manual_base_margin_low_0.469864`: `base_predicted_margin` low `0.4698638916015625` -> loop `3`; discovery delta `None`, rescue/harm `None/None`, routed `None`

## Held-Out Results

### arc_easy

- Categories: `{'harmable': 23, 'rescuable': 10, 'stable_correct': 53, 'stable_wrong': 42}`
- Rescue AUCs: `base_predicted_margin` 0.7703389830508475, `loop1_predicted_margin` 0.7584745762711864, `loop1_score_entropy` 0.6889830508474576

- `max_net`: correct `75/128`, delta `-1`, gap capture `-0.125`, routed `36`, W/L `8/9`, rescue/harm `8/9`
- `zero_harm`: correct `76/128`, delta `0`, gap capture `0.0`, routed `3`, W/L `1/1`, rescue/harm `1/1`
- `harm_budget_1`: correct `76/128`, delta `0`, gap capture `0.0`, routed `7`, W/L `1/1`, rescue/harm `1/1`
- `harm_budget_2`: correct `77/128`, delta `1`, gap capture `0.1`, routed `9`, W/L `1/0`, rescue/harm `1/0`
- `manual_base_margin_low_0.196586`: correct `72/128`, delta `-4`, gap capture `-0.5`, routed `20`, W/L `2/6`, rescue/harm `2/6`
- `manual_base_margin_low_0.364714`: correct `75/128`, delta `-1`, gap capture `-0.125`, routed `36`, W/L `8/9`, rescue/harm `8/9`
- `manual_base_margin_low_0.469864`: correct `73/128`, delta `-3`, gap capture `-0.375`, routed `49`, W/L `8/11`, rescue/harm `8/11`

#### Transferred Curve Summary

- `max_net`: `loop1_score_entropy` high `1.1556681120664505` -> loop `2`; correct `80/128`, delta `4`, gap capture `0.4`, routed `42`, W/L `7/3`, rescue/harm `7/3`
- `zero_harm`: `loop1_predicted_margin` low `0.048300743103027344` -> loop `2`; correct `77/128`, delta `1`, gap capture `0.1`, routed `2`, W/L `1/0`, rescue/harm `1/0`
- `harm_budget_1`: `loop1_mean_halt_entropy` high `0.6104329377412796` -> loop `2`; correct `78/128`, delta `2`, gap capture `0.2`, routed `25`, W/L `3/1`, rescue/harm `3/1`
- `harm_budget_2`: `loop1_mean_halt_entropy` high `0.6104329377412796` -> loop `2`; correct `78/128`, delta `2`, gap capture `0.2`, routed `25`, W/L `3/1`, rescue/harm `3/1`

#### Supervised Probe Transfer

- `whitened_rescue_score_shrinkage_0.1`: max-net delta `2`, rescue/harm `9/7`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_1`: max-net delta `2`, rescue/harm `9/7`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_10`: max-net delta `2`, rescue/harm `5/3`; zero-harm delta `0`, rescue/harm `0/0`

### arc_challenge

- Categories: `{'harmable': 7, 'rescuable': 4, 'stable_correct': 4, 'stable_wrong': 28}`
- Rescue AUCs: `loop1_prediction_halt_entropy` 0.7948717948717949, `loop1_prediction_expected_loops` 0.7948717948717949, `loop1_mean_expected_loops` 0.7564102564102564

- `max_net`: correct `8/43`, delta `-3`, gap capture `-0.75`, routed `20`, W/L `2/5`, rescue/harm `2/5`
- `zero_harm`: correct `11/43`, delta `0`, gap capture `0.0`, routed `1`, W/L `0/0`, rescue/harm `0/0`
- `harm_budget_1`: correct `11/43`, delta `0`, gap capture `0.0`, routed `1`, W/L `0/0`, rescue/harm `0/0`
- `harm_budget_2`: correct `11/43`, delta `0`, gap capture `0.0`, routed `6`, W/L `0/0`, rescue/harm `0/0`
- `manual_base_margin_low_0.196586`: correct `7/43`, delta `-4`, gap capture `-1.0`, routed `13`, W/L `1/5`, rescue/harm `1/5`
- `manual_base_margin_low_0.364714`: correct `8/43`, delta `-3`, gap capture `-0.75`, routed `20`, W/L `2/5`, rescue/harm `2/5`
- `manual_base_margin_low_0.469864`: correct `8/43`, delta `-3`, gap capture `-0.75`, routed `23`, W/L `2/5`, rescue/harm `2/5`

#### Transferred Curve Summary

- `max_net`: `loop1_prediction_expected_loops` low `1.039893627166748` -> loop `3`; correct `13/43`, delta `2`, gap capture `0.5`, routed `13`, W/L `3/1`, rescue/harm `3/1`
- `zero_harm`: `loop1_score_entropy` low `1.0899640461932867` -> loop `3`; correct `13/43`, delta `2`, gap capture `0.5`, routed `15`, W/L `2/0`, rescue/harm `2/0`
- `harm_budget_1`: `loop1_prediction_expected_loops` low `1.039893627166748` -> loop `3`; correct `13/43`, delta `2`, gap capture `0.5`, routed `13`, W/L `3/1`, rescue/harm `3/1`
- `harm_budget_2`: `loop1_prediction_expected_loops` low `1.039893627166748` -> loop `3`; correct `13/43`, delta `2`, gap capture `0.5`, routed `13`, W/L `3/1`, rescue/harm `3/1`

#### Supervised Probe Transfer

- `whitened_rescue_score_shrinkage_0.1`: max-net delta `-1`, rescue/harm `1/2`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_1`: max-net delta `-1`, rescue/harm `1/2`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_10`: max-net delta `0`, rescue/harm `0/0`; zero-harm delta `0`, rescue/harm `0/0`

### open_hard_arc_challenge

- Categories: `{'harmable': 21, 'rescuable': 12, 'stable_correct': 19, 'stable_wrong': 76}`
- Rescue AUCs: `base_predicted_margin` 0.6602011494252873, `loop1_predicted_margin` 0.6264367816091954, `loop1_margin_minus_base_margin` 0.610632183908046

- `max_net`: correct `37/128`, delta `-3`, gap capture `-0.2727272727272727`, routed `57`, W/L `6/9`, rescue/harm `6/9`
- `zero_harm`: correct `40/128`, delta `0`, gap capture `0.0`, routed `6`, W/L `0/0`, rescue/harm `0/0`
- `harm_budget_1`: correct `39/128`, delta `-1`, gap capture `-0.09090909090909091`, routed `17`, W/L `1/2`, rescue/harm `1/2`
- `harm_budget_2`: correct `36/128`, delta `-4`, gap capture `-0.4444444444444444`, routed `19`, W/L `0/4`, rescue/harm `0/4`
- `manual_base_margin_low_0.196586`: correct `42/128`, delta `2`, gap capture `0.18181818181818182`, routed `36`, W/L `5/3`, rescue/harm `5/3`
- `manual_base_margin_low_0.364714`: correct `37/128`, delta `-3`, gap capture `-0.2727272727272727`, routed `57`, W/L `6/9`, rescue/harm `6/9`
- `manual_base_margin_low_0.469864`: correct `38/128`, delta `-2`, gap capture `-0.18181818181818182`, routed `65`, W/L `7/9`, rescue/harm `7/9`

#### Transferred Curve Summary

- `max_net`: `loop1_margin_minus_base_margin` high `0.027623891830444336` -> loop `2`; correct `44/128`, delta `4`, gap capture `0.4444444444444444`, routed `65`, W/L `7/3`, rescue/harm `7/3`
- `zero_harm`: `base_predicted_margin` low `0.04163932800292969` -> loop `3`; correct `44/128`, delta `4`, gap capture `0.36363636363636365`, routed `10`, W/L `4/0`, rescue/harm `4/0`
- `harm_budget_1`: `base_predicted_margin` low `0.08781838417053223` -> loop `3`; correct `44/128`, delta `4`, gap capture `0.36363636363636365`, routed `23`, W/L `5/1`, rescue/harm `5/1`
- `harm_budget_2`: `base_predicted_margin` low `0.08781838417053223` -> loop `2`; correct `44/128`, delta `4`, gap capture `0.4444444444444444`, routed `23`, W/L `6/2`, rescue/harm `6/2`

#### Supervised Probe Transfer

- `whitened_rescue_score_shrinkage_0.1`: max-net delta `1`, rescue/harm `1/0`; zero-harm delta `1`, rescue/harm `1/0`
- `whitened_rescue_score_shrinkage_1`: max-net delta `0`, rescue/harm `8/8`; zero-harm delta `None`, rescue/harm `None/None`
- `whitened_rescue_score_shrinkage_10`: max-net delta `1`, rescue/harm `7/6`; zero-harm delta `None`, rescue/harm `None/None`
