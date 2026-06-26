# Latent Criticality Probe - stage5_latent_criticality_20260626_032132

- Discovery sweep: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Held-out sweep: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Max examples per benchmark: `64`
- Jacobian examples per benchmark: `8`
- Jacobian method: `finite_difference_random_gain`

## Top Transfer Selectors

- `participation_ratio`/min: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `effective_rank`/min: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `dragon_king_gap`/max: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `dragon_king_z`/max: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `state_rms`/min: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `pooled_norm`/min: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `logit_entropy`/min: selected `69/171`, loop1 `69/171`, oracle `81/171`, delta `0`, capture `0.0`
- `jacobian_gain_mean`/max: selected `9/24`, loop1 `9/24`, oracle `11/24`, delta `0`, capture `0.0`
- `jacobian_gain_max`/max: selected `9/24`, loop1 `9/24`, oracle `11/24`, delta `0`, capture `0.0`
- `logit_top_prob`/min: selected `62/171`, loop1 `69/171`, oracle `81/171`, delta `-7`, capture `-0.5833333333333334`

## Feature AUC

### discovery
- `participation_ratio`: `0.5544464609800362`
- `effective_rank`: `0.5590970961887477`
- `dragon_king_gap`: `0.4833257713248639`
- `dragon_king_z`: `0.518489110707804`
- `state_rms`: `0.4050589836660617`
- `pooled_norm`: `0.4109573502722323`
- `move_from_prev`: `0.5783854166666667`
- `move_to_next`: `0.5969129554655871`
- `decel`: `0.5833333333333334`
- `logit_entropy`: `0.5191696914700544`
- `logit_top_prob`: `0.4726633393829401`
- `logit_top2_margin`: `0.477540834845735`
- `jacobian_gain_mean`: `0.18518518518518517`
- `jacobian_gain_max`: `0.18518518518518517`

### heldout
- `participation_ratio`: `0.47907007332872975`
- `effective_rank`: `0.4819802045312551`
- `dragon_king_gap`: `0.5357436453914702`
- `dragon_king_z`: `0.543010752688172`
- `state_rms`: `0.4948538357831048`
- `pooled_norm`: `0.49363717076058006`
- `move_from_prev`: `0.48679962013295347`
- `move_to_next`: `0.526984126984127`
- `decel`: `0.507201646090535`
- `logit_entropy`: `0.44615435204366843`
- `logit_top_prob`: `0.5725231001940088`
- `logit_top2_margin`: `0.5807191476768274`
- `jacobian_gain_mean`: `0.5270629991126885`
- `jacobian_gain_max`: `0.5270629991126885`
