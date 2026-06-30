# Rescue Detectability Gate - stage5_prelude_rescue_detectability_cyclic

- Source sweep: `outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json`
- Benchmark: `arc_challenge`
- Score target: `cyclic_label_aggregated` / `permutation_mean`
- Loops: `[1, 2, 4, 8]`
- Status: `passed`
- Category counts: `{'harmable': 11, 'rescuable': 7, 'stable_correct': 10, 'stable_wrong': 15}`

## Direction Agreement Gate

- Best shrinkage: `10.0`
- Observed agreement: `0.9744511428386475`
- Null mean agreement: `0.7340768971832943`
- Null p95 agreement: `0.9697230351834751`
- Observed minus null p95: `0.004728107655172398`
- Clears null p95: `True`

- shrinkage `0.1`: observed `0.6877861246225411`, null_p95 `0.9150601207878738`, margin `-0.22727399616533273`, clears `False`
- shrinkage `1.0`: observed `0.8372238332313918`, null_p95 `0.8990713826377755`, margin `-0.061847549406383706`, clears `False`
- shrinkage `10.0`: observed `0.9744511428386475`, null_p95 `0.9697230351834751`, margin `0.004728107655172398`, clears `True`

## Supervised Probe Discovery Curves

### whitened_rescue_score_shrinkage_0.1
- zero_harm: correct `22`, delta `1`, rescue `1`, harm `0`, routed `9`
- harm_budget_1: correct `25`, delta `4`, rescue `5`, harm `1`, routed `11`
- harm_budget_2: correct `26`, delta `5`, rescue `7`, harm `2`, routed `18`
- max_net: correct `26`, delta `5`, rescue `7`, harm `2`, routed `18`

### whitened_rescue_score_shrinkage_1
- zero_harm: correct `22`, delta `1`, rescue `1`, harm `0`, routed `3`
- harm_budget_1: correct `23`, delta `2`, rescue `3`, harm `1`, routed `9`
- harm_budget_2: correct `25`, delta `4`, rescue `6`, harm `2`, routed `18`
- max_net: correct `25`, delta `4`, rescue `6`, harm `2`, routed `18`

### whitened_rescue_score_shrinkage_10
- zero_harm: correct `22`, delta `1`, rescue `1`, harm `0`, routed `3`
- harm_budget_1: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
- harm_budget_2: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
- max_net: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
