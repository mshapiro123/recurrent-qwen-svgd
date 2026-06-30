# Rescue Detectability Gate - stage5_prelude_rescue_detectability_cyclic_seed29_confirm

- Source sweep: `outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json`
- Benchmark: `arc_challenge`
- Score target: `cyclic_label_aggregated` / `permutation_mean`
- Loops: `[1, 2, 4, 8]`
- Status: `passed`
- Category counts: `{'harmable': 11, 'rescuable': 7, 'stable_correct': 10, 'stable_wrong': 15}`

## Direction Agreement Gate

- Best shrinkage: `10.0`
- Observed agreement: `0.9788902806806411`
- Null mean agreement: `0.7197936477979673`
- Null p95 agreement: `0.94163849867991`
- Observed minus null p95: `0.03725178200073109`
- Clears null p95: `True`

- shrinkage `10.0`: observed `0.9788902806806411`, null_p95 `0.94163849867991`, margin `0.03725178200073109`, clears `True`

## Supervised Probe Discovery Curves

### whitened_rescue_score_shrinkage_10
- zero_harm: correct `22`, delta `1`, rescue `1`, harm `0`, routed `3`
- harm_budget_1: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
- harm_budget_2: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
- max_net: correct `26`, delta `5`, rescue `6`, harm `1`, routed `15`
