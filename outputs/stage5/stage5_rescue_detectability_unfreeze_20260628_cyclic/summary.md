# Rescue Detectability Gate - stage5_rescue_detectability_unfreeze_20260628_cyclic

- Source sweep: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260628_031842/summary.json`
- Benchmark: `arc_challenge`
- Score target: `cyclic_label_aggregated` / `permutation_mean`
- Loops: `[1, 2, 3, 4, 8]`
- Status: `passed`
- Category counts: `{'harmable': 111, 'rescuable': 42, 'stable_correct': 44, 'stable_wrong': 59}`

## Direction Agreement Gate

- Best shrinkage: `1.0`
- Observed agreement: `0.9761710563898854`
- Null mean agreement: `0.685355659697776`
- Null p95 agreement: `0.8869196497273119`
- Observed minus null p95: `0.08925140666257358`
- Clears null p95: `True`

- shrinkage `0.1`: observed `0.8988608219314718`, null_p95 `0.9391503738795757`, margin `-0.04028955194810391`, clears `False`
- shrinkage `1.0`: observed `0.9761710563898854`, null_p95 `0.8869196497273119`, margin `0.08925140666257358`, clears `True`
- shrinkage `10.0`: observed `0.9931134255545857`, null_p95 `0.9561318284860677`, margin `0.036981597068518`, clears `True`

## Supervised Probe Discovery Curves

### whitened_rescue_score_shrinkage_0.1
- max_net: correct `158`, delta `3`, rescue `11`, harm `8`, routed `39`

### whitened_rescue_score_shrinkage_1
- max_net: correct `159`, delta `4`, rescue `12`, harm `8`, routed `39`

### whitened_rescue_score_shrinkage_10
- max_net: correct `159`, delta `4`, rescue `9`, harm `5`, routed `26`
