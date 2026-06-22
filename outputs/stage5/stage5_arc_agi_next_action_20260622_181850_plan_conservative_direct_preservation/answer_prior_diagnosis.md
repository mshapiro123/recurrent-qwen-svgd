# ARC-Mix Answer-Prior Diagnosis - stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation

- Status: `direct_answer_prior_not_preserved`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/summary.json`
- Next step: Do not launch another A100 SFT run from this branch. The base-confident direct bucket is still below base; revise the objective toward direct-route/base-logit preservation or a hard max_loops=1 path.

| comparison | base | candidate | delta | wins | losses | margin delta | max pred shift | direct-bucket delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_vs_start` | 87/128 | 82/128 | -5 | 15 | 20 | -1.5996 | 22 | -14 |
| `base_vs_best` | 87/128 | 81/128 | -6 | 11 | 17 | -1.5245 | 20 | -11 |
| `start_vs_best` | 82/128 | 81/128 | -1 | 4 | 5 | 0.0751 | 5 | 0 |

## Label Priors

### `base_vs_start`

| label | base | candidate | answer | candidate-base |
|---|---:|---:|---:|---:|
| `A` | 36 | 58 | 30 | +22 |
| `B` | 33 | 15 | 27 | -18 |
| `C` | 42 | 22 | 38 | -20 |
| `D` | 17 | 33 | 33 | +16 |

### `base_vs_best`

| label | base | candidate | answer | candidate-base |
|---|---:|---:|---:|---:|
| `A` | 36 | 56 | 30 | +20 |
| `B` | 33 | 18 | 27 | -15 |
| `C` | 42 | 26 | 38 | -16 |
| `D` | 17 | 28 | 33 | +11 |

### `start_vs_best`

| label | base | candidate | answer | candidate-base |
|---|---:|---:|---:|---:|
| `A` | 58 | 56 | 30 | -2 |
| `B` | 15 | 18 | 27 | +3 |
| `C` | 22 | 26 | 38 | +4 |
| `D` | 33 | 28 | 33 | -5 |
