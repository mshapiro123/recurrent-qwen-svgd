# Support-6 Replication Receipts - stage5_support6_replication_receipts_20260708_003055

- Status: `replication_needs_dosed_seed_resolution`
- Canonical frontier: `bar_crossing_frontier`, bar `0.71`
- Target band: `9.0 +/- 1.0`

## Scores

### original
- run_id: `stage5_depth_support_route_20260705_124320`
- canonical_frontier: `9.005454545454546`
- canonical_frontier_pass: `True`
- deepest_passing_selection_frontier: `10`
- selected_correct: `{'7': 116, '8': 106, '9': 91, '10': 69}`

### seed_20260716
- run_id: `stage5_support6_seed_replication_20260707_122930_seed20260716`
- canonical_frontier: `7.0471111111111115`
- canonical_frontier_pass: `False`
- deepest_passing_selection_frontier: `9`
- selected_correct: `{'7': 93, '8': 48, '9': 25, '10': 13}`

### seed_20260726
- run_id: `stage5_support6_seed_replication_20260707_122930_seed20260726`
- canonical_frontier: `6.9511111111111115`
- canonical_frontier_pass: `False`
- deepest_passing_selection_frontier: `9`
- selected_correct: `{'7': 90, '8': 47, '9': 16, '10': 13}`

## Config Diffs Against Original
- `seed_20260716`: `{'data_config': {'original': {'n_symbols': 16, 'max_depth': 6, 'rows_per_depth': 256, 'seed': 20260705, 'num_choices': 4, 'max_target_loops': 6, 'value_prefix': 'letter:'}, 'candidate': {'n_symbols': 16, 'max_depth': 6, 'rows_per_depth': 256, 'seed': 20260716, 'num_choices': 4, 'max_target_loops': 6, 'value_prefix': 'letter:'}}}`
- `seed_20260726`: `{'data_config': {'original': {'n_symbols': 16, 'max_depth': 6, 'rows_per_depth': 256, 'seed': 20260705, 'num_choices': 4, 'max_target_loops': 6, 'value_prefix': 'letter:'}, 'candidate': {'n_symbols': 16, 'max_depth': 6, 'rows_per_depth': 256, 'seed': 20260726, 'num_choices': 4, 'max_target_loops': 6, 'value_prefix': 'letter:'}}}`

## Decision
At least one replicate seed fails the canonical bar-crossing frontier band. Run the pre-registered dosed-seed resolution before using this cell as robustness evidence.
