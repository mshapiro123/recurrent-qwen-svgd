# Gradient-Path Audit

status: `graph_connected`
issues: `[]`

## Selected Batch

- rows: `48`
- depth counts: `{'4': 48}`
- target validity: `{'checked_loop_targets': 192, 'invalid_loop_targets': 0, 'invalid_fraction': 0.0, 'examples': []}`
- precision: `{'match_train_precision': True, 'requested_dtype': 'bfloat16', 'autocast_dtype': 'bfloat16', 'manual_loss_scale': 1.0, 'note': 'No torch GradScaler is used here unless represented by manual_loss_scale; current chain configs use AdamW without scaler.'}`

## Per-Loop Gradient Distribution

| loop | active_rows | group | median | q10 | q90 | zero_fraction |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 48 | bridge_prelude | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | bridge_state | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | bridge_prelude_norm | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | recurrent_block | 4.046e-06 | 8.263e-08 | 1.547e-02 | 0.00 |
| 1 | 48 | coda | 2.139e-06 | 6.124e-08 | 7.953e-03 | 0.00 |
| 2 | 48 | bridge_prelude | 1.110e-02 | 7.544e-03 | 2.332e-02 | 0.00 |
| 2 | 48 | bridge_state | 1.725e-02 | 1.253e-02 | 3.752e-02 | 0.00 |
| 2 | 48 | bridge_prelude_norm | 3.098e-04 | 1.919e-04 | 7.221e-04 | 0.00 |
| 2 | 48 | recurrent_block | 5.839e-03 | 3.828e-03 | 1.253e-02 | 0.00 |
| 2 | 48 | coda | 1.449e-02 | 1.002e-02 | 1.880e-02 | 0.00 |
| 3 | 48 | bridge_prelude | 9.712e-03 | 7.192e-03 | 1.299e-02 | 0.00 |
| 3 | 48 | bridge_state | 2.343e-02 | 1.717e-02 | 3.090e-02 | 0.00 |
| 3 | 48 | bridge_prelude_norm | 2.614e-04 | 1.510e-04 | 4.609e-04 | 0.00 |
| 3 | 48 | recurrent_block | 3.092e-03 | 2.057e-03 | 4.207e-03 | 0.00 |
| 3 | 48 | coda | 1.321e-02 | 1.218e-02 | 1.451e-02 | 0.00 |
| 4 | 48 | bridge_prelude | 8.798e-03 | 7.129e-03 | 1.228e-02 | 0.00 |
| 4 | 48 | bridge_state | 2.802e-02 | 2.198e-02 | 3.246e-02 | 0.00 |
| 4 | 48 | bridge_prelude_norm | 2.386e-04 | 1.127e-04 | 3.982e-04 | 0.00 |
| 4 | 48 | recurrent_block | 1.968e-03 | 1.570e-03 | 3.152e-03 | 0.00 |
| 4 | 48 | coda | 1.212e-02 | 1.144e-02 | 1.279e-02 | 0.00 |

## Bridge Prelude Finite Difference

| loop | records | median_abs_delta | q10 | q90 | zero_fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 2 | 8 | 8.289e-02 | 1.937e-02 | 2.843e-01 | 0.00 |
| 3 | 8 | 1.115e-01 | 2.726e-02 | 2.394e-01 | 0.00 |
| 4 | 8 | 6.118e-02 | 2.056e-02 | 1.229e-01 | 0.00 |

## Cross-Loop Finite Difference

records: `[{'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.5485986471176147, 'perturbed_loss': 1.5801893472671509, 'delta': 0.03159070014953613, 'abs_delta': 0.03159070014953613, 'delta_per_epsilon': 3.1590700149536133, 'active_label_tokens': 1, 'row_id': 'train_d04_00000', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.3914066553115845, 'perturbed_loss': 1.3576674461364746, 'delta': -0.03373920917510986, 'abs_delta': 0.03373920917510986, 'delta_per_epsilon': -3.3739209175109863, 'active_label_tokens': 1, 'row_id': 'train_d04_00001', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.1779046058654785, 'perturbed_loss': 1.119826316833496, 'delta': -0.05807828903198242, 'abs_delta': 0.05807828903198242, 'delta_per_epsilon': -5.807828903198242, 'active_label_tokens': 1, 'row_id': 'train_d04_00002', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.365290641784668, 'perturbed_loss': 1.36527419090271, 'delta': -1.6450881958007812e-05, 'abs_delta': 1.6450881958007812e-05, 'delta_per_epsilon': -0.0016450881958007812, 'active_label_tokens': 1, 'row_id': 'train_d04_00003', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.5165557861328125, 'perturbed_loss': 1.5165610313415527, 'delta': 5.245208740234375e-06, 'abs_delta': 5.245208740234375e-06, 'delta_per_epsilon': 0.0005245208740234375, 'active_label_tokens': 1, 'row_id': 'train_d04_00004', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.4236178398132324, 'perturbed_loss': 1.423628330230713, 'delta': 1.049041748046875e-05, 'abs_delta': 1.049041748046875e-05, 'delta_per_epsilon': 0.001049041748046875, 'active_label_tokens': 1, 'row_id': 'train_d04_00005', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.3651506900787354, 'perturbed_loss': 1.3652279376983643, 'delta': 7.724761962890625e-05, 'abs_delta': 7.724761962890625e-05, 'delta_per_epsilon': 0.007724761962890625, 'active_label_tokens': 1, 'row_id': 'train_d04_00006', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.30617094039917, 'perturbed_loss': 1.3061742782592773, 'delta': 3.337860107421875e-06, 'abs_delta': 3.337860107421875e-06, 'delta_per_epsilon': 0.0003337860107421875, 'active_label_tokens': 1, 'row_id': 'train_d04_00007', 'depth': 4}]`
