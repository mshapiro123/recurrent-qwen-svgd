# Gradient-Path Audit

status: `graph_connected`
issues: `[]`

## Selected Batch

- rows: `48`
- depth counts: `{'1': 12, '2': 12, '3': 12, '4': 12}`
- target validity: `{'checked_loop_targets': 120, 'invalid_loop_targets': 0, 'invalid_fraction': 0.0, 'examples': []}`
- precision: `{'match_train_precision': True, 'requested_dtype': 'bfloat16', 'autocast_dtype': 'bfloat16', 'manual_loss_scale': 1.0, 'note': 'No torch GradScaler is used here unless represented by manual_loss_scale; current chain configs use AdamW without scaler.'}`
- multiplier check: `{'trainer': 'training/train_unfrozen_recurrent.py', 'configured_bridge_prelude_grad_multiplier': 8.0, 'optimizer': 'adamw', 'implementation': 'slice_gradient_scaled_before_optimizer_step', 'gradient_slice_scaled': True, 'scaling_before_optimizer_step': True, 'inert_under_adamw_or_muon_risk': True, 'reason': 'AdamW normalizes first-step gradient scale through moments; Muon orthogonalization discards update magnitude. A true raised prelude rate requires an optimizer param group or refactored bridge parameterization.', 'current_bridge_param_group_limitation': 'bridge prelude/state are slices of bridge.proj.weight, so ordinary torch param groups cannot assign a distinct LR to prelude only'}`

## Per-Loop Gradient Distribution

| loop | active_rows | group | median | q10 | q90 | zero_fraction |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 48 | bridge_prelude | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | bridge_state | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | bridge_prelude_norm | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 1 | 48 | recurrent_block | 9.269e-07 | 7.807e-08 | 6.141e-04 | 0.00 |
| 1 | 48 | coda | 4.566e-07 | 5.646e-08 | 2.669e-04 | 0.00 |
| 2 | 36 | bridge_prelude | 1.302e-02 | 7.375e-03 | 2.403e-02 | 0.00 |
| 2 | 36 | bridge_state | 2.028e-02 | 1.312e-02 | 3.728e-02 | 0.00 |
| 2 | 36 | bridge_prelude_norm | 4.272e-04 | 1.827e-04 | 7.472e-04 | 0.00 |
| 2 | 36 | recurrent_block | 6.893e-03 | 3.805e-03 | 1.329e-02 | 0.00 |
| 2 | 36 | coda | 1.469e-02 | 8.293e-03 | 1.854e-02 | 0.00 |
| 3 | 24 | bridge_prelude | 1.010e-02 | 7.383e-03 | 1.543e-02 | 0.00 |
| 3 | 24 | bridge_state | 2.295e-02 | 1.717e-02 | 3.367e-02 | 0.00 |
| 3 | 24 | bridge_prelude_norm | 2.551e-04 | 1.550e-04 | 6.937e-04 | 0.00 |
| 3 | 24 | recurrent_block | 3.069e-03 | 2.351e-03 | 4.557e-03 | 0.00 |
| 3 | 24 | coda | 1.332e-02 | 1.220e-02 | 1.446e-02 | 0.00 |
| 4 | 12 | bridge_prelude | 8.316e-03 | 7.264e-03 | 1.001e-02 | 0.00 |
| 4 | 12 | bridge_state | 2.590e-02 | 2.230e-02 | 2.919e-02 | 0.00 |
| 4 | 12 | bridge_prelude_norm | 1.394e-04 | 1.092e-04 | 3.190e-04 | 0.00 |
| 4 | 12 | recurrent_block | 1.834e-03 | 1.521e-03 | 2.261e-03 | 0.00 |
| 4 | 12 | coda | 1.215e-02 | 1.182e-02 | 1.275e-02 | 0.00 |

## Bridge Gradient Coherence

| loop | group | rows | coherence | floor | mean_grad_norm | zero_fraction |
|---:|---|---:|---:|---:|---:|---:|
| 1 | bridge_prelude | 48 | 0.000 | 0.144 | 0.000e+00 | 1.00 |
| 1 | bridge_prelude_norm | 48 | 0.000 | 0.144 | 0.000e+00 | 1.00 |
| 1 | bridge_state | 48 | 0.000 | 0.144 | 0.000e+00 | 1.00 |
| 2 | bridge_prelude | 36 | 0.335 | 0.167 | 1.372e+01 | 0.00 |
| 2 | bridge_prelude_norm | 36 | 0.340 | 0.167 | 1.744e-02 | 0.00 |
| 2 | bridge_state | 36 | 0.326 | 0.167 | 2.160e+01 | 0.00 |
| 3 | bridge_prelude | 24 | 0.213 | 0.204 | 9.445e+00 | 0.00 |
| 3 | bridge_prelude_norm | 24 | 0.374 | 0.204 | 1.335e-02 | 0.00 |
| 3 | bridge_state | 24 | 0.224 | 0.204 | 2.195e+01 | 0.00 |
| 4 | bridge_prelude | 12 | 0.421 | 0.289 | 7.627e+00 | 0.00 |
| 4 | bridge_prelude_norm | 12 | 0.553 | 0.289 | 7.880e-03 | 0.00 |
| 4 | bridge_state | 12 | 0.433 | 0.289 | 2.328e+01 | 0.00 |

## Bridge Prelude Finite Difference

| loop | records | median_abs_delta | q10 | q90 | zero_fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.00 |
| 2 | 8 | 1.184e-01 | 5.652e-02 | 2.057e-01 | 0.00 |
| 3 | 8 | 6.679e-02 | 2.719e-02 | 1.574e-01 | 0.00 |
| 4 | 8 | 3.206e-02 | 1.848e-02 | 1.558e-01 | 0.00 |

## Cross-Loop Finite Difference

records: `[{'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.6147809028625488, 'perturbed_loss': 1.61478590965271, 'delta': 5.0067901611328125e-06, 'abs_delta': 5.0067901611328125e-06, 'delta_per_epsilon': 0.0005006790161132812, 'active_label_tokens': 1, 'row_id': 'train_d04_00000', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.3576003313064575, 'perturbed_loss': 1.3914334774017334, 'delta': 0.03383314609527588, 'abs_delta': 0.03383314609527588, 'delta_per_epsilon': 3.383314609527588, 'active_label_tokens': 1, 'row_id': 'train_d04_00001', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.17790687084198, 'perturbed_loss': 1.205697774887085, 'delta': 0.02779090404510498, 'abs_delta': 0.02779090404510498, 'delta_per_epsilon': 2.779090404510498, 'active_label_tokens': 1, 'row_id': 'train_d04_00002', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.365273118019104, 'perturbed_loss': 1.3306589126586914, 'delta': -0.0346142053604126, 'abs_delta': 0.0346142053604126, 'delta_per_epsilon': -3.4614205360412598, 'active_label_tokens': 1, 'row_id': 'train_d04_00003', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.4203746318817139, 'perturbed_loss': 1.456262469291687, 'delta': 0.035887837409973145, 'abs_delta': 0.035887837409973145, 'delta_per_epsilon': 3.5887837409973145, 'active_label_tokens': 1, 'row_id': 'train_d04_00004', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.3648191690444946, 'perturbed_loss': 1.3909934759140015, 'delta': 0.026174306869506836, 'abs_delta': 0.026174306869506836, 'delta_per_epsilon': 2.6174306869506836, 'active_label_tokens': 1, 'row_id': 'train_d04_00005', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.3651498556137085, 'perturbed_loss': 1.3060442209243774, 'delta': -0.059105634689331055, 'abs_delta': 0.059105634689331055, 'delta_per_epsilon': -5.9105634689331055, 'active_label_tokens': 1, 'row_id': 'train_d04_00006', 'depth': 4}, {'perturb_loop': 2, 'read_loop': 4, 'base_loss': 1.306131362915039, 'perturbed_loss': 1.306168556213379, 'delta': 3.719329833984375e-05, 'abs_delta': 3.719329833984375e-05, 'delta_per_epsilon': 0.003719329833984375, 'active_label_tokens': 1, 'row_id': 'train_d04_00007', 'depth': 4}]`
