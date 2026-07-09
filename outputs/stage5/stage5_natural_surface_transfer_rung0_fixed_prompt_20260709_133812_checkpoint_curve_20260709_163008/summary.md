# Natural-Surface Checkpoint Curve - stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812_checkpoint_curve_20260709_163008

- Status: `evaluated_step_2000`
- Source run: `stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812`
- Source summary: `outputs/stage5/stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/summary.json`
- Steps: `[2000, 4000, 6000]`

## Frozen Baseline

- Relay: `{'1': 0.921875, '2': 0.2109375, '3': 0.09375, '4': 0.09375, '5': 0.1015625, '6': 0.15625, '7': 0.1015625, '8': 0.09375, '9': 0.078125, '10': 0.09375, '11': 0.0859375, '12': 0.0546875}`
- Pointer: `{'1': 0.9609375, '2': 0.5390625, '3': 0.15625, '4': 0.109375, '5': 0.15625, '6': 0.0859375, '7': 0.078125, '8': 0.0546875, '9': 0.0703125, '10': 0.0625, '11': 0.078125, '12': 0.078125}`
- Synthetic rehearsal: `{'1': 1.0, '2': 1.0, '3': 0.99609375, '4': 0.98828125, '5': 0.99609375, '6': 0.99609375, '7': 1.0, '8': 0.984375}`

## Checkpoint Curve

### Step 2000

- Relay diagonal: `{'1': 0.9921875, '2': 0.9609375, '3': 0.96875, '4': 0.9296875, '5': 0.890625, '6': 0.890625, '7': 0.890625, '8': 0.8046875, '9': 0.8203125, '10': 0.765625, '11': 0.6640625, '12': 0.546875}`
- Pointer diagonal: `{'1': 0.96875, '2': 0.9140625, '3': 0.96875, '4': 0.90625, '5': 0.8515625, '6': 0.8515625, '7': 0.7890625, '8': 0.78125, '9': 0.7578125, '10': 0.6484375, '11': 0.5625, '12': 0.546875}`
- Synthetic rehearsal diagonal: `{'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0, '8': 0.98828125}`
- Decision read: `{'relay_train_depth_min': 0.8046875, 'relay_extrap_depth_min': 0.546875, 'pointer_train_depth_min': 0.78125, 'pointer_extrap_depth_min': 0.546875, 'synthetic_rehearsal_min': 0.98828125, 'synthetic_rehearsal_delta_by_depth': {'1': 0.0, '2': 0.0, '3': 0.00390625, '4': 0.01171875, '5': 0.00390625, '6': 0.00390625, '7': 0.0, '8': 0.00390625}, 'synthetic_rehearsal_min_delta': 0.0, 'status': 'verbal_rung_zero_finished', 'step': 2000}`

## Best By Metric

`{'relay_train_depth_min': {'step': 2000, 'value': 0.8046875}, 'relay_extrap_depth_min': {'step': 2000, 'value': 0.546875}, 'pointer_train_depth_min': {'step': 2000, 'value': 0.78125}, 'pointer_extrap_depth_min': {'step': 2000, 'value': 0.546875}, 'synthetic_rehearsal_min': {'step': 2000, 'value': 0.98828125}, 'synthetic_rehearsal_min_delta': {'step': 2000, 'value': 0.0}}`
