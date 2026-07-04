# Chain-Continuation Attribution - stage5_chain_continuation_attribution_20260704_163056

- Status: `finished_with_extrapolation`
- Continuation loss mode: `per_loop_labels`
- Source checkpoint: `outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/restored/scaled_corrected_final.pt`
- Final active diagonal through train horizon: `{'1': 1.0, '2': 0.96875, '3': 0.96875, '4': 0.953125}`
- Extrapolation active diagonal: `{'1': 1.0, '2': 0.9609375, '3': 0.96875, '4': 0.96875, '5': 0.890625, '6': 0.6484375, '7': 0.328125, '8': 0.09375}`
- Extrapolation read: `{'5': {'observed': 0.890625, 'classification': 'inside_or_above_conservative_band', 'conservative_interval': [0.831, 0.976], 'threshold': 0.71}, '6': {'observed': 0.6484375, 'classification': 'below_bar', 'conservative_interval': [0.807, 0.963], 'threshold': 0.71}, '7': {'observed': 0.328125, 'classification': 'below_bar', 'conservative_interval': None, 'threshold': 0.71}, '8': {'observed': 0.09375, 'classification': 'below_bar', 'conservative_interval': None, 'threshold': 0.71}}`
