# Corrected Scaled Synthetic-Depth Chain Run - stage5_chain_scaled_corrected_20260702_182827

- Status: `stage_depth_le2_finished`
- N symbols: `16`
- Rows/depth train: `256`
- Heldout rows/depth eval: `64`
- Primary threshold: `0.71`

## Existing Microtest Active-Label Readout

- Train active diagonal: `None`
- Test active diagonal: `None`

## chain_scaled_corrected_depth_le2

- Checkpoint: `outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/train/chain_scaled_corrected_depth_le2/unfrozen_recurrent_step_2000.pt`
- Drive backup: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_chain_scaled_corrected_20260702_182827/chain_scaled_corrected_depth_le2/unfrozen_recurrent_step_2000.pt`
- Train active diagonal: `{'1': 1.0, '2': 0.96875, '3': 0.046875, '4': 0.03125}`
- Heldout active diagonal: `{'1': 1.0, '2': 0.875, '3': 0.125, '4': 0.046875}`
- Heldout active min: `0.046875`
- Final-answer heldout hits: `{'correct': 371, 'total': 1024}`
- Bridge prelude stats: `{'bridge_prelude_weight_rms': 0.002700953045859933, 'bridge_prelude_weight_max_abs': 0.014273237437009811, 'bridge_state_identity_max_abs_diff': 0.0025808343198150396}`
