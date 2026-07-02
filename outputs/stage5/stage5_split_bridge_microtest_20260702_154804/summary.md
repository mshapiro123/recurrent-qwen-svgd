# Split Bridge True-LR Micro-Test - stage5_split_bridge_microtest_20260702_154804

- Status: `stage_depth_le2_finished`
- N symbols: `8`
- Rows per depth: `4`
- Prelude LR multiplier: `10.0`

## split_chain_depth_le2

- Checkpoint: `outputs/stage5/stage5_split_bridge_microtest_20260702_154804/train/split_chain_depth_le2/unfrozen_recurrent_step_2000.pt`
- Drive backup: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_split_bridge_microtest_20260702_154804/split_chain_depth_le2/unfrozen_recurrent_step_2000.pt`
- Optimizer setup: `{'bridge_prelude_optimizer_group_ok': True, 'bridge_prelude_optimizer_group_lr': 0.0001, 'bridge_prelude_optimizer_group_weight_decay': 0.0, 'bridge_prelude_optimizer_group_num_tensors': 1}`
- Bridge prelude stats: `{'bridge_prelude_weight_rms': 0.0008476690272800624, 'bridge_prelude_weight_max_abs': 0.0032267894130200148, 'bridge_state_identity_max_abs_diff': 0.00035384789225645363}`
- Train total hits: `{'correct': 26, 'total': 64}`
- Test total hits: `{'correct': 18, 'total': 64}`
- Test frontier: `{'1': 0, '2': 0, '3': 0, '4': 0}`
