# Natural-Surface Replication Dose - stage5_natural_surface_replication_dose_seed931337_20260710

- Status: `evaluated_step_1000`
- Source eval rows: `outputs/stage5/stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/summary.json`
- Train seed: `931337`
- Save steps: `[1000, 1500, 2000, 2500, 3000, 4000, 6000]`
- Init checkpoint: `{'source_summary': 'outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json', 'source_run_id': 'stage5_n24_support12_rung_20260707_140139', 'source_kind': 'stage5_n24_support12_rung', 'preferred_step': 6000, 'restored_checkpoint': 'outputs/stage5/stage5_natural_surface_replication_dose_seed931337_20260710/restored/n24_support12_step6000.pt', 'drive_checkpoint_root': '/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints', 'selected_checkpoint_reference': '/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt', 'selected_checkpoint_source': '/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt', 'selected_checkpoint_sha256': '898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc'}`

## Checkpoint Evals

- Step `1000`: `{'relay_train_depth_min': 0.5078125, 'relay_extrap_depth_min': 0.3125, 'pointer_train_depth_min': 0.4375, 'pointer_extrap_depth_min': 0.1953125, 'synthetic_rehearsal_min': 0.98046875, 'synthetic_rehearsal_delta_by_depth': {'1': 0.0, '2': -0.01171875, '3': 0.0, '4': 0.0, '5': 0.00390625, '6': 0.0, '7': -0.01953125, '8': 0.01171875}, 'synthetic_rehearsal_min_delta': -0.01953125, 'status': 'verbal_rung_zero_finished', 'synthetic_full_width_min_1_12': 0.921875, 'synthetic_full_width_delta_vs_frozen': -0.046875, 'synthetic_full_width_nonregression_pass': False}`

## Curve Shape

`{'values': [{'step': 1000, 'pooled_tail_min': 0.1953125}], 'best': {'step': 1000, 'pooled_tail_min': 0.1953125}, 'final': {'step': 1000, 'pooled_tail_min': 0.1953125}, 'tail_peaks_before_final': False}`
