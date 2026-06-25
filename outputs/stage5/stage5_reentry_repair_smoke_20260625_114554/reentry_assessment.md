# Stage 5 Re-entry Assessment - stage5_reentry_repair_smoke_20260625_114554

- Source kind: `stage5_reentry_repair_smoke`
- Status: `bridge_repair_smoke_passed`
- Recommendation: `run_bounded_recovery_training_with_reentry_repair`
- Reason: Bridge and re-entry repair path are gradient-live and changed during the smoke run.

## Metrics
- `post_bridge_gate`: `1.0002319812774658`
- `pre_bridge_delta_rms`: `0.0`
- `post_bridge_delta_rms`: `0.08213316649198532`
- `post_bridge_proj_identity_max_abs_diff`: `0.0002565568720456213`
- `post_bridge_proj_bias_max_abs`: `0.00025284389266744256`
- `bridge_gate_active`: `True`
- `bridge_projection_moved`: `True`
- `bridge_output_moved`: `True`
- `post_weight_grad_rms`: `0.38929283618927`
- `post_bias_grad_rms`: `0.007555652409791946`
- `bridge_live`: `True`
- `bridge_moved`: `True`
- `reentry_rescale_mode`: `entry_rms`
- `expected_reentry_rescale_mode`: `entry_rms`
- `reentry_rescale_mode_ok`: `True`
- `use_reentry_adapter`: `True`
- `adapter_delta_rms`: `0.0018250071443617344`
- `adapter_scale_identity_max_abs_diff`: `0.0002409219741821289`
- `adapter_bias_max_abs`: `0.00025282433489337564`
- `adapter_scale_grad_rms`: `11.344264030456543`
- `adapter_bias_grad_rms`: `0.0075557054951786995`
- `adapter_live`: `True`
- `adapter_moved`: `True`
- `train_metrics_available`: `True`
- `train_last_step`: `20`
- `train_loss`: `1.4865`
- `train_expected_ce`: `1.3386`
- `train_mean_expected_loops`: `1.2025`
- `train_target_loop_abs_error`: `0.7975`
- `train_halting_target_nll`: `1.5986`
- `depth_supervision_metrics_present`: `True`
- `loop1_preservation_available`: `True`
- `source_loop1_task_groups`: `6.0`
- `trained_loop1_task_groups`: `6.0`
- `source_loop1_best_hits`: `1.0`
- `trained_loop1_best_hits`: `1.0`
- `loop1_best_hits_delta`: `0.0`
- `source_loop1_candidate_hits`: `1.0`
- `trained_loop1_candidate_hits`: `1.0`
- `loop1_candidate_hits_delta`: `0.0`
- `loop1_source_has_correct_signal`: `True`
- `loop1_regressed`: `False`

## Loop-1 Preservation Gate
- Source has correct signal: `True`
- Source best hits: `1.0`
- Trained best hits: `1.0`
- Best-hit delta: `0.0`
- Candidate-hit delta: `0.0`
