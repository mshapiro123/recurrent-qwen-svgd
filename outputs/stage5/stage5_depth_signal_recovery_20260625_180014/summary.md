# Stage 5 Re-entry Recovery Training - stage5_depth_signal_recovery_20260625_180014

- Cell version: `reentry_recovery_training_v4_post_reentry_health`
- Child curriculum summary: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_repair_smoke_20260625_153526/reentry_assessment.json`
- Status: `validation_sane`
- Passed: `True`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.765795,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.339791,
    "deep_narrow_mean_expected_loops": 1.978797,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.339791,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.991162,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.941703,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6513710000000001,
      -0.04945900000000014
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014/post_reentry_drift.json`
- Health status: `reentry_health_sane`
```json
{
  "status": "reentry_health_sane",
  "issues": [],
  "thresholds": {
    "min_bridge_gate_abs": 0.05,
    "max_loop8_output_over_entry_rms": 3.0
  },
  "metrics": {
    "bridge_gate": 1.0005296468734741,
    "bridge_delta_rms": 0.10798023641109467,
    "bridge_weight_grad_rms": 0.3892977237701416,
    "bridge_bias_grad_rms": 0.00757050234824419,
    "reentry_adapter_mode": "spectral",
    "reentry_adapter_delta_rms": 0.005204830318689346,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "reentry_adapter_spectral_u_grad_rms": 0.10277573019266129,
    "reentry_adapter_spectral_v_grad_rms": 0.04048735648393631,
    "reentry_adapter_spectral_theta_grad_abs": 0.11835695058107376,
    "reentry_adapter_gradient_live": true,
    "loop8_output_over_entry_rms": 1.0068777799606323,
    "loop8_output_over_input_rms": 1.0061099131902058,
    "mean_exit_over_entry_rms": 1.001762827237447,
    "subspace_overlap": 0.3456194996833801
  }
}
```

## Next Step
Run debiased_benchmark_suite against this repaired deterministic recurrent checkpoint before dense control, breadth diagnostics, particles, or SVGD.
