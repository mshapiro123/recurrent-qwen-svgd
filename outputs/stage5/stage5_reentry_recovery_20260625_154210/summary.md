# Stage 5 Re-entry Recovery Training - stage5_reentry_recovery_20260625_154210

- Cell version: `reentry_recovery_training_v4_post_reentry_health`
- Child curriculum summary: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_repair_smoke_20260625_153526/reentry_assessment.json`
- Status: `validation_needs_review`
- Passed: `False`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/phase1/phase1_step_75.pt`

## Validation Checks
```json
{
  "status": "validation_needs_review",
  "issues": [
    "target_loop_gradient_not_observed"
  ],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.766761,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.338771,
    "deep_narrow_mean_expected_loops": 1.980757,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": true,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.338771,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.994509,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.939499,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6557380000000002,
      -0.05501
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_reentry_recovery_20260625_154210/post_reentry_drift.json`
- Health status: `reentry_health_needs_review`
```json
{
  "status": "reentry_health_needs_review",
  "issues": [
    "reentry_adapter_gradient_not_live_after_recovery"
  ],
  "thresholds": {
    "min_bridge_gate_abs": 0.05,
    "max_loop8_output_over_entry_rms": 3.0
  },
  "metrics": {
    "bridge_gate": 1.0004091262817383,
    "bridge_delta_rms": 0.09639152139425278,
    "bridge_weight_grad_rms": 0.38928231596946716,
    "bridge_bias_grad_rms": 0.007569462992250919,
    "reentry_adapter_delta_rms": 0.005607573315501213,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "loop8_output_over_entry_rms": 1.0062079032262166,
    "loop8_output_over_input_rms": 1.0060652891794841,
    "mean_exit_over_entry_rms": 1.001758098602295,
    "subspace_overlap": 0.34590932726860046
  }
}
```

## Next Step
Run debiased_benchmark_suite against this repaired deterministic recurrent checkpoint before dense control, breadth diagnostics, particles, or SVGD.
