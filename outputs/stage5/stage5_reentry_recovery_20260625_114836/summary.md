# Stage 5 Re-entry Recovery Training - stage5_reentry_recovery_20260625_114836

- Cell version: `reentry_recovery_training_v4_post_reentry_health`
- Child curriculum summary: `outputs/stage5/stage5_reentry_recovery_20260625_114836_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_repair_smoke_20260625_114554/reentry_assessment.json`
- Status: `validation_needs_review`
- Passed: `False`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_114836_curriculum_sft/phase1/phase1_step_75.pt`

## Validation Checks
```json
{
  "status": "validation_needs_review",
  "issues": [
    "target_loop_gradient_not_observed"
  ],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.760285,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.331041,
    "deep_narrow_mean_expected_loops": 1.974906,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": true,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.331041,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.988347,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.934583,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6573060000000002,
      -0.053764000000000145
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_reentry_recovery_20260625_114836/post_reentry_drift.json`
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
    "bridge_gate": 1.0004149675369263,
    "bridge_delta_rms": 0.1004943698644638,
    "bridge_weight_grad_rms": 0.3892625570297241,
    "bridge_bias_grad_rms": 0.007569043897092342,
    "reentry_adapter_delta_rms": 0.00032979800016619265,
    "reentry_adapter_scale_grad_rms": 11.3426513671875,
    "reentry_adapter_bias_grad_rms": 0.0075681316666305065,
    "loop8_output_over_entry_rms": 1.0066660642623901,
    "loop8_output_over_input_rms": 1.0060979922612507,
    "mean_exit_over_entry_rms": 1.0017553965250652,
    "subspace_overlap": 0.34565362334251404
  }
}
```

## Next Step
Run debiased_benchmark_suite against this repaired deterministic recurrent checkpoint before dense control, breadth diagnostics, particles, or SVGD.
