# Stage 5 Curriculum SFT - stage5_reentry_recovery_20260625_154210_curriculum_sft

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_194537', 'summary_json': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_194537/summary.json'}`
- Positive rows: `63`
- Train / validation rows: `57` / `6`
- Train mode counts: `{'deep_narrow': 33, 'direct': 24}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 2}`
- Train target-loop counts: `{'1': 24, '2': 25, '3': 8}`
- Validation target-loop counts: `{'1': 2, '2': 3, '3': 1}`
- Depth hint style: `natural`
- Target loop control: `False`
- Learned loop control: `True`
- Re-entry rescale: `entry_rms`
- Re-entry adapter: `True`
- Re-entry adapter mode: `spectral`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_needs_review`
- Validation issues: `['target_loop_gradient_not_observed']`

## Training
- Resume from: `outputs/stage5/stage5_reentry_repair_smoke_20260625_153526/phase1_reentry_repair/phase1_step_25.pt`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/phase1/phase1_step_75.pt`
- Steps: `75`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 6.0,
  "expected_ce": 1.378972,
  "halting_kl": 0.526721,
  "loop_control_ce": 0.76638,
  "loss": 4.497166,
  "mean_expected_loops": 1.766761,
  "mean_halt_entropy": 0.344169,
  "target_loop_abs_error": 0.301641,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.095258,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.618149,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.819557,
  "group/curriculum_mode/deep_narrow/loss": 4.435299,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.980757,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.278653,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.283076,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 1.946402,
  "group/curriculum_mode/direct/halting_kl": 0.343865,
  "group/curriculum_mode/direct/loop_control_ce": 0.660028,
  "group/curriculum_mode/direct/loss": 4.6209,
  "group/curriculum_mode/direct/mean_expected_loops": 1.338771,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.475201,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.338771,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 1.946402,
  "group/target_loop_count/1/halting_kl": 0.343865,
  "group/target_loop_count/1/loop_control_ce": 0.660028,
  "group/target_loop_count/1/loss": 4.6209,
  "group/target_loop_count/1/mean_expected_loops": 1.338771,
  "group/target_loop_count/1/mean_halt_entropy": 0.475201,
  "group/target_loop_count/1/target_loop_abs_error": 0.338771,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 1.054346,
  "group/target_loop_count/2/halting_kl": 0.511711,
  "group/target_loop_count/2/loop_control_ce": 0.123537,
  "group/target_loop_count/2/loss": 1.599665,
  "group/target_loop_count/2/mean_expected_loops": 1.994509,
  "group/target_loop_count/2/mean_halt_entropy": 0.267178,
  "group/target_loop_count/2/target_loop_abs_error": 0.023935,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.217992,
  "group/target_loop_count/3/halting_kl": 0.937465,
  "group/target_loop_count/3/loop_control_ce": 2.907616,
  "group/target_loop_count/3/loss": 12.942202,
  "group/target_loop_count/3/mean_expected_loops": 1.939499,
  "group/target_loop_count/3/mean_halt_entropy": 0.313078,
  "group/target_loop_count/3/target_loop_abs_error": 1.060501,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.095258,
    "halting_kl": 0.618149,
    "loop_control_ce": 0.819557,
    "loss": 4.435299,
    "mean_expected_loops": 1.980757,
    "mean_halt_entropy": 0.278653,
    "target_loop_abs_error": 0.283076,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 1.946402,
    "halting_kl": 0.343865,
    "loop_control_ce": 0.660028,
    "loss": 4.6209,
    "mean_expected_loops": 1.338771,
    "mean_halt_entropy": 0.475201,
    "target_loop_abs_error": 0.338771,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 1.946402,
    "halting_kl": 0.343865,
    "loop_control_ce": 0.660028,
    "loss": 4.6209,
    "mean_expected_loops": 1.338771,
    "mean_halt_entropy": 0.475201,
    "target_loop_abs_error": 0.338771,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 1.054346,
    "halting_kl": 0.511711,
    "loop_control_ce": 0.123537,
    "loss": 1.599665,
    "mean_expected_loops": 1.994509,
    "mean_halt_entropy": 0.267178,
    "target_loop_abs_error": 0.023935,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.217992,
    "halting_kl": 0.937465,
    "loop_control_ce": 2.907616,
    "loss": 12.942202,
    "mean_expected_loops": 1.939499,
    "mean_halt_entropy": 0.313078,
    "target_loop_abs_error": 1.060501,
    "target_mean_loops": 3.0
  }
}
```

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

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
