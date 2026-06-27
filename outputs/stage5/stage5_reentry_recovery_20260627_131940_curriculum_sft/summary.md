# Stage 5 Curriculum SFT - stage5_reentry_recovery_20260627_131940_curriculum_sft

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
- Re-entry tail damper: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/tail_damper.pt`
- Re-entry tail damper strength: `1.0`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 6.0,
  "expected_ce": 1.249848,
  "halting_kl": 0.547446,
  "loop_control_ce": 0.767749,
  "loss": 4.37559,
  "mean_expected_loops": 1.771071,
  "mean_halt_entropy": 0.319149,
  "target_loop_abs_error": 0.298297,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.99855,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.646212,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.821914,
  "group/curriculum_mode/deep_narrow/loss": 4.350829,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.98702,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.244028,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.277859,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 1.752444,
  "group/curriculum_mode/direct/halting_kl": 0.349915,
  "group/curriculum_mode/direct/loop_control_ce": 0.659419,
  "group/curriculum_mode/direct/loss": 4.425111,
  "group/curriculum_mode/direct/mean_expected_loops": 1.339174,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.469392,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.339174,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 1.752444,
  "group/target_loop_count/1/halting_kl": 0.349915,
  "group/target_loop_count/1/loop_control_ce": 0.659419,
  "group/target_loop_count/1/loss": 4.425111,
  "group/target_loop_count/1/mean_expected_loops": 1.339174,
  "group/target_loop_count/1/mean_halt_entropy": 0.469392,
  "group/target_loop_count/1/target_loop_abs_error": 0.339174,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 0.948221,
  "group/target_loop_count/2/halting_kl": 0.538327,
  "group/target_loop_count/2/loop_control_ce": 0.121536,
  "group/target_loop_count/2/loss": 1.488197,
  "group/target_loop_count/2/mean_expected_loops": 1.997607,
  "group/target_loop_count/2/mean_halt_entropy": 0.234952,
  "group/target_loop_count/2/target_loop_abs_error": 0.022231,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.149537,
  "group/target_loop_count/3/halting_kl": 0.969868,
  "group/target_loop_count/3/loop_control_ce": 2.92305,
  "group/target_loop_count/3/loss": 12.938725,
  "group/target_loop_count/3/mean_expected_loops": 1.955259,
  "group/target_loop_count/3/mean_halt_entropy": 0.271253,
  "group/target_loop_count/3/target_loop_abs_error": 1.044741,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.99855,
    "halting_kl": 0.646212,
    "loop_control_ce": 0.821914,
    "loss": 4.350829,
    "mean_expected_loops": 1.98702,
    "mean_halt_entropy": 0.244028,
    "target_loop_abs_error": 0.277859,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 1.752444,
    "halting_kl": 0.349915,
    "loop_control_ce": 0.659419,
    "loss": 4.425111,
    "mean_expected_loops": 1.339174,
    "mean_halt_entropy": 0.469392,
    "target_loop_abs_error": 0.339174,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 1.752444,
    "halting_kl": 0.349915,
    "loop_control_ce": 0.659419,
    "loss": 4.425111,
    "mean_expected_loops": 1.339174,
    "mean_halt_entropy": 0.469392,
    "target_loop_abs_error": 0.339174,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 0.948221,
    "halting_kl": 0.538327,
    "loop_control_ce": 0.121536,
    "loss": 1.488197,
    "mean_expected_loops": 1.997607,
    "mean_halt_entropy": 0.234952,
    "target_loop_abs_error": 0.022231,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.149537,
    "halting_kl": 0.969868,
    "loop_control_ce": 2.92305,
    "loss": 12.938725,
    "mean_expected_loops": 1.955259,
    "mean_halt_entropy": 0.271253,
    "target_loop_abs_error": 1.044741,
    "target_mean_loops": 3.0
  }
}
```

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.771071,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.339174,
    "deep_narrow_mean_expected_loops": 1.98702,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.339174,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.997607,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.955259,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6584329999999998,
      -0.04234799999999983
    ],
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
