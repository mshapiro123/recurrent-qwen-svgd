# Stage 5 Curriculum SFT - stage5_reentry_recovery_20260627_190155_curriculum_sft

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
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "skipped_keys": 168.0,
  "examples": 6.0,
  "expected_ce": 1.494869,
  "halting_kl": 0.543077,
  "loop_control_ce": 0.76822,
  "loss": 4.622057,
  "mean_expected_loops": 1.774222,
  "mean_halt_entropy": 0.330482,
  "target_loop_abs_error": 0.309714,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.212737,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.640958,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.822492,
  "group/curriculum_mode/deep_narrow/loss": 4.5668,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.977819,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.251401,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.281057,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 2.059133,
  "group/curriculum_mode/direct/halting_kl": 0.347315,
  "group/curriculum_mode/direct/loop_control_ce": 0.659677,
  "group/curriculum_mode/direct/loss": 4.732571,
  "group/curriculum_mode/direct/mean_expected_loops": 1.367029,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.488644,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.367029,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 2.059133,
  "group/target_loop_count/1/halting_kl": 0.347315,
  "group/target_loop_count/1/loop_control_ce": 0.659677,
  "group/target_loop_count/1/loss": 4.732571,
  "group/target_loop_count/1/mean_expected_loops": 1.367029,
  "group/target_loop_count/1/mean_halt_entropy": 0.488644,
  "group/target_loop_count/1/target_loop_abs_error": 0.367029,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 1.178091,
  "group/target_loop_count/2/halting_kl": 0.536553,
  "group/target_loop_count/2/loop_control_ce": 0.121648,
  "group/target_loop_count/2/loss": 1.718338,
  "group/target_loop_count/2/mean_expected_loops": 1.989368,
  "group/target_loop_count/2/mean_halt_entropy": 0.237143,
  "group/target_loop_count/2/target_loop_abs_error": 0.022467,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.316674,
  "group/target_loop_count/3/halting_kl": 0.954172,
  "group/target_loop_count/3/loop_control_ce": 2.925024,
  "group/target_loop_count/3/loss": 13.112187,
  "group/target_loop_count/3/mean_expected_loops": 1.943173,
  "group/target_loop_count/3/mean_halt_entropy": 0.294174,
  "group/target_loop_count/3/target_loop_abs_error": 1.056827,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.212737,
    "halting_kl": 0.640958,
    "loop_control_ce": 0.822492,
    "loss": 4.5668,
    "mean_expected_loops": 1.977819,
    "mean_halt_entropy": 0.251401,
    "target_loop_abs_error": 0.281057,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 2.059133,
    "halting_kl": 0.347315,
    "loop_control_ce": 0.659677,
    "loss": 4.732571,
    "mean_expected_loops": 1.367029,
    "mean_halt_entropy": 0.488644,
    "target_loop_abs_error": 0.367029,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 2.059133,
    "halting_kl": 0.347315,
    "loop_control_ce": 0.659677,
    "loss": 4.732571,
    "mean_expected_loops": 1.367029,
    "mean_halt_entropy": 0.488644,
    "target_loop_abs_error": 0.367029,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 1.178091,
    "halting_kl": 0.536553,
    "loop_control_ce": 0.121648,
    "loss": 1.718338,
    "mean_expected_loops": 1.989368,
    "mean_halt_entropy": 0.237143,
    "target_loop_abs_error": 0.022467,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.316674,
    "halting_kl": 0.954172,
    "loop_control_ce": 2.925024,
    "loss": 13.112187,
    "mean_expected_loops": 1.943173,
    "mean_halt_entropy": 0.294174,
    "target_loop_abs_error": 1.056827,
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
  "mean_expected_loops": 1.774222,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.367029,
    "deep_narrow_mean_expected_loops": 1.977819,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.367029,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.989368,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.943173,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.622339,
      -0.046194999999999986
    ],
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
