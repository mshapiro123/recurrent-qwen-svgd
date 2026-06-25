# Stage 5 Curriculum SFT - stage5_depth_signal_recovery_20260625_180014_curriculum_sft

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
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_reentry_repair_smoke_20260625_153526/phase1_reentry_repair/phase1_step_25.pt`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 6.0,
  "expected_ce": 1.35729,
  "halting_kl": 0.538296,
  "loop_control_ce": 0.766897,
  "loss": 5.245605,
  "mean_expected_loops": 1.765795,
  "mean_halt_entropy": 0.330711,
  "target_loop_abs_error": 0.300942,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.074409,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.635271,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.820056,
  "group/curriculum_mode/deep_narrow/loss": 5.238217,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.978797,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.258401,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.281518,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 1.923053,
  "group/curriculum_mode/direct/halting_kl": 0.344345,
  "group/curriculum_mode/direct/loop_control_ce": 0.660579,
  "group/curriculum_mode/direct/loss": 5.260381,
  "group/curriculum_mode/direct/mean_expected_loops": 1.339791,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.475331,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.339791,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 1.923053,
  "group/target_loop_count/1/halting_kl": 0.344345,
  "group/target_loop_count/1/loop_control_ce": 0.660579,
  "group/target_loop_count/1/loss": 5.260381,
  "group/target_loop_count/1/mean_expected_loops": 1.339791,
  "group/target_loop_count/1/mean_halt_entropy": 0.475331,
  "group/target_loop_count/1/target_loop_abs_error": 0.339791,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 1.032521,
  "group/target_loop_count/2/halting_kl": 0.529429,
  "group/target_loop_count/2/loop_control_ce": 0.122972,
  "group/target_loop_count/2/loss": 1.700325,
  "group/target_loop_count/2/mean_expected_loops": 1.991162,
  "group/target_loop_count/2/mean_halt_entropy": 0.245725,
  "group/target_loop_count/2/target_loop_abs_error": 0.022592,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.200072,
  "group/target_loop_count/3/halting_kl": 0.952797,
  "group/target_loop_count/3/loop_control_ce": 2.911309,
  "group/target_loop_count/3/loss": 15.851895,
  "group/target_loop_count/3/mean_expected_loops": 1.941703,
  "group/target_loop_count/3/mean_halt_entropy": 0.296429,
  "group/target_loop_count/3/target_loop_abs_error": 1.058297,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.074409,
    "halting_kl": 0.635271,
    "loop_control_ce": 0.820056,
    "loss": 5.238217,
    "mean_expected_loops": 1.978797,
    "mean_halt_entropy": 0.258401,
    "target_loop_abs_error": 0.281518,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 1.923053,
    "halting_kl": 0.344345,
    "loop_control_ce": 0.660579,
    "loss": 5.260381,
    "mean_expected_loops": 1.339791,
    "mean_halt_entropy": 0.475331,
    "target_loop_abs_error": 0.339791,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 1.923053,
    "halting_kl": 0.344345,
    "loop_control_ce": 0.660579,
    "loss": 5.260381,
    "mean_expected_loops": 1.339791,
    "mean_halt_entropy": 0.475331,
    "target_loop_abs_error": 0.339791,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 1.032521,
    "halting_kl": 0.529429,
    "loop_control_ce": 0.122972,
    "loss": 1.700325,
    "mean_expected_loops": 1.991162,
    "mean_halt_entropy": 0.245725,
    "target_loop_abs_error": 0.022592,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.200072,
    "halting_kl": 0.952797,
    "loop_control_ce": 2.911309,
    "loss": 15.851895,
    "mean_expected_loops": 1.941703,
    "mean_halt_entropy": 0.296429,
    "target_loop_abs_error": 1.058297,
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

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
