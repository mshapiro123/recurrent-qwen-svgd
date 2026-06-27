# Stage 5 Curriculum SFT - stage5_capacity_localization_20260627_205028_lora64_curriculum_sft

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
- Checkpoint: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64_curriculum_sft/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "skipped_keys": 168.0,
  "examples": 6.0,
  "expected_ce": 1.519531,
  "halting_kl": 0.54592,
  "loop_control_ce": 0.76826,
  "loss": 4.647165,
  "mean_expected_loops": 1.773289,
  "mean_halt_entropy": 0.327348,
  "target_loop_abs_error": 0.309867,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.241979,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.64533,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.822022,
  "group/curriculum_mode/deep_narrow/loss": 4.594598,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.976418,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.246591,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.281284,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 2.074635,
  "group/curriculum_mode/direct/halting_kl": 0.347101,
  "group/curriculum_mode/direct/loop_control_ce": 0.660738,
  "group/curriculum_mode/direct/loss": 4.752298,
  "group/curriculum_mode/direct/mean_expected_loops": 1.367033,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.488861,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.367033,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 2.074635,
  "group/target_loop_count/1/halting_kl": 0.347101,
  "group/target_loop_count/1/loop_control_ce": 0.660738,
  "group/target_loop_count/1/loss": 4.752298,
  "group/target_loop_count/1/mean_expected_loops": 1.367033,
  "group/target_loop_count/1/mean_halt_entropy": 0.488861,
  "group/target_loop_count/1/target_loop_abs_error": 0.367033,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 1.209871,
  "group/target_loop_count/2/halting_kl": 0.540806,
  "group/target_loop_count/2/loop_control_ce": 0.121556,
  "group/target_loop_count/2/loss": 1.750177,
  "group/target_loop_count/2/mean_expected_loops": 1.987881,
  "group/target_loop_count/2/mean_halt_entropy": 0.232079,
  "group/target_loop_count/2/target_loop_abs_error": 0.022388,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.338302,
  "group/target_loop_count/3/halting_kl": 0.958901,
  "group/target_loop_count/3/loop_control_ce": 2.923418,
  "group/target_loop_count/3/loss": 13.127863,
  "group/target_loop_count/3/mean_expected_loops": 1.94203,
  "group/target_loop_count/3/mean_halt_entropy": 0.290129,
  "group/target_loop_count/3/target_loop_abs_error": 1.05797,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.241979,
    "halting_kl": 0.64533,
    "loop_control_ce": 0.822022,
    "loss": 4.594598,
    "mean_expected_loops": 1.976418,
    "mean_halt_entropy": 0.246591,
    "target_loop_abs_error": 0.281284,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 2.074635,
    "halting_kl": 0.347101,
    "loop_control_ce": 0.660738,
    "loss": 4.752298,
    "mean_expected_loops": 1.367033,
    "mean_halt_entropy": 0.488861,
    "target_loop_abs_error": 0.367033,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 2.074635,
    "halting_kl": 0.347101,
    "loop_control_ce": 0.660738,
    "loss": 4.752298,
    "mean_expected_loops": 1.367033,
    "mean_halt_entropy": 0.488861,
    "target_loop_abs_error": 0.367033,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 1.209871,
    "halting_kl": 0.540806,
    "loop_control_ce": 0.121556,
    "loss": 1.750177,
    "mean_expected_loops": 1.987881,
    "mean_halt_entropy": 0.232079,
    "target_loop_abs_error": 0.022388,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.338302,
    "halting_kl": 0.958901,
    "loop_control_ce": 2.923418,
    "loss": 13.127863,
    "mean_expected_loops": 1.94203,
    "mean_halt_entropy": 0.290129,
    "target_loop_abs_error": 1.05797,
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
  "mean_expected_loops": 1.773289,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.367033,
    "deep_narrow_mean_expected_loops": 1.976418,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.367033,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.987881,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.94203,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6208480000000001,
      -0.045851000000000086
    ],
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
