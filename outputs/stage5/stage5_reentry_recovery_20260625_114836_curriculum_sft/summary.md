# Stage 5 Curriculum SFT - stage5_reentry_recovery_20260625_114836_curriculum_sft

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
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_needs_review`
- Validation issues: `['target_loop_gradient_not_observed']`

## Training
- Resume from: `outputs/stage5/stage5_reentry_repair_smoke_20260625_114554/phase1_reentry_repair/phase1_step_25.pt`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_114836_curriculum_sft/phase1/phase1_step_75.pt`
- Steps: `75`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 6.0,
  "expected_ce": 1.372015,
  "halting_kl": 0.532007,
  "loop_control_ce": 0.76632,
  "loss": 4.490495,
  "mean_expected_loops": 1.760285,
  "mean_halt_entropy": 0.33675,
  "target_loop_abs_error": 0.299996,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.095761,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.62751,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.820062,
  "group/curriculum_mode/deep_narrow/loss": 4.438759,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.974906,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.268404,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.284473,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 1.924522,
  "group/curriculum_mode/direct/halting_kl": 0.341002,
  "group/curriculum_mode/direct/loop_control_ce": 0.658836,
  "group/curriculum_mode/direct/loss": 4.593967,
  "group/curriculum_mode/direct/mean_expected_loops": 1.331041,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.473443,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.331041,
  "group/curriculum_mode/direct/target_mean_loops": 1.0,
  "group/target_loop_count/1/examples": 2.0,
  "group/target_loop_count/1/expected_ce": 1.924522,
  "group/target_loop_count/1/halting_kl": 0.341002,
  "group/target_loop_count/1/loop_control_ce": 0.658836,
  "group/target_loop_count/1/loss": 4.593967,
  "group/target_loop_count/1/mean_expected_loops": 1.331041,
  "group/target_loop_count/1/mean_halt_entropy": 0.473443,
  "group/target_loop_count/1/target_loop_abs_error": 0.331041,
  "group/target_loop_count/1/target_mean_loops": 1.0,
  "group/target_loop_count/2/examples": 3.0,
  "group/target_loop_count/2/expected_ce": 1.055458,
  "group/target_loop_count/2/halting_kl": 0.522427,
  "group/target_loop_count/2/loop_control_ce": 0.123669,
  "group/target_loop_count/2/loss": 1.602376,
  "group/target_loop_count/2/mean_expected_loops": 1.988347,
  "group/target_loop_count/2/mean_halt_entropy": 0.254298,
  "group/target_loop_count/2/target_loop_abs_error": 0.024158,
  "group/target_loop_count/2/target_mean_loops": 2.0,
  "group/target_loop_count/3/examples": 1.0,
  "group/target_loop_count/3/expected_ce": 1.216672,
  "group/target_loop_count/3/halting_kl": 0.942758,
  "group/target_loop_count/3/loop_control_ce": 2.90924,
  "group/target_loop_count/3/loss": 12.947906,
  "group/target_loop_count/3/mean_expected_loops": 1.934583,
  "group/target_loop_count/3/mean_halt_entropy": 0.310724,
  "group/target_loop_count/3/target_loop_abs_error": 1.065417,
  "group/target_loop_count/3/target_mean_loops": 3.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.095761,
    "halting_kl": 0.62751,
    "loop_control_ce": 0.820062,
    "loss": 4.438759,
    "mean_expected_loops": 1.974906,
    "mean_halt_entropy": 0.268404,
    "target_loop_abs_error": 0.284473,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 1.924522,
    "halting_kl": 0.341002,
    "loop_control_ce": 0.658836,
    "loss": 4.593967,
    "mean_expected_loops": 1.331041,
    "mean_halt_entropy": 0.473443,
    "target_loop_abs_error": 0.331041,
    "target_mean_loops": 1.0
  }
}
```

## Validation By Target Loop
```json
{
  "1": {
    "examples": 2.0,
    "expected_ce": 1.924522,
    "halting_kl": 0.341002,
    "loop_control_ce": 0.658836,
    "loss": 4.593967,
    "mean_expected_loops": 1.331041,
    "mean_halt_entropy": 0.473443,
    "target_loop_abs_error": 0.331041,
    "target_mean_loops": 1.0
  },
  "2": {
    "examples": 3.0,
    "expected_ce": 1.055458,
    "halting_kl": 0.522427,
    "loop_control_ce": 0.123669,
    "loss": 1.602376,
    "mean_expected_loops": 1.988347,
    "mean_halt_entropy": 0.254298,
    "target_loop_abs_error": 0.024158,
    "target_mean_loops": 2.0
  },
  "3": {
    "examples": 1.0,
    "expected_ce": 1.216672,
    "halting_kl": 0.942758,
    "loop_control_ce": 2.90924,
    "loss": 12.947906,
    "mean_expected_loops": 1.934583,
    "mean_halt_entropy": 0.310724,
    "target_loop_abs_error": 1.065417,
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

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
