# Stage 5 Curriculum SFT - stage5_halt_only_repair_l4_20260623_062410

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_halt_only_repair_l4_20260623_062410_mcq_ladder', 'summary_json': 'data/curriculum/stage5_halt_only_repair_l4_20260623_062410_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_needs_review`
- Validation issues: `['depth_gradient_not_observed']`

## Training
- Resume from: `outputs/stage5/stage5_halt_target_repair_strong_l4_20260623_061610/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_halt_only_repair_l4_20260623_062410/phase1/phase1_step_200.pt`
- Steps: `200`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.40069,
  "halting_kl": 0.545275,
  "loss": 0.466123,
  "mean_expected_loops": 1.717586,
  "mean_halt_entropy": 1.086706,
  "target_loop_abs_error": 1.317203,
  "target_mean_loops": 2.857143,
  "group/curriculum_mode/deep_narrow/examples": 6.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.420005,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.62604,
  "group/curriculum_mode/deep_narrow/loss": 0.49513,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.733557,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.098896,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.43311,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.166667,
  "group/curriculum_mode/direct/examples": 1.0,
  "group/curriculum_mode/direct/expected_ce": 0.2848,
  "group/curriculum_mode/direct/halting_kl": 0.060688,
  "group/curriculum_mode/direct/loss": 0.292083,
  "group/curriculum_mode/direct/mean_expected_loops": 1.621761,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.013564,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.621761,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 6.0,
    "expected_ce": 0.420005,
    "halting_kl": 0.62604,
    "loss": 0.49513,
    "mean_expected_loops": 1.733557,
    "mean_halt_entropy": 1.098896,
    "target_loop_abs_error": 1.43311,
    "target_mean_loops": 3.166667
  },
  "direct": {
    "examples": 1.0,
    "expected_ce": 0.2848,
    "halting_kl": 0.060688,
    "loss": 0.292083,
    "mean_expected_loops": 1.621761,
    "mean_halt_entropy": 1.013564,
    "target_loop_abs_error": 0.621761,
    "target_mean_loops": 1.0
  }
}
```

## Validation Checks
```json
{
  "status": "validation_needs_review",
  "issues": [
    "depth_gradient_not_observed"
  ],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.717586,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.621761,
    "deep_narrow_mean_expected_loops": 1.733557,
    "required_margin": 0.25,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
