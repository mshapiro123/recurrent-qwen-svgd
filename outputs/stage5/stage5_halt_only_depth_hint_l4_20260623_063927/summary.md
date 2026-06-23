# Stage 5 Curriculum SFT - stage5_halt_only_depth_hint_l4_20260623_063927

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_halt_only_depth_hint_l4_20260623_063927_mcq_ladder', 'summary_json': 'data/curriculum/stage5_halt_only_depth_hint_l4_20260623_063927_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Train mode counts: `{'deep_narrow': 36, 'direct': 30}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 3}`
- Depth hint style: `natural`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_needs_review`
- Validation issues: `['depth_gradient_not_observed']`

## Training
- Resume from: `outputs/stage5/stage5_halt_target_repair_strong_l4_20260623_061610/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_halt_only_depth_hint_l4_20260623_063927/phase1/phase1_step_200.pt`
- Steps: `200`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.453824,
  "halting_kl": 0.362735,
  "loss": 0.497352,
  "mean_expected_loops": 1.757546,
  "mean_halt_entropy": 1.106286,
  "target_loop_abs_error": 1.125096,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.351847,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.593763,
  "group/curriculum_mode/deep_narrow/loss": 0.423098,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.803394,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.13786,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.446606,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.589793,
  "group/curriculum_mode/direct/halting_kl": 0.054698,
  "group/curriculum_mode/direct/loss": 0.596357,
  "group/curriculum_mode/direct/mean_expected_loops": 1.696415,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.064186,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.696415,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.351847,
    "halting_kl": 0.593763,
    "loss": 0.423098,
    "mean_expected_loops": 1.803394,
    "mean_halt_entropy": 1.13786,
    "target_loop_abs_error": 1.446606,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.589793,
    "halting_kl": 0.054698,
    "loss": 0.596357,
    "mean_expected_loops": 1.696415,
    "mean_halt_entropy": 1.064186,
    "target_loop_abs_error": 0.696415,
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
  "mean_expected_loops": 1.757546,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.696415,
    "deep_narrow_mean_expected_loops": 1.803394,
    "required_margin": 0.25,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
