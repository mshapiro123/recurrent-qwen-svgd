# Stage 5 Curriculum SFT - stage5_loopaware_halt_only_depth_hint_l4_20260623_064747

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_loopaware_halt_only_depth_hint_l4_20260623_064747_mcq_ladder', 'summary_json': 'data/curriculum/stage5_loopaware_halt_only_depth_hint_l4_20260623_064747_mcq_ladder/summary.json'}`
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
- Checkpoint: `outputs/stage5/stage5_loopaware_halt_only_depth_hint_l4_20260623_064747/phase1/phase1_step_200.pt`
- Steps: `200`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.452989,
  "halting_kl": 0.3627,
  "loss": 0.496513,
  "mean_expected_loops": 1.762566,
  "mean_halt_entropy": 1.10513,
  "target_loop_abs_error": 1.124184,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.352102,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.590789,
  "group/curriculum_mode/deep_narrow/loss": 0.422997,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.808585,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.136602,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.441415,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.587505,
  "group/curriculum_mode/direct/halting_kl": 0.058581,
  "group/curriculum_mode/direct/loss": 0.594534,
  "group/curriculum_mode/direct/mean_expected_loops": 1.701208,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.063167,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.701208,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.352102,
    "halting_kl": 0.590789,
    "loss": 0.422997,
    "mean_expected_loops": 1.808585,
    "mean_halt_entropy": 1.136602,
    "target_loop_abs_error": 1.441415,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.587505,
    "halting_kl": 0.058581,
    "loss": 0.594534,
    "mean_expected_loops": 1.701208,
    "mean_halt_entropy": 1.063167,
    "target_loop_abs_error": 0.701208,
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
  "mean_expected_loops": 1.762566,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.701208,
    "deep_narrow_mean_expected_loops": 1.808585,
    "required_margin": 0.25,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
