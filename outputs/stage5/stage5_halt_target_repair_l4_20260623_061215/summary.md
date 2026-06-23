# Stage 5 Curriculum SFT - stage5_halt_target_repair_l4_20260623_061215

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_halt_target_repair_l4_20260623_061215_mcq_ladder', 'summary_json': 'data/curriculum/stage5_halt_target_repair_l4_20260623_061215_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_needs_review`
- Validation issues: `['depth_gradient_not_observed']`

## Training
- Resume from: `outputs/stage5/stage5_mcq_ladder_sft_nodrive_20260623_052428/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_halt_target_repair_l4_20260623_061215/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.373688,
  "halting_kl": 0.380446,
  "loss": 0.419342,
  "mean_expected_loops": 3.121408,
  "mean_halt_entropy": 1.137735,
  "target_loop_abs_error": 1.016588,
  "target_mean_loops": 2.857143,
  "group/curriculum_mode/deep_narrow/examples": 6.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.37879,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.30537,
  "group/curriculum_mode/deep_narrow/loss": 0.415435,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 3.121673,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.137533,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.832716,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.166667,
  "group/curriculum_mode/direct/examples": 1.0,
  "group/curriculum_mode/direct/expected_ce": 0.343076,
  "group/curriculum_mode/direct/halting_kl": 0.830906,
  "group/curriculum_mode/direct/loss": 0.442785,
  "group/curriculum_mode/direct/mean_expected_loops": 3.119821,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.138942,
  "group/curriculum_mode/direct/target_loop_abs_error": 2.119821,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 6.0,
    "expected_ce": 0.37879,
    "halting_kl": 0.30537,
    "loss": 0.415435,
    "mean_expected_loops": 3.121673,
    "mean_halt_entropy": 1.137533,
    "target_loop_abs_error": 0.832716,
    "target_mean_loops": 3.166667
  },
  "direct": {
    "examples": 1.0,
    "expected_ce": 0.343076,
    "halting_kl": 0.830906,
    "loss": 0.442785,
    "mean_expected_loops": 3.119821,
    "mean_halt_entropy": 1.138942,
    "target_loop_abs_error": 2.119821,
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
  "mean_expected_loops": 3.121408,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 3.119821,
    "deep_narrow_mean_expected_loops": 3.121673,
    "required_margin": 0.25,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
