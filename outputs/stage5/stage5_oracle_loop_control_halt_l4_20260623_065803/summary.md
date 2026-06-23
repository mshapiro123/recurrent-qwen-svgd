# Stage 5 Curriculum SFT - stage5_oracle_loop_control_halt_l4_20260623_065803

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_oracle_loop_control_halt_l4_20260623_065803_mcq_ladder', 'summary_json': 'data/curriculum/stage5_oracle_loop_control_halt_l4_20260623_065803_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Train mode counts: `{'deep_narrow': 36, 'direct': 30}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 3}`
- Depth hint style: `natural`
- Target loop control: `True`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_halt_target_repair_strong_l4_20260623_061610/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_oracle_loop_control_halt_l4_20260623_065803/phase1/phase1_step_200.pt`
- Steps: `200`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.538774,
  "halting_kl": 0.703523,
  "loss": 0.623197,
  "mean_expected_loops": 1.549337,
  "mean_halt_entropy": 0.548582,
  "target_loop_abs_error": 0.801766,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.348948,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.869333,
  "group/curriculum_mode/deep_narrow/loss": 0.453268,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.904125,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.76074,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.345875,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.791876,
  "group/curriculum_mode/direct/halting_kl": 0.482443,
  "group/curriculum_mode/direct/loss": 0.849769,
  "group/curriculum_mode/direct/mean_expected_loops": 1.076288,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.265705,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.076288,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.348948,
    "halting_kl": 0.869333,
    "loss": 0.453268,
    "mean_expected_loops": 1.904125,
    "mean_halt_entropy": 0.76074,
    "target_loop_abs_error": 1.345875,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.791876,
    "halting_kl": 0.482443,
    "loss": 0.849769,
    "mean_expected_loops": 1.076288,
    "mean_halt_entropy": 0.265705,
    "target_loop_abs_error": 0.076288,
    "target_mean_loops": 1.0
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
  "mean_expected_loops": 1.549337,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.076288,
    "deep_narrow_mean_expected_loops": 1.904125,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
