# Stage 5 Curriculum SFT - stage5_learned_loop_control_ce8_continue_l4_20260623_071135

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_learned_loop_control_ce8_continue_l4_20260623_071135_mcq_ladder', 'summary_json': 'data/curriculum/stage5_learned_loop_control_ce8_continue_l4_20260623_071135_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Train mode counts: `{'deep_narrow': 36, 'direct': 30}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 3}`
- Depth hint style: `natural`
- Target loop control: `False`
- Learned loop control: `True`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_learned_loop_control_halt_l4_20260623_070436/phase1/phase1_step_400.pt`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Steps: `400`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.513664,
  "halting_kl": 0.502499,
  "loop_control_ce": 1.774681,
  "loss": 14.771414,
  "mean_expected_loops": 1.819131,
  "mean_halt_entropy": 0.802629,
  "target_loop_abs_error": 0.892717,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.361744,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.657417,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 2.550281,
  "group/curriculum_mode/deep_narrow/loss": 20.842885,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 2.060613,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.876737,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.189387,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.716225,
  "group/curriculum_mode/direct/halting_kl": 0.295941,
  "group/curriculum_mode/direct/loop_control_ce": 0.740548,
  "group/curriculum_mode/direct/loss": 6.676119,
  "group/curriculum_mode/direct/mean_expected_loops": 1.497156,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.703817,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.497156,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.361744,
    "halting_kl": 0.657417,
    "loop_control_ce": 2.550281,
    "loss": 20.842885,
    "mean_expected_loops": 2.060613,
    "mean_halt_entropy": 0.876737,
    "target_loop_abs_error": 1.189387,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.716225,
    "halting_kl": 0.295941,
    "loop_control_ce": 0.740548,
    "loss": 6.676119,
    "mean_expected_loops": 1.497156,
    "mean_halt_entropy": 0.703817,
    "target_loop_abs_error": 0.497156,
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
  "mean_expected_loops": 1.819131,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.497156,
    "deep_narrow_mean_expected_loops": 2.060613,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
