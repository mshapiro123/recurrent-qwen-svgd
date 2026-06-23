# Stage 5 Curriculum SFT - stage5_learned_loop_control_halt_l4_20260623_070436

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_learned_loop_control_halt_l4_20260623_070436_mcq_ladder', 'summary_json': 'data/curriculum/stage5_learned_loop_control_halt_l4_20260623_070436_mcq_ladder/summary.json'}`
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
- Resume from: `outputs/stage5/stage5_halt_target_repair_strong_l4_20260623_061610/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_learned_loop_control_halt_l4_20260623_070436/phase1/phase1_step_400.pt`
- Steps: `400`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.4684,
  "halting_kl": 0.687974,
  "loop_control_ce": 2.18081,
  "loss": 4.912577,
  "mean_expected_loops": 1.811266,
  "mean_halt_entropy": 0.68337,
  "target_loop_abs_error": 1.019653,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.354049,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.916519,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 2.905321,
  "group/curriculum_mode/deep_narrow/loss": 6.274674,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.942661,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.671228,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.307339,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.620867,
  "group/curriculum_mode/direct/halting_kl": 0.383248,
  "group/curriculum_mode/direct/loop_control_ce": 1.214795,
  "group/curriculum_mode/direct/loss": 3.096447,
  "group/curriculum_mode/direct/mean_expected_loops": 1.636073,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.69956,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.636073,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.354049,
    "halting_kl": 0.916519,
    "loop_control_ce": 2.905321,
    "loss": 6.274674,
    "mean_expected_loops": 1.942661,
    "mean_halt_entropy": 0.671228,
    "target_loop_abs_error": 1.307339,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.620867,
    "halting_kl": 0.383248,
    "loop_control_ce": 1.214795,
    "loss": 3.096447,
    "mean_expected_loops": 1.636073,
    "mean_halt_entropy": 0.69956,
    "target_loop_abs_error": 0.636073,
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
  "mean_expected_loops": 1.811266,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.636073,
    "deep_narrow_mean_expected_loops": 1.942661,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
