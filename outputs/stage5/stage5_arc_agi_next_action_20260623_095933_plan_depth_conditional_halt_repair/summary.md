# Stage 5 Curriculum SFT - stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair_mcq_ladder', 'summary_json': 'data/curriculum/stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair_mcq_ladder/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Train mode counts: `{'deep_narrow': 36, 'direct': 30}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 3}`
- Depth hint style: `none`
- Target loop control: `False`
- Learned loop control: `False`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260623_095933_plan_depth_conditional_halt_repair/phase1/phase1_step_100.pt`
- Steps: `100`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.378417,
  "halting_kl": 0.522875,
  "loss": 0.441162,
  "mean_expected_loops": 1.832545,
  "mean_halt_entropy": 0.836447,
  "target_loop_abs_error": 0.991965,
  "target_mean_loops": 2.285714,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.433035,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.683023,
  "group/curriculum_mode/deep_narrow/loss": 0.514998,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.985508,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.887037,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 1.264492,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 3.25,
  "group/curriculum_mode/direct/examples": 3.0,
  "group/curriculum_mode/direct/expected_ce": 0.305592,
  "group/curriculum_mode/direct/halting_kl": 0.309346,
  "group/curriculum_mode/direct/loss": 0.342714,
  "group/curriculum_mode/direct/mean_expected_loops": 1.628595,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.768992,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.628595,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 0.433035,
    "halting_kl": 0.683023,
    "loss": 0.514998,
    "mean_expected_loops": 1.985508,
    "mean_halt_entropy": 0.887037,
    "target_loop_abs_error": 1.264492,
    "target_mean_loops": 3.25
  },
  "direct": {
    "examples": 3.0,
    "expected_ce": 0.305592,
    "halting_kl": 0.309346,
    "loss": 0.342714,
    "mean_expected_loops": 1.628595,
    "mean_halt_entropy": 0.768992,
    "target_loop_abs_error": 0.628595,
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
  "mean_expected_loops": 1.832545,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.628595,
    "deep_narrow_mean_expected_loops": 1.985508,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
