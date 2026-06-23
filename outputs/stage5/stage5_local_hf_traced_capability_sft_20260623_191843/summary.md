# Stage 5 Curriculum SFT - stage5_local_hf_traced_capability_sft_20260623_191843

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_191836', 'summary_json': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_191836/summary.json'}`
- Positive rows: `32`
- Train / validation rows: `29` / `3`
- Train mode counts: `{'deep_narrow': 16, 'direct': 13}`
- Validation mode counts: `{'deep_narrow': 2, 'direct': 1}`
- Depth hint style: `natural`
- Target loop control: `False`
- Learned loop control: `True`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': False, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_learned_loop_control_ce8_continue_l4_20260623_071135/phase1/phase1_step_400.pt`
- Checkpoint: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_191843/phase1/phase1_step_150.pt`
- Steps: `150`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 3.0,
  "expected_ce": 1.40305,
  "halting_kl": 0.292337,
  "loop_control_ce": 0.320229,
  "loss": 3.999966,
  "mean_expected_loops": 1.660203,
  "mean_halt_entropy": 0.49934,
  "target_loop_abs_error": 0.062921,
  "target_mean_loops": 1.666667,
  "group/curriculum_mode/deep_narrow/examples": 2.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.205509,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.25012,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.351701,
  "group/curriculum_mode/deep_narrow/loss": 4.049131,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.947961,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.603814,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.052039,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.0,
  "group/curriculum_mode/direct/examples": 1.0,
  "group/curriculum_mode/direct/expected_ce": 1.798134,
  "group/curriculum_mode/direct/halting_kl": 0.376771,
  "group/curriculum_mode/direct/loop_control_ce": 0.257286,
  "group/curriculum_mode/direct/loss": 3.901636,
  "group/curriculum_mode/direct/mean_expected_loops": 1.084686,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.290394,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.084686,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 2.0,
    "expected_ce": 1.205509,
    "halting_kl": 0.25012,
    "loop_control_ce": 0.351701,
    "loss": 4.049131,
    "mean_expected_loops": 1.947961,
    "mean_halt_entropy": 0.603814,
    "target_loop_abs_error": 0.052039,
    "target_mean_loops": 2.0
  },
  "direct": {
    "examples": 1.0,
    "expected_ce": 1.798134,
    "halting_kl": 0.376771,
    "loop_control_ce": 0.257286,
    "loss": 3.901636,
    "mean_expected_loops": 1.084686,
    "mean_halt_entropy": 0.290394,
    "target_loop_abs_error": 0.084686,
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
  "mean_expected_loops": 1.660203,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.084686,
    "deep_narrow_mean_expected_loops": 1.947961,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
