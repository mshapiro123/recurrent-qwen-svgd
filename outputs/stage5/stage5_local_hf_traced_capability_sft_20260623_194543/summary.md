# Stage 5 Curriculum SFT - stage5_local_hf_traced_capability_sft_20260623_194543

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_194537', 'summary_json': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_194537/summary.json'}`
- Positive rows: `63`
- Train / validation rows: `57` / `6`
- Train mode counts: `{'deep_narrow': 33, 'direct': 24}`
- Validation mode counts: `{'deep_narrow': 4, 'direct': 2}`
- Depth hint style: `natural`
- Target loop control: `False`
- Learned loop control: `True`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': False, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_191843/phase1/phase1_step_150.pt`
- Checkpoint: `outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/phase1/phase1_step_200.pt`
- Steps: `200`
- Max loops: `3`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 6.0,
  "expected_ce": 1.540417,
  "halting_kl": 0.510772,
  "loop_control_ce": 0.765825,
  "loss": 7.728313,
  "mean_expected_loops": 1.775313,
  "mean_halt_entropy": 0.36536,
  "target_loop_abs_error": 0.308823,
  "target_mean_loops": 1.833333,
  "group/curriculum_mode/deep_narrow/examples": 4.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.18521,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.59439,
  "group/curriculum_mode/deep_narrow/loop_control_ce": 0.818609,
  "group/curriculum_mode/deep_narrow/loss": 7.80541,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 1.987598,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 0.306697,
  "group/curriculum_mode/deep_narrow/target_loop_abs_error": 0.287862,
  "group/curriculum_mode/deep_narrow/target_mean_loops": 2.25,
  "group/curriculum_mode/direct/examples": 2.0,
  "group/curriculum_mode/direct/expected_ce": 2.250831,
  "group/curriculum_mode/direct/halting_kl": 0.343537,
  "group/curriculum_mode/direct/loop_control_ce": 0.660258,
  "group/curriculum_mode/direct/loss": 7.574118,
  "group/curriculum_mode/direct/mean_expected_loops": 1.350744,
  "group/curriculum_mode/direct/mean_halt_entropy": 0.482687,
  "group/curriculum_mode/direct/target_loop_abs_error": 0.350744,
  "group/curriculum_mode/direct/target_mean_loops": 1.0
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 4.0,
    "expected_ce": 1.18521,
    "halting_kl": 0.59439,
    "loop_control_ce": 0.818609,
    "loss": 7.80541,
    "mean_expected_loops": 1.987598,
    "mean_halt_entropy": 0.306697,
    "target_loop_abs_error": 0.287862,
    "target_mean_loops": 2.25
  },
  "direct": {
    "examples": 2.0,
    "expected_ce": 2.250831,
    "halting_kl": 0.343537,
    "loop_control_ce": 0.660258,
    "loss": 7.574118,
    "mean_expected_loops": 1.350744,
    "mean_halt_entropy": 0.482687,
    "target_loop_abs_error": 0.350744,
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
  "mean_expected_loops": 1.775313,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.350744,
    "deep_narrow_mean_expected_loops": 1.987598,
    "required_margin": 0.25,
    "observed": true
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
