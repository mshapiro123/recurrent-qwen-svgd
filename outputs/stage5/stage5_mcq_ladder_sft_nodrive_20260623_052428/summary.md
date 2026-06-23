# Stage 5 Curriculum SFT - stage5_mcq_ladder_sft_nodrive_20260623_052428

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_capability_ladder_mcq_probe_20260623_023702', 'summary_json': 'data/curriculum/stage5_capability_ladder_mcq_probe_20260623_023702/summary.json'}`
- Positive rows: `73`
- Train / validation rows: `66` / `7`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `outputs/stage5/stage5_traced_capability_ladder_sft_durable_20260623_044343/phase1/phase1_step_75.pt`
- Checkpoint: `outputs/stage5/stage5_mcq_ladder_sft_nodrive_20260623_052428/phase1/phase1_step_150.pt`
- Steps: `150`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 7.0,
  "expected_ce": 0.355546,
  "halting_kl": 0.392573,
  "loss": 0.402654,
  "mean_expected_loops": 3.139538,
  "mean_halt_entropy": 1.124877,
  "group/curriculum_mode/deep_narrow/examples": 6.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.364144,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.3152,
  "group/curriculum_mode/deep_narrow/loss": 0.401968,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 3.139508,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.124868,
  "group/curriculum_mode/direct/examples": 1.0,
  "group/curriculum_mode/direct/expected_ce": 0.303953,
  "group/curriculum_mode/direct/halting_kl": 0.856812,
  "group/curriculum_mode/direct/loss": 0.406771,
  "group/curriculum_mode/direct/mean_expected_loops": 3.13972,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.124932
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 6.0,
    "expected_ce": 0.364144,
    "halting_kl": 0.3152,
    "loss": 0.401968,
    "mean_expected_loops": 3.139508,
    "mean_halt_entropy": 1.124868
  },
  "direct": {
    "examples": 1.0,
    "expected_ce": 0.303953,
    "halting_kl": 0.856812,
    "loss": 0.406771,
    "mean_expected_loops": 3.13972,
    "mean_halt_entropy": 1.124932
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
  "mean_expected_loops": 3.139538,
  "require_depth_gradient": false,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 3.13972,
    "deep_narrow_mean_expected_loops": 3.139508,
    "required_margin": 0.15,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
