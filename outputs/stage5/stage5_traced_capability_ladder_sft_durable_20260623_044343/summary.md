# Stage 5 Curriculum SFT - stage5_traced_capability_ladder_sft_durable_20260623_044343

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_043104', 'summary_json': 'data/curriculum/stage5_capability_ladder_trace_collection_20260623_043104/summary.json'}`
- Positive rows: `32`
- Train / validation rows: `29` / `3`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': True}`
- Validation status: `validation_sane`
- Validation issues: `[]`

## Training
- Resume from: `None`
- Checkpoint: `outputs/stage5/stage5_traced_capability_ladder_sft_durable_20260623_044343/phase1/phase1_step_75.pt`
- Steps: `75`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 3.0,
  "expected_ce": 2.203661,
  "halting_kl": 0.589096,
  "loss": 2.250788,
  "mean_expected_loops": 3.111473,
  "mean_halt_entropy": 1.144066,
  "group/curriculum_mode/deep_narrow/examples": 2.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 1.956955,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.468432,
  "group/curriculum_mode/deep_narrow/loss": 1.994429,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 3.107868,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.146727,
  "group/curriculum_mode/direct/examples": 1.0,
  "group/curriculum_mode/direct/expected_ce": 2.697073,
  "group/curriculum_mode/direct/halting_kl": 0.830423,
  "group/curriculum_mode/direct/loss": 2.763506,
  "group/curriculum_mode/direct/mean_expected_loops": 3.118684,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.138745
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 2.0,
    "expected_ce": 1.956955,
    "halting_kl": 0.468432,
    "loss": 1.994429,
    "mean_expected_loops": 3.107868,
    "mean_halt_entropy": 1.146727
  },
  "direct": {
    "examples": 1.0,
    "expected_ce": 2.697073,
    "halting_kl": 0.830423,
    "loss": 2.763506,
    "mean_expected_loops": 3.118684,
    "mean_halt_entropy": 1.138745
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
  "mean_expected_loops": 3.111473,
  "require_depth_gradient": false,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 3.118684,
    "deep_narrow_mean_expected_loops": 3.107868,
    "required_margin": 0.25,
    "observed": false
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
