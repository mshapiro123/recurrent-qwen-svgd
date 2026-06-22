# Stage 5 Curriculum SFT - stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': False, 'work_dir': 'data/curriculum/programmatic_direct_deep_001', 'summary_json': 'data/curriculum/programmatic_direct_deep_001/summary.json'}`
- Positive rows: `2000`
- Train / validation rows: `1800` / `200`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': False}`

## Training
- Resume from: `None`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Steps: `150`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 200.0,
  "expected_ce": 0.189663,
  "halting_kl": 0.434806,
  "loss": 0.224447,
  "mean_expected_loops": 2.964177,
  "mean_halt_entropy": 1.228205,
  "group/curriculum_mode/deep_narrow/examples": 97.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.268978,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.20589,
  "group/curriculum_mode/deep_narrow/loss": 0.285449,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 2.963509,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.229711,
  "group/curriculum_mode/direct/examples": 103.0,
  "group/curriculum_mode/direct/expected_ce": 0.114968,
  "group/curriculum_mode/direct/halting_kl": 0.650387,
  "group/curriculum_mode/direct/loss": 0.166999,
  "group/curriculum_mode/direct/mean_expected_loops": 2.964806,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.226787
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 97.0,
    "expected_ce": 0.268978,
    "halting_kl": 0.20589,
    "loss": 0.285449,
    "mean_expected_loops": 2.963509,
    "mean_halt_entropy": 1.229711
  },
  "direct": {
    "examples": 103.0,
    "expected_ce": 0.114968,
    "halting_kl": 0.650387,
    "loss": 0.166999,
    "mean_expected_loops": 2.964806,
    "mean_halt_entropy": 1.226787
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
