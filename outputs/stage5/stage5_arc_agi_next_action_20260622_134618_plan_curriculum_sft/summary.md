# Stage 5 Curriculum SFT - stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft

## Question
Can a strict generated depth/width curriculum improve the deterministic recurrent model without leaking unsafe traces?

## Safety
- SFT gate: `go_train_recurrent_sft`
- Input restore: `{'restored': True, 'source': '/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs/programmatic_direct_deep_001', 'work_dir': 'data/curriculum/programmatic_direct_deep_001', 'summary_json': 'data/curriculum/programmatic_direct_deep_001/summary.json'}`
- Positive rows: `2000`
- Train / validation rows: `1800` / `200`
- Drive preflight: `{'drive_root': '/content/drive/MyDrive/recurrent-qwen-svgd-artifacts', 'available': True, 'allow_no_backup': False}`

## Training
- Resume from: `None`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft/phase1/phase1_step_150.pt`
- Steps: `150`
- Max loops: `4`

## Validation
```json
{
  "lora_recurrent_modules": 84.0,
  "examples": 200.0,
  "expected_ce": 0.18099,
  "halting_kl": 0.428164,
  "loss": 0.215243,
  "mean_expected_loops": 2.954928,
  "mean_halt_entropy": 1.233534,
  "group/curriculum_mode/deep_narrow/examples": 97.0,
  "group/curriculum_mode/deep_narrow/expected_ce": 0.255056,
  "group/curriculum_mode/deep_narrow/halting_kl": 0.202681,
  "group/curriculum_mode/deep_narrow/loss": 0.271271,
  "group/curriculum_mode/deep_narrow/mean_expected_loops": 2.953152,
  "group/curriculum_mode/deep_narrow/mean_halt_entropy": 1.235421,
  "group/curriculum_mode/direct/examples": 103.0,
  "group/curriculum_mode/direct/expected_ce": 0.111238,
  "group/curriculum_mode/direct/halting_kl": 0.640512,
  "group/curriculum_mode/direct/loss": 0.162479,
  "group/curriculum_mode/direct/mean_expected_loops": 2.9566,
  "group/curriculum_mode/direct/mean_halt_entropy": 1.231757
}
```

## Validation By Curriculum Mode
```json
{
  "deep_narrow": {
    "examples": 97.0,
    "expected_ce": 0.255056,
    "halting_kl": 0.202681,
    "loss": 0.271271,
    "mean_expected_loops": 2.953152,
    "mean_halt_entropy": 1.235421
  },
  "direct": {
    "examples": 103.0,
    "expected_ce": 0.111238,
    "halting_kl": 0.640512,
    "loss": 0.162479,
    "mean_expected_loops": 2.9566,
    "mean_halt_entropy": 1.231757
  }
}
```

## Next Decision
If validation is finite and loop depth remains non-collapsed, run the paired benchmark suite before any Phase 2/SVGD training.
