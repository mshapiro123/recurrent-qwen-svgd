# Stage 5 Unfreeze Recurrent Curriculum: stage5_unfreeze_recurrent_curriculum_20260628_024602

- Status: `finished`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json`
- Source checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_024602/unfrozen/unfrozen_recurrent_step_50.pt`
- Training rows: `57`; validation rows: `6`
- Optimizer: `muon`; max steps: `50`
- Merge LoRA before unfreeze: `True`

## Validation Loop Sweep

- Loops `1`: expected_ce `1.00415`, loss `1.00415`, mean_expected_loops `1.0`
- Loops `2`: expected_ce `1.174033`, loss `1.196952`, mean_expected_loops `1.784663`
- Loops `4`: expected_ce `1.229243`, loss `1.265261`, mean_expected_loops `1.969425`
- Loops `8`: expected_ce `1.238367`, loss `1.282683`, mean_expected_loops `1.988822`

## Interpretation

- Any deeper loop beats loop 1 on validation CE: `False`
- Best validation loop by CE: `1`
- Next step: `run_debiased_forced_depth_benchmark_lora_rank0`
