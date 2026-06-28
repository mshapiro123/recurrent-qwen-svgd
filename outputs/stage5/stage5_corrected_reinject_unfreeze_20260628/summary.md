# Stage 5 Unfreeze Recurrent Curriculum: stage5_corrected_reinject_unfreeze_20260628

- Status: `finished`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json`
- Source checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Checkpoint: `outputs/stage5/stage5_corrected_reinject_unfreeze_20260628/unfrozen/unfrozen_recurrent_step_50.pt`
- Training rows: `57`; validation rows: `6`
- Optimizer: `muon`; max steps: `50`
- Merge LoRA before unfreeze: `True`

## Validation Loop Sweep

- Loops `1`: expected_ce `1.073213`, loss `1.073213`, mean_expected_loops `1.0`
- Loops `2`: expected_ce `1.255144`, loss `1.279378`, mean_expected_loops `1.801065`
- Loops `4`: expected_ce `1.364016`, loss `1.400634`, mean_expected_loops `2.136203`
- Loops `8`: expected_ce `1.45039`, loss `1.488492`, mean_expected_loops `2.291135`

## Interpretation

- Any deeper loop beats loop 1 on validation CE: `False`
- Best validation loop by CE: `1`
- Next step: `run_debiased_forced_depth_benchmark_lora_rank0`
