# Stage 5 Unfreeze Recurrent Curriculum: stage5_unfreeze_recurrent_curriculum_20260628_023241

- Status: `finished`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json`
- Source checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Checkpoint: `outputs/stage5/stage5_unfreeze_recurrent_curriculum_20260628_023241/unfrozen/unfrozen_recurrent_step_50.pt`
- Training rows: `57`; validation rows: `6`
- Optimizer: `muon`; max steps: `50`
- Merge LoRA before unfreeze: `True`

## Validation Loop Sweep

- Loops `1`: expected_ce `1.202172`, loss `1.202172`, mean_expected_loops `1.0`
- Loops `2`: expected_ce `1.4133`, loss `1.43632`, mean_expected_loops `1.809983`
- Loops `4`: expected_ce `1.465194`, loss `1.502267`, mean_expected_loops `1.971606`
- Loops `8`: expected_ce `1.472253`, loss `1.518222`, mean_expected_loops `1.986687`

## Interpretation

- Any deeper loop beats loop 1 on validation CE: `False`
- Best validation loop by CE: `1`
- Next step: `run_debiased_forced_depth_benchmark_lora_rank0`
