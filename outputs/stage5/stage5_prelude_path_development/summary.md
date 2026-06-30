# Stage 5 Unfreeze Recurrent Curriculum: stage5_prelude_path_development

- Status: `finished`
- Source summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json`
- Source checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Checkpoint: `outputs/stage5/stage5_prelude_path_development/unfrozen/unfrozen_recurrent_step_300.pt`
- Training rows: `57`; validation rows: `6`
- Optimizer: `muon`; max steps: `300`
- Bridge prelude grad multiplier: `10.0`
- Save every: `50`
- Interval checkpoints: `5`
- Merge LoRA before unfreeze: `True`

## Validation Loop Sweep

- Loops `1`: expected_ce `0.927838`, loss `0.927838`, mean_expected_loops `1.0`
- Loops `2`: expected_ce `0.914016`, loss `0.975686`, mean_expected_loops `1.996819`
- Loops `4`: expected_ce `1.22003`, loss `1.436108`, mean_expected_loops `3.983641`
- Loops `8`: expected_ce `1.846208`, loss `2.30735`, mean_expected_loops `7.945592`

## Interpretation

- Prelude ablation summary: `outputs/stage5/stage5_prelude_path_development/prelude_ablation.json`
- Bridge prelude weight stats: `{'bridge_prelude_weight_rms': 0.00073597626760602, 'bridge_prelude_weight_max_abs': 0.005899106152355671, 'bridge_state_identity_max_abs_diff': 0.003418166423216462}`
- Any deeper loop beats loop 1 on validation CE: `True`
- Best validation loop by CE: `2`
- Next step: `run_debiased_forced_depth_benchmark_lora_rank0`
