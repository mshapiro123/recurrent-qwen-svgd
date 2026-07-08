# Regression Battery: 1_stage5_reentry_recovery_20260625_154210

- Status: `green_noninferior`
- Source suite: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_lineage_regression_battery_current_1_stage5_reentry_recovery_20260625_154210/summary.json`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260625_154210_curriculum_sft/phase1/phase1_step_75.pt`
- Loop: `1`
- Accuracy non-inferiority margin: `0.030`

## Pooled
- `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0049, ci95=[-0.0005, +0.0103], n=7787, base=4001, recurrent=4039
- `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=+0.0015, ci95=[-0.0020, +0.0051], n=7787, base=5498, recurrent=5510

## Rows
- `arc_easy` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0058, ci95=[-0.0011, +0.0126], n=5197, wins=181, losses=151
- `arc_easy` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=+0.0015, ci95=[-0.0025, +0.0056], n=5197, wins=61, losses=53
- `arc_challenge` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0031, ci95=[-0.0055, +0.0117], n=2590, wins=69, losses=61
- `arc_challenge` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=+0.0015, ci95=[-0.0055, +0.0086], n=2590, wins=45, losses=41

## Pending Extensions
- Tier 1 natural-text NLL canary is not wired yet.
- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.
