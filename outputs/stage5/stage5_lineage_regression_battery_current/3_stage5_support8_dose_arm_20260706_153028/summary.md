# Regression Battery: 3_stage5_support8_dose_arm_20260706_153028

- Status: `green_noninferior`
- Source suite: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_lineage_regression_battery_current_3_stage5_support8_dose_arm_20260706_153028/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_support8_dose_arm_20260706_153028/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Loop: `1`
- Accuracy non-inferiority margin: `0.030`

## Pooled
- `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0005, ci95=[-0.0025, +0.0015], n=7787, base=4001, recurrent=3997
- `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0009, ci95=[-0.0029, +0.0011], n=7787, base=5498, recurrent=5491

## Rows
- `arc_easy` `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0010, ci95=[-0.0034, +0.0015], n=5197, wins=18, losses=23
- `arc_easy` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0015, ci95=[-0.0039, +0.0008], n=5197, wins=16, losses=24
- `arc_challenge` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0004, ci95=[-0.0032, +0.0040], n=2590, wins=12, losses=11
- `arc_challenge` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=+0.0004, ci95=[-0.0034, +0.0042], n=2590, wins=13, losses=12

## Pending Extensions
- Tier 1 natural-text NLL canary is not wired yet.
- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.
