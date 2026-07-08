# Regression Battery: 2_stage5_depth_support_route_20260705_124320

- Status: `green_noninferior`
- Source suite: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_lineage_regression_battery_current_2_stage5_depth_support_route_20260705_124320/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_depth_support_route_20260705_124320/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Loop: `1`
- Accuracy non-inferiority margin: `0.030`

## Pooled
- `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0006, ci95=[-0.0027, +0.0014], n=7787, base=4001, recurrent=3996
- `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0004, ci95=[-0.0025, +0.0017], n=7787, base=5498, recurrent=5495

## Rows
- `arc_easy` `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0010, ci95=[-0.0035, +0.0016], n=5197, wins=21, losses=26
- `arc_easy` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0010, ci95=[-0.0035, +0.0016], n=5197, wins=21, losses=26
- `arc_challenge` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0000, ci95=[-0.0032, +0.0032], n=2590, wins=9, losses=9
- `arc_challenge` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=+0.0008, ci95=[-0.0028, +0.0043], n=2590, wins=12, losses=10

## Pending Extensions
- Tier 1 natural-text NLL canary is not wired yet.
- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.
