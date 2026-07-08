# Regression Battery: 4_stage5_n24_support12_rung_20260707_140139

- Status: `green_noninferior`
- Source suite: `/content/recurrent-qwen-svgd/outputs/stage5/stage5_lineage_regression_battery_current_4_stage5_n24_support12_rung_20260707_140139/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt`
- Loop: `1`
- Accuracy non-inferiority margin: `0.030`

## Pooled
- `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0010, ci95=[-0.0010, +0.0030], n=7787, base=4001, recurrent=4009
- `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0021, ci95=[-0.0041, -0.0000], n=7787, base=5498, recurrent=5482

## Rows
- `arc_easy` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0006, ci95=[-0.0020, +0.0031], n=5197, wins=24, losses=21
- `arc_easy` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0027, ci95=[-0.0051, -0.0003], n=5197, wins=13, losses=27
- `arc_challenge` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0019, ci95=[-0.0014, +0.0052], n=2590, wins=12, losses=7
- `arc_challenge` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0008, ci95=[-0.0045, +0.0029], n=2590, wins=11, losses=13

## Pending Extensions
- Tier 1 natural-text NLL canary is not wired yet.
- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.
