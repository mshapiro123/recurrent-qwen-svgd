# Regression Battery: 1_stage5_chain_continuation_attribution_20260704_163056

- Status: `green_noninferior`
- Source suite: `C:/Users/mshap/Documents/Codex/2026-06-15/below-is-a-codex-ready-handoff/outputs/stage5/stage5_regression_battery_loop1_current_1_stage5_chain_continuation_attribution_20260704_163056/summary.json`
- Checkpoint: `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_chain_continuation_attribution_20260704_163056/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt`
- Loop: `1`
- Accuracy non-inferiority margin: `0.030`

## Pooled
- `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0001, ci95=[-0.0022, +0.0019], n=7787, base=4001, recurrent=4000
- `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0017, ci95=[-0.0036, +0.0003], n=7787, base=5498, recurrent=5485

## Rows
- `arc_easy` `content_question_only:mean`: verdict=`green_noninferior`, delta=+0.0004, ci95=[-0.0021, +0.0029], n=5197, wins=23, losses=21
- `arc_easy` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0023, ci95=[-0.0046, +0.0000], n=5197, wins=13, losses=25
- `arc_challenge` `content_question_only:mean`: verdict=`green_noninferior`, delta=-0.0012, ci95=[-0.0048, +0.0025], n=2590, wins=10, losses=13
- `arc_challenge` `cyclic_label_aggregated:permutation_mean`: verdict=`green_noninferior`, delta=-0.0004, ci95=[-0.0040, +0.0032], n=2590, wins=11, losses=12

## Pending Extensions
- Tier 1 natural-text NLL canary is not wired yet.
- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.
