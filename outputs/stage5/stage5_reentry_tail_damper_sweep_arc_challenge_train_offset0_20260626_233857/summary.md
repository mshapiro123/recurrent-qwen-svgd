# Tail-Damper Forced-Depth Sweep - stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857

- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Examples: `512`
- Loops scored: `[1, 2, 3]`
- Tail loops: `[1, 2, 3, 4, 8]`
- Re-entry rescale: `none`

## Calibration

- Correction class: `tail_damper`
- Tail mismatch: `2.087932`
- After damper: `0.418288`
- Damper scales: `[0.6722984550664833, 0.5465434628600714, 0.647222170507681, 0.6951712283448915, 0.5215631171468692, 0.7003045379120896, 0.6544134936280323]`

## Strength Sweep

| strength | loop8 tail ratio | loop1 | loop2 | loop3 | oracle | gap | rescued | harmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 33.650 | 180/512 | 167/512 | 160/512 | 229/512 | 49 | 49 | 70 |
| 0.50 | 7.670 | 180/512 | 168/512 | 158/512 | 229/512 | 49 | 49 | 67 |
| 1.00 | 2.913 | 180/512 | 165/512 | 159/512 | 232/512 | 52 | 52 | 67 |
