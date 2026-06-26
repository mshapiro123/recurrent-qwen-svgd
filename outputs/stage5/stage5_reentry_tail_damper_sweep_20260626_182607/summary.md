# Tail-Damper Forced-Depth Sweep - stage5_reentry_tail_damper_sweep_20260626_182607

- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Examples: `256`
- Loops scored: `[1, 2, 3]`
- Tail loops: `[1, 2, 3, 4, 8]`
- Re-entry rescale: `none`

## Calibration

- Correction class: `tail_damper`
- Tail mismatch: `2.110969`
- After damper: `0.425675`
- Damper scales: `[0.6696660591827204, 0.553306075506213, 0.6190147475859962, 0.6686157341805897, 0.5114603848996762, 0.7036401033167234, 0.7227361800113572]`

## Strength Sweep

| strength | loop8 tail ratio | loop1 | loop2 | loop3 | oracle | gap | rescued | harmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 34.144 | 88/256 | 87/256 | 86/256 | 116/256 | 28 | 28 | 31 |
| 0.25 | 15.297 | 88/256 | 86/256 | 82/256 | 116/256 | 28 | 28 | 33 |
| 0.50 | 7.692 | 88/256 | 82/256 | 83/256 | 117/256 | 29 | 29 | 35 |
| 0.75 | 4.469 | 88/256 | 84/256 | 83/256 | 117/256 | 29 | 29 | 34 |
| 1.00 | 2.944 | 88/256 | 84/256 | 83/256 | 117/256 | 29 | 29 | 34 |
