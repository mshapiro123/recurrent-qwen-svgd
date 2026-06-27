# Tail-Damper Forced-Depth Sweep - fixed_tail_damper_depth_readout

- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/phase1/phase1_step_100.pt`
- Examples: `512`
- Loops scored: `[1, 2, 3]`
- Tail loops: `[1, 2, 3, 4, 8]`
- Re-entry rescale: `entry_rms`

## Calibration

- Correction class: `tail_damper`
- Tail mismatch: `2.087932`
- After damper: `0.418288`
- Damper scales: `[0.6722984550664833, 0.5465434628600714, 0.647222170507681, 0.6951712283448915, 0.5215631171468692, 0.7003045379120896, 0.6544134936280323]`

## Strength Sweep

| strength | loop8 tail ratio | loop1 | loop2 | loop3 | oracle | gap | rescued | harmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 33.150 | 179/512 | 173/512 | 159/512 | 228/512 | 49 | 49 | 68 |
| 1.00 | 2.995 | 179/512 | 168/512 | 159/512 | 233/512 | 54 | 54 | 70 |
