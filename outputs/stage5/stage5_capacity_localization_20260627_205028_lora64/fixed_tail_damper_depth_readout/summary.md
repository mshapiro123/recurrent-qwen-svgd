# Tail-Damper Forced-Depth Sweep - fixed_tail_damper_depth_readout

- Checkpoint: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64_curriculum_sft/phase1/phase1_step_100.pt`
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
| 0.00 | 32.731 | 182/512 | 157/512 | 158/512 | 229/512 | 47 | 47 | 68 |
| 1.00 | 2.732 | 182/512 | 157/512 | 158/512 | 235/512 | 53 | 53 | 74 |
