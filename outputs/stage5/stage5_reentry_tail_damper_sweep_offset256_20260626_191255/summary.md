# Tail-Damper Forced-Depth Sweep - stage5_reentry_tail_damper_sweep_offset256_20260626_191255

- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Examples: `43`
- Loops scored: `[1, 2, 3]`
- Tail loops: `[1, 2, 3, 4, 8]`
- Re-entry rescale: `none`

## Calibration

- Correction class: `tail_damper`
- Tail mismatch: `1.910262`
- After damper: `0.405668`
- Damper scales: `[0.6931606889784759, 0.5782763710858294, 0.7128397195190034, 0.6533378659949781, 0.49660190326608844, 0.7228137933098839, 0.7736424480536456]`

## Strength Sweep

| strength | loop8 tail ratio | loop1 | loop2 | loop3 | oracle | gap | rescued | harmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 32.600 | 11/43 | 8/43 | 7/43 | 15/43 | 4 | 4 | 8 |
| 0.25 | 15.167 | 11/43 | 7/43 | 9/43 | 16/43 | 5 | 5 | 7 |
| 0.50 | 7.918 | 11/43 | 7/43 | 8/43 | 15/43 | 4 | 4 | 8 |
| 0.75 | 4.634 | 11/43 | 7/43 | 7/43 | 14/43 | 3 | 3 | 8 |
| 1.00 | 3.001 | 11/43 | 7/43 | 9/43 | 15/43 | 4 | 4 | 7 |
