# Stage 5 Re-entry Repair Smoke - stage5_reentry_repair_smoke_20260625_114554

- Source checkpoint: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/phase1_direct_preserve/phase1_step_75.pt`
- Checkpoint source: `stage2_norm_assessment`
- Checkpoint from Stage 2 norm assessment: `True`
- Trained checkpoint: `outputs/stage5/stage5_reentry_repair_smoke_20260625_114554/phase1_reentry_repair/phase1_step_25.pt`
- Cell version: `stage5_reentry_repair_smoke_v1_trainable`
- Max steps: `25`
- Optimizer modules: `bridge,reentry,halt`
- Re-entry mode: `entry_rms`
- Use re-entry adapter: `True`

## Bridge Liveness
| stage | gate | proj identity max diff | proj bias max | bridge delta RMS | weight grad RMS | bias grad RMS | loop4 out/in RMS | loop8 out/in RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0042314529418945 | 1.007548749446869 |
| post | 1.0002319812774658 | 0.0002565568720456213 | 0.00025284389266744256 | 0.08213316649198532 | 0.38929283618927 | 0.007555652409791946 | 1.0042655318975449 | 1.0074928253889084 |

## Re-entry Adapter
| stage | scale identity max diff | bias max abs | adapter delta RMS | scale grad RMS | bias grad RMS |
|---|---:|---:|---:|---:|---:|
| pre | 0.0 | 0.0 | 0.0 | 11.342706680297852 | 0.007554761134088039 |
| post | 0.0002409219741821289 | 0.00025282433489337564 | 0.0018250071443617344 | 11.344264030456543 | 0.0075557054951786995 |

## Training Smoke Metrics
- Last logged step: `20`
- Last logged loss: `1.4865`
- Last logged expected CE: `1.3386`
- Last logged mean expected loops: `1.2025`
- Last logged target loop abs error: `0.7975`
- Last logged halting target NLL: `1.5986`

## Loop-1 Preservation
- Source loop-1 best hits: `1/6`
- Trained loop-1 best hits: `1/6`
- Best-hit delta: `0`
- Candidate-hit delta: `0`

## Readout Pause
This run intentionally stops after Stage 3. Review bridge movement and loop behavior before recovery training.
