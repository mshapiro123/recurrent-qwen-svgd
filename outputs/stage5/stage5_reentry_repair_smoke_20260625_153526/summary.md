# Stage 5 Re-entry Repair Smoke - stage5_reentry_repair_smoke_20260625_153526

- Source checkpoint: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/phase1_direct_preserve/phase1_step_75.pt`
- Checkpoint source: `stage2_norm_assessment`
- Checkpoint from Stage 2 norm assessment: `True`
- Trained checkpoint: `outputs/stage5/stage5_reentry_repair_smoke_20260625_153526/phase1_reentry_repair/phase1_step_25.pt`
- Cell version: `stage5_reentry_repair_smoke_v2_spectral_optional`
- Max steps: `25`
- Optimizer modules: `bridge,reentry,halt`
- Re-entry mode: `entry_rms`
- Use re-entry adapter: `True`
- Re-entry adapter mode: `spectral`

## Bridge Liveness
| stage | gate | proj identity max diff | proj bias max | bridge delta RMS | weight grad RMS | bias grad RMS | loop4 out/in RMS | loop8 out/in RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0042328536510468 | 1.0075366646051407 |
| post | 1.000227928161621 | 0.00025817271671257913 | 0.0002511607308406383 | 0.08328552544116974 | 0.3893011212348938 | 0.007555678486824036 | 1.0042699128389359 | 1.0074955970048904 |

## Re-entry Adapter
| stage | scale identity max diff | bias max abs | adapter delta RMS | scale grad RMS | bias grad RMS |
|---|---:|---:|---:|---:|---:|
| pre | 0.0 | 0.0 | 0.00446340162307024 | 0.0 | 0.0 |
| post | 0.0 | 0.0 | 0.005200215615332127 | 0.0 | 0.0 |

## Training Smoke Metrics
- Last logged step: `20`
- Last logged loss: `1.9402`
- Last logged expected CE: `1.6078`
- Last logged mean expected loops: `1.7387`
- Last logged target loop abs error: `1.2613`
- Last logged halting target NLL: `5.0349`

## Loop-1 Preservation
- Source loop-1 best hits: `1/6`
- Trained loop-1 best hits: `1/6`
- Best-hit delta: `0`
- Candidate-hit delta: `0`

## Readout Pause
This run intentionally stops after Stage 3. Review bridge movement and loop behavior before recovery training.
