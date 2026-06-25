# Stage 5 Re-entry Drift - stage5_reentry_drift_20260625_011444

- Checkpoint: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/phase1_direct_preserve/phase1_step_75.pt`
- Prompts: `eval/smoke_exact_tasks_v2.jsonl`
- Max loops: `8`
- Cell version: `stage5_reentry_drift_v1_readonly`

## Aggregate
- Mean entry RMS: `11.867156028747559`
- Mean exit RMS: `11.892430305480957`
- Mean exit/entry RMS: `1.002428486943245`
- Mean pooled entry/exit cosine: `0.9757358208298683`
- Subspace overlap: `0.3702599108219147`
- Aligned dims >= 0.8: `1`
- Aligned dims >= 0.9: `1`

## Bridge
- Bridge gate: `0.0`
- Projection identity max abs diff: `0.0`
- Bridge delta RMS: `0.0`
- Gate grad abs: `0.0`
- Weight grad RMS: `0.0`
- Bias grad RMS: `0.0`

## Loop Drift
| loop | input/entry RMS | output/entry RMS | output/input RMS | bridge delta RMS | input-output cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 1.002428486943245 | 1.002428486943245 | 0.0 | 0.9757358208298683 |
| 2 | 1.002428486943245 | 1.0049241036176682 | 1.0024871230125427 | 0.0 | 0.9914550185203552 |
| 3 | 1.0049241036176682 | 1.0082857310771942 | 1.0033385753631592 | 0.0 | 0.9959690943360329 |
| 4 | 1.0082857310771942 | 1.0124891549348831 | 1.0041560381650925 | 0.0 | 0.9971648827195168 |
| 5 | 1.0124891549348831 | 1.0174884349107742 | 1.0049159824848175 | 0.0 | 0.9976935833692551 |
| 6 | 1.0174884349107742 | 1.0233152955770493 | 1.00569386780262 | 0.0 | 0.9979888498783112 |
| 7 | 1.0233152955770493 | 1.0299247801303864 | 1.006412461400032 | 0.0 | 0.9982864931225777 |
| 8 | 1.0299247801303864 | 1.0373438447713852 | 1.0071412026882172 | 0.0 | 0.9984563440084457 |

## Readout Pause
This run intentionally stops after Stage 1. Review these numbers before running re-entry normalization or bridge repair.
