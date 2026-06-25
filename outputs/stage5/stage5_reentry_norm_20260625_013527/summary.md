# Stage 5 Re-entry Norm - stage5_reentry_norm_20260625_013527

- Checkpoint: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/phase1_direct_preserve/phase1_step_75.pt`
- Prompts/tasks: `eval/smoke_exact_tasks_v2.jsonl`
- Cell version: `stage5_reentry_norm_v1_eval_only`

## Drift
| mode | exit/entry RMS | loop8 input/entry RMS | loop8 output/entry RMS | subspace overlap |
|---|---:|---:|---:|---:|
| none | 1.002428486943245 | 1.0299247801303864 | 1.0373438447713852 | 0.3702599108219147 |
| entry_rms | 1.002428486943245 | 1.0005650743842125 | 1.0081169456243515 | 0.3702599108219147 |

## Effective Pathways
| mode | initial distance | final distance | spread ratio | q2 pathways | unique next-token argmax |
|---|---:|---:|---:|---:|---:|
| none | 0.47468289360404015 | 83.33644771575928 | 179.48424157522487 | 1.7299222946166992 | 7.375 |
| entry_rms | 0.47468289360404015 | 55.199382305145264 | 117.48429077000047 | 1.4507311433553696 | 7.625 |

## Candidate Conversion
| mode | task groups | best | candidates | mean unique |
|---|---:|---:|---:|---:|
| none | 56 | 20 | 240/672 | 1.0 |
| entry_rms | 56 | 20 | 240/672 | 1.0 |

## Readout Pause
This run intentionally stops after Stage 2. Review before implementing trainable bridge/re-entry repair.
