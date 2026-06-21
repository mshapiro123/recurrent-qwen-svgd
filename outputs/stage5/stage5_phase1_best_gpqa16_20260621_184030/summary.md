# Stage 5 Benchmark Suite - stage5_phase1_best_gpqa16_20260621_184030

- Status: `completed_with_failures`
- Source summary: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/summary.json`
- Checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt`
- Benchmarks: `['gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `1.87`

## Recurrent vs Base

### gpqa_lite
- score target `label`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_phase1_best_gpqa16_20260621_184030/gpqa_diamond_16.jsonl
