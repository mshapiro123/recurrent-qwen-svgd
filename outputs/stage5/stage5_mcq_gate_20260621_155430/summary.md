# Stage 5 Benchmark Suite - stage5_mcq_gate_20260621_155430

- Status: `completed_with_failures`
- Source summary: `None`
- Checkpoint: `outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt`
- Benchmarks: `['arc_challenge', 'gpqa_lite']`
- Recurrent mode: `phase1`
- Recurrent trajectories: `1`
- Elapsed seconds: `58.16`

## Recurrent vs Base

### arc_challenge
- score target `label`
  - aggregate `mean`: correct delta `0`, accuracy delta `0.0000` (base `20/32`, recurrent `20/32`)
  - paired evidence
    - aggregate `mean`: recurrent `20` / `32`, base `20` / `32`, delta `0`, W/L/T `4/4/24`, p `1.0`
### gpqa_lite
- score target `label`

## Failures

- `prepare` `gpqa_lite`: command failed: /usr/bin/python3 eval/prepare_gpqa_mcq.py --config gpqa_diamond --split train --limit 16 --seed 0 --output_jsonl /content/recurrent-qwen-svgd/data/stage5_benchmark_suite/stage5_mcq_gate_20260621_155430/gpqa_diamond_16.jsonl
