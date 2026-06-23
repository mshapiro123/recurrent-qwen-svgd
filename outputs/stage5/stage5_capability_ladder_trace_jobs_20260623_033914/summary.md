# Capability-Ladder Trace Jobs: stage5_capability_ladder_trace_jobs_20260623_033914

- Status: `ready`
- Source summary: `outputs/stage5/stage5_capability_ladder_mcq_probe_20260623_033635/summary.json`
- Jobs: `146`
- Selected rows: `73`
- Target loops: `{'1': 66, '2': 44, '3': 20, '4': 16}`
- Tiers: `{'base_preservation': 33, 'qwen_0_5b_miss_qwen_1_5b_miss_qwen_3b_miss_qwen_7b_solve': 8, 'qwen_0_5b_miss_qwen_1_5b_miss_qwen_3b_solve': 10, 'qwen_0_5b_miss_qwen_1_5b_solve': 22}`

## Next Action

Run training/run_curriculum_job_responses.py with a provider/model map to generate trace responses, then training/collect_capability_ladder_trace_outputs.py to build traced scored rows.
