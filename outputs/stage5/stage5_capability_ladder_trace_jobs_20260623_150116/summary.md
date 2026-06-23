# Capability-Ladder Trace Jobs: stage5_capability_ladder_trace_jobs_20260623_150116

- Status: `ready`
- Source summary: `outputs/stage5/stage5_capability_ladder_mcq_probe_20260623_185608/summary.json`
- Jobs: `132`
- Selected rows: `66`
- Target loops: `{'1': 68, '2': 40, '3': 24}`
- Tiers: `{'base_preservation': 34, 'qwen_0_5b_miss_qwen_1_5b_miss_qwen_3b_solve': 12, 'qwen_0_5b_miss_qwen_1_5b_solve': 20}`

## Next Action

Run training/run_curriculum_job_responses.py with a provider/model map to generate trace responses, then training/collect_capability_ladder_trace_outputs.py to build traced scored rows.
