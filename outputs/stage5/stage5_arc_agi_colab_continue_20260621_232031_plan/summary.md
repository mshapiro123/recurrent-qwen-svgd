# Stage 5 Next-Run Plan - stage5_arc_agi_colab_continue_20260621_232031_plan

- Source summary: `outputs/stage5/stage5_benchmark_assessment_20260621_183952/summary.json`
- Source kind: `benchmark_suite_assessment`

## Recommended Actions

1. **Run competence-preserving ARC-mix proxy gate**
   - Priority: `10`
   - Reason: Credit-saving probe: the broader benchmark gate says recurrent still trails base, mostly through competence tax; run one ARC-Easy-weighted mixed-objective proxy gate and stop before a full assessment.
   - Command: `STAGE5_ARC_MIX_RUN_ID=stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe STAGE5_ARC_MIX_SOURCE_SUMMARY=outputs/stage5/stage5_benchmark_assessment_20260621_183952/summary.json STAGE5_ARC_MIX_ARMS=arc_mix_response_w005_lr2e6 STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT=2 STAGE5_ARC_MIX_ARC_EASY_REPEAT=4 STAGE5_ARC_MIX_ARC_EVAL_LIMIT=128 STAGE5_ARC_MIX_OPUS_LIMIT=3000 python colab/run_stage5_balanced_arc_mix_gate.py`
