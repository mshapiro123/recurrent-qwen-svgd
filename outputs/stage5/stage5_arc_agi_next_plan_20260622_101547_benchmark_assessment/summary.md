# Stage 5 Broader Benchmark Assessment - stage5_arc_agi_next_plan_20260622_101547_benchmark_assessment

- Status: `needs_review`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_134618_plan_curriculum_sft_benchmark_suite/summary.json`
- Score target / aggregate: `label` / `mean`
- Next step: Inspect benchmark-suite logs and rerun failed slices before using the result.

## Criteria

- `suite_completed` passed `False`: Benchmark suite had failures or did not complete.
- `paired_coverage` passed `False`: Need more paired examples before interpreting broader benchmark deltas.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_challenge` paired `128` / required `128`; delta `-10`; W/L/T `18/28/82`; p `0.18392482137699062`
- `arc_easy` paired `128` / required `1`; delta `-15`; W/L/T `7/22/99`; p `0.008130058646202087`
- `gpqa_lite` paired `0` / required `16`; delta `0`; W/L/T `0/0/0`; p `None`
