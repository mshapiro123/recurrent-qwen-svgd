# Stage 5 Broader Benchmark Assessment - stage5_debiased_benchmark_assessment_20260625_121302

- Status: `needs_review`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_115004/summary.json`
- Score target / aggregate: `cyclic_label_aggregated` / `permutation_mean`
- Next step: Inspect benchmark-suite logs and rerun failed slices before using the result.

## Criteria

- `suite_completed` passed `False`: Benchmark suite had failures or did not complete.
- `paired_coverage` passed `False`: Need more paired examples before interpreting broader benchmark deltas.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_easy` paired `128` / required `128`; delta `3`; W/L/T `4/1/123`; p `0.375`
- `arc_challenge` paired `128` / required `128`; delta `-1`; W/L/T `1/2/125`; p `1.0`
- `gpqa_lite` paired `0` / required `16`; delta `0`; W/L/T `0/0/0`; p `None`
