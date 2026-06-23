# Stage 5 Broader Benchmark Assessment - stage5_debiased_benchmark_assessment_20260623_155046

- Status: `needs_benchmark_confirmation`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260623_145438/summary.json`
- Score target / aggregate: `cyclic_label_aggregated` / `permutation_mean`
- Next step: Rerun the broader benchmark suite with larger ARC-Challenge/GPQA-lite limits.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `False`: Need more paired examples before interpreting broader benchmark deltas.
- `recurrent_nonnegative_vs_base` passed `True`: Recurrent is non-negative versus base on required paired benchmark slices.

## Benchmark Evidence

- `arc_easy` paired `512` / required `1`; delta `0`; W/L/T `4/4/504`; p `1.0`
- `arc_challenge` paired `299` / required `512`; delta `0`; W/L/T `6/6/287`; p `1.0`
