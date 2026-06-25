# Stage 5 Broader Benchmark Assessment - stage5_debiased_benchmark_assessment_20260625_184717

- Status: `needs_benchmark_confirmation`
- Passed: `False`
- Instrument complete: `False`
- Model negative evidence: `True`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260625_180322/summary.json`
- Score target / aggregate: `cyclic_label_aggregated` / `permutation_mean`
- Next step: Rerun the broader benchmark suite with enough paired examples before making a claim.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `False`: Need more paired examples before interpreting broader benchmark deltas.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent has statistically meaningful negative evidence versus base.

## Benchmark Evidence

- `arc_easy` paired `128` / required `256`; delta `2`; W/L/T `3/1/124`; p `0.625`; negative evidence `False`
- `arc_challenge` paired `256` / required `256`; delta `-5`; W/L/T `4/9/243`; p `0.266845703125`; negative evidence `True`
- `open_hard_arc_challenge` paired `256` / required `256`; delta `-5`; W/L/T `4/9/243`; p `0.266845703125`; negative evidence `True`
