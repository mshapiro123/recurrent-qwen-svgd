# Stage 5 Broader Benchmark Assessment - stage5_debiased_benchmark_assessment_20260627_154522

- Status: `inconclusive`
- Passed: `False`
- Instrument complete: `False`
- Model negative evidence: `False`
- Source summary: `outputs/stage5/stage5_debiased_benchmark_suite_20260627_152542/summary.json`
- Score target / aggregate: `cyclic_label_aggregated` / `permutation_mean`
- Next step: Complete the benchmark instrument by rerunning failed or missing slices before making a model-quality call.

## Criteria

- `suite_completed` passed `False`: Benchmark suite had failures or did not complete.
- `paired_coverage` passed `False`: Need more paired examples before interpreting broader benchmark deltas.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent has only noise-level negative deltas or missing slices.

## Benchmark Evidence

- `arc_easy` paired `128` / required `128`; delta `4`; W/L/T `5/1/122`; p `0.21875`; flagged regression `False`; negative evidence `False`
- `arc_challenge` paired `128` / required `128`; delta `-2`; W/L/T `1/3/124`; p `0.625`; flagged regression `True`; negative evidence `False`
- `gpqa_lite` paired `0` / required `16`; delta `0`; W/L/T `0/0/0`; p `None`; flagged regression `None`; negative evidence `False`
