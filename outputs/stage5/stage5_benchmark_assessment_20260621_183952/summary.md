# Stage 5 Broader Benchmark Assessment - stage5_benchmark_assessment_20260621_183952

- Status: `needs_recurrent_recovery`
- Passed: `False`
- Source summary: `outputs\stage5\stage5_recovery_full_assessment_current_balanced_full\summary.json`
- Score target / aggregate: `label` / `mean`
- Next step: Return to deterministic recurrent recovery before GPQA Diamond or release claims.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `True`: All benchmark slices have enough paired base-vs-recurrent examples.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_easy` paired `570` / required `1`; delta `-9`; W/L/T `20/29/521`; p `0.2528697301676033`
- `arc_challenge` paired `299` / required `128`; delta `2`; W/L/T `13/11/275`; p `0.8388197422027588`
