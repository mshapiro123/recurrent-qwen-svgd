# Stage 5 Broader Benchmark Assessment - stage5_score_alignment_repair_content_route_20260624_assessment

- Status: `needs_recurrent_recovery`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_score_alignment_repair_content_route_20260624_benchmark/summary.json`
- Score target / aggregate: `content_question_only` / `mean`
- Next step: Return to deterministic recurrent recovery before GPQA Diamond or release claims.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `True`: All benchmark slices have enough paired base-vs-recurrent examples.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_easy` paired `256` / required `1`; delta `-7`; W/L/T `6/13/237`; p `0.1670684814453125`
- `arc_challenge` paired `256` / required `128`; delta `4`; W/L/T `9/5/242`; p `0.4239501953125`
