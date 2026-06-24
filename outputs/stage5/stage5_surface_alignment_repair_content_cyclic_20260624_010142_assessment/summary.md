# Stage 5 Broader Benchmark Assessment - stage5_surface_alignment_repair_content_cyclic_20260624_010142_assessment

- Status: `needs_recurrent_recovery`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_surface_alignment_repair_content_cyclic_20260624_010142_benchmark/summary.json`
- Score target / aggregate: `content_question_only` / `mean`
- Next step: Return to deterministic recurrent recovery before GPQA Diamond or release claims.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `True`: All benchmark slices have enough paired base-vs-recurrent examples.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_easy` paired `256` / required `1`; delta `-8`; W/L/T `6/14/236`; p `0.11531829833984375`
- `arc_challenge` paired `256` / required `128`; delta `3`; W/L/T `9/6/241`; p `0.60723876953125`
