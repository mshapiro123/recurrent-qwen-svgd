# Stage 5 Broader Benchmark Assessment - stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment

- Status: `needs_recurrent_recovery`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Score target / aggregate: `content_question_only` / `mean`
- Next step: Return to deterministic recurrent recovery before GPQA Diamond or release claims.

## Criteria

- `suite_completed` passed `True`: Benchmark suite completed without failures.
- `paired_coverage` passed `True`: All benchmark slices have enough paired base-vs-recurrent examples.
- `recurrent_nonnegative_vs_base` passed `False`: Recurrent trails base on at least one paired benchmark slice.

## Benchmark Evidence

- `arc_easy` paired `256` / required `1`; delta `-8`; W/L/T `8/16/232`; p `0.15158963203430176`
- `arc_challenge` paired `256` / required `128`; delta `-1`; W/L/T `7/8/241`; p `1.0`
