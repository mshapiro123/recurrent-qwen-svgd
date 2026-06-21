# Stage 5 ARC-AGI Candidate Gate - stage5_arc_fast_smoke5_20260621_154412

- ARC version: `1`
- Split: `evaluation`
- Limit: `5`
- Grid format: `compact`
- Selection strategy: `heuristic`
- Phase1 checkpoint: `outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt`
- Symbolic exact coverage: `0` / `5` = `0.0`
- Symbolic task solve coverage: `0` / `5` = `0.0`

## Comparison

| Variant | First | Selected | Best-of-K | Tasks best | Valid rate |
|---|---:|---:|---:|---:|---:|
| `base_model_only` | 0/5 | 0/5 | 0/5 | 0/5 | 0.0000 |
| `phase1_model_only` | 0/5 | 0/5 | 0/5 | 0/5 | 0.0000 |

## Candidate Source Summaries

### base_model_only
- `model`: count `5`, valid `0`, exact `0`, selected `5`, selected_exact `0`

### phase1_model_only
- `model`: count `5`, valid `0`, exact `0`, selected `5`, selected_exact `0`

Interpretation guide: symbolic-only tells us how much simple transform coverage exists. Hybrid-symbolic-first is not a deployable verifier; it is a value gate for whether useful non-neural candidates exist on this slice.
