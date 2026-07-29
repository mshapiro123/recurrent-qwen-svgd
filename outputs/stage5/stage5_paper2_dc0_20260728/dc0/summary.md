# DC0 Depth-by-Append Diagnostic

Status: `complete_diagnostic_requires_strategy_review`

No training occurred. EVAL-B is now spent for this registered comparison.

## First Transition

| Arm | Helps | Hurts | Net delta | Harm/help |
|---|---:|---:|---:|---:|
| in-place 1->2 | 7991 | 29399 | -21408 | 3.679013890626955 |
| append_raw | 6617 | 46098 | -39481 | 6.966601178781925 |
| append_rms_matched | 2458 | 119847 | -117389 | 48.75793327908869 |
| neutral_append | 2633 | 117785 | -115152 | 44.73414356247626 |
| append_read_at_t_query | 2729 | 107001 | -104272 | 39.2088677171125 |

## Execution-Path Diagnostic

Append-grid k=0 is anchored to the registered full-sequence depth-1 prediction. The incremental-cache k=0 prediction is retained as a descriptive diagnostic, not silently substituted for the registered baseline.

| Arm | Cached-vs-registered disagreements | Rate | Cached-path net | Registered-anchor net |
|---|---:|---:|---:|---:|
| append_raw | 2910 | 0.014585 | -39397 | -39481 |
| append_rms_matched | 2910 | 0.014585 | -117305 | -117389 |
| neutral_append | 2910 | 0.014585 | -115068 | -115152 |
| append_read_at_t_query | 2910 | 0.014585 | -104188 | -104272 |
