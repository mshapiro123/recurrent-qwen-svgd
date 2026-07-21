# Paper One Frontier Threshold Sensitivity

The registered `0.71` bar was set during synthetic-task design. Four-option chance is `0.25`; the per-step accuracy whose four-step product equals chance is `0.25^(1/4) = 0.707107`, rounded to `0.71`. This was a fixed design heuristic, not a confidence bound.

| Support | Alphabet | Frontier at 0.60 | Ratio | Frontier at 0.71 | Ratio | Frontier at 0.80 | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | N16 | 6.151 | 1.538 | 5.746 | 1.436 | 5.374 | 1.344 |
| 6 | N16 | 9.645 | 1.608 | 9.005 | 1.501 | 8.240 | 1.373 |
| 8 | N16 | 12.340 | 1.542 | 11.612 | 1.452 | 10.662 | 1.333 |
| 12 | N24 | 19.226 | 1.602 | 17.932 | 1.494 | 17.046 | 1.421 |

| Bar | Mean ratio | Min | Max | Range | CV |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 1.573 | 1.538 | 1.608 | 0.070 | 2.06% |
| 0.71 | 1.471 | 1.436 | 1.501 | 0.064 | 1.87% |
| 0.80 | 1.368 | 1.333 | 1.421 | 0.088 | 2.49% |

**Reading.** The qualitative law is insensitive to the bar: frontier remains approximately proportional to supervised support at all three levels. The numerical ratio is not invariant to the bar and decreases as the criterion becomes stricter.

**Arithmetic correction.** The N24 frontier at the registered bar is `17.932308`, which rounds to `17.93`, not `17.92`.

## Paste-Ready Paper Language

> The 0.71 bar was fixed during task design, motivated by the rounded four-step root of four-choice chance, 0.25^(1/4) = 0.707. As a sensitivity check, we recomputed all four frontiers at bars of 0.60 and 0.80. The frontier/support ratios remained tightly grouped within each bar (1.54-1.61 at 0.60, 1.44-1.50 at 0.71, and 1.33-1.42 at 0.80). Thus the near-proportional scaling conclusion is insensitive to the bar's level, although the numerical ratio decreases as the criterion is raised.
