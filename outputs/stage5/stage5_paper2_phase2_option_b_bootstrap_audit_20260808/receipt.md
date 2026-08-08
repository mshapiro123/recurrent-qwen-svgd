# Option B Document-Bootstrap Audit

Read-only completion of the confidence intervals required by the locked Option B protocol.
No model was loaded and no optimizer update occurred.

## Corrected reading

- Interpretation: `curve_supports_E1_recipe_transfer`
- E1 support in both seeds: `true`
- Retain writeback for E1: `true`

## Seed-level estimates

All cells are estimate [document-bootstrap 95% CI].

| Seed | Full dose slope | Full fresh slope | Full late slope | Gap growth 0-20k | Endpoint relative gain | Corrected E1 support |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.000632 [0.000106, 0.001138] | 0.003583 [0.002778, 0.004438] | 0.000228 [0.000104, 0.000351] | 0.005106 [0.003186, 0.007201] | 0.003507 [0.002398, 0.004768] | true |
| 1 | 0.001466 [0.000962, 0.001971] | 0.002203 [0.001553, 0.002834] | 0.000152 [0.000041, 0.000264] | 0.005603 [0.003636, 0.007903] | 0.004960 [0.003700, 0.006553] | true |

## Estimator correction

The landed matrix labeled a positive late-slope point estimate as E1 support.
The governing protocol additionally requires the document-bootstrap 95% interval
to exclude zero. This receipt applies that requirement without changing the source
matrix, thresholds, rows, or training lineage.
