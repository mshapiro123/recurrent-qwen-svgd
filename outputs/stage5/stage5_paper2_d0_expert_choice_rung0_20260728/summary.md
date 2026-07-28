# D0 Expert-Choice Rung 0

CPU-only re-scoring of the frozen out-of-fold scalar probe. No features, folds, or model weights changed.

- OOF help-vs-hurt AUC: `0.7240`
- Local verdict: `all_local_budgets_negative`
- Banked curve replay: `numerically_equivalent_not_bit_exact`
- Pre-D0 harm/help ratio: `0.0807386440221054`
- Post-D0 harm/help ratio: `3.5039701074264364`

## Causal Local Windows

| Window | Budget | Helps | Hurts | Net delta | Mean loops |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.5% | 244 | 515 | -271 | 1.0171 |
| 256 | 1.0% | 306 | 596 | -290 | 1.0204 |
| 256 | 2.0% | 486 | 907 | -421 | 1.0299 |
| 256 | 5.0% | 988 | 1813 | -825 | 1.0556 |
| 256 | 10.0% | 2003 | 3610 | -1607 | 1.1021 |
| 256 | 20.0% | 3744 | 7357 | -3613 | 1.1930 |
| 256 | 27.0% | 4818 | 10185 | -5367 | 1.2589 |
| 1024 | 0.5% | 222 | 477 | -255 | 1.0159 |
| 1024 | 1.0% | 286 | 571 | -285 | 1.0195 |
| 1024 | 2.0% | 432 | 820 | -388 | 1.0277 |
| 1024 | 5.0% | 954 | 1758 | -804 | 1.0536 |
| 1024 | 10.0% | 1926 | 3451 | -1525 | 1.0983 |
| 1024 | 20.0% | 3630 | 7031 | -3401 | 1.1855 |
| 1024 | 27.0% | 4738 | 9820 | -5082 | 1.2503 |
