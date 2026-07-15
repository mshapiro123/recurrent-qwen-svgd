# Phase G-alpha Margin-Lock Inputs

Source: `outputs/stage5/stage5_part1_closeout_pivot_20260715/branching_screen/natural_step2000_N20_verbal/rows.jsonl`

## Overall

| Rows | Valid | Invalid | Validity | Mean reachable set | Mean score entropy (nats) | Mean top-1 probability | Modal prediction rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 389 | 123 | 0.7598 | 3.709 | 0.1432 | 0.9471 | 0.0703 |

## Validity by depth and stratum

| Depth | Stratum | Valid | Rows | Validity |
|---:|:---|---:|---:|---:|
| 1 | 2 | 127 | 128 | 0.9922 |
| 2 | 2 | 41 | 64 | 0.6406 |
| 2 | 3-4 | 54 | 64 | 0.8438 |
| 3 | 2 | 19 | 43 | 0.4419 |
| 3 | 3-4 | 35 | 43 | 0.8140 |
| 3 | 5-8 | 33 | 42 | 0.7857 |
| 4 | 2 | 11 | 32 | 0.3438 |
| 4 | 3-4 | 15 | 32 | 0.4688 |
| 4 | 5-8 | 24 | 32 | 0.7500 |
| 4 | 9-16 | 30 | 32 | 0.9375 |

## Reachable-set-size distribution

| Set size | Rows | Fraction |
|---:|---:|---:|
| 2 | 267 | 0.5215 |
| 3 | 70 | 0.1367 |
| 4 | 69 | 0.1348 |
| 5 | 19 | 0.0371 |
| 6 | 19 | 0.0371 |
| 7 | 18 | 0.0352 |
| 8 | 18 | 0.0352 |
| 9 | 4 | 0.0078 |
| 10 | 4 | 0.0078 |
| 11 | 4 | 0.0078 |
| 12 | 4 | 0.0078 |
| 13 | 4 | 0.0078 |
| 14 | 4 | 0.0078 |
| 15 | 4 | 0.0078 |
| 16 | 4 | 0.0078 |

## Deterministic collapse profile by stratum

| Stratum | Rows | Score entropy (nats) | Normalized entropy | Top-1 probability | Modal symbol | Modal rate | Empirical answer entropy (nats) |
|:---|---:|---:|---:|---:|:---|---:|---:|
| 2 | 267 | 0.1540 | 0.0514 | 0.9433 | Tom | 0.0749 | 2.9508 |
| 3-4 | 139 | 0.1310 | 0.0437 | 0.9504 | Una | 0.1151 | 2.9088 |
| 5-8 | 74 | 0.1252 | 0.0418 | 0.9552 | Joe | 0.1486 | 2.7358 |
| 9-16 | 32 | 0.1476 | 0.0493 | 0.9455 | Jan | 0.1250 | 2.6205 |

## Interpretation boundary

These statistics characterize the frozen deterministic keeper and set the entropy-matching target for Phase G-alpha comparators. They do not measure stochastic coverage and do not authorize a G-alpha launch without a separately locked powered margin.
