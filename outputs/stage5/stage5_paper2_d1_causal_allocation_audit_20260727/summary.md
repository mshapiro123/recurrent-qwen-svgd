# D0 Causal Allocation Audit and D1 Label Construction

- Status: `complete`
- Checkpoint: `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`
- Evaluation positions: 199,529
- Label-train dry-run positions: 100,000
- Training or optimizer steps: 0

## Transition outcomes on evaluation

| Transition | Helps | Hurts | Neutral |
|---|---:|---:|---:|
| 1 to 2 | 8,564 | 30,008 | 160,957 |
| 2 to 3 | 2,771 | 35,677 | 161,081 |
| 3 to 4 | 1,554 | 26,498 | 171,477 |

## Accepted-position policy damage

- Total baseline-accepted losses: 5,928
- Already lost at loop 1 after training: 974
- Post-loop policy losses: 4,954
- Preventable by stopping on non-help labels: 83.6%

## Scope

This is a post-hoc read-only audit. It cannot alter D0's registered verdict and does not authorize D1 training.
