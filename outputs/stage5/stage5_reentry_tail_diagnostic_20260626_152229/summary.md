# Tail-Resolved Re-entry Diagnostic - 

## Decision
- Correction class: `tail_damper`
- Reasons: `per_axis_tail_damping_removes_most_mismatch`

## Tail Decomposition
- Tail mismatch: `2.110969`
- After damper: `0.425675`
- After rotation: `1.973531`
- After rotation then damper: `0.362344`
- Exit/entry diagonal ratios: `[2.2298899176230913, 3.266398158631298, 2.609744603418381, 2.2369012654458236, 3.8227508869790343, 2.0197556542205164, 1.914434041619949]`

## Loop Tail Trace
| stage | tail trace | ratio vs entry |
|---|---:|---:|
| entry | 26.111161 | 1.000000 |
| loop1 | 68.150388 | 2.610010 |
| loop2 | 134.162458 | 5.138127 |
| loop3 | 219.367949 | 8.401310 |
| loop4 | 321.863305 | 12.326656 |
| loop8 | 891.542817 | 34.144127 |

## Harmed vs Rescued
- `harmed`: n=31, mean tipping tail ratio=14.785237107746974
- `rescued`: n=28, mean tipping tail ratio=16.57963746876615
- `stable_correct`: n=58, mean tipping tail ratio=0.0
- `stable_wrong`: n=139, mean tipping tail ratio=0.0
- Harmed minus rescued tipping ratio delta: `-1.7944003610191768`
