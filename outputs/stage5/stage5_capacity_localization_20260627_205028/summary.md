# Stage 5 Capacity Localization - stage5_capacity_localization_20260627_205028

- Baseline rank: `32`
- Target ranks: `64`
- Status: `completed`
- Decision: `capacity_signal_present`

## Capacity Ledger

| rank | alpha | LoRA trainable M | stored params M | loop1 | loop2 | loop3 | oracle | rescued | harmed | loop8 tail ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 9.388 | 494.0 | 182 | 163 | 159 | 234 | 52 | 71 | 2.8300413441051058 |
| 64 | 128 | 18.776 | 494.0 | 182 | 157 | 158 | 235 | 53 | 74 | 2.732335107047776 |

## Deltas Vs Rank 32

```json
{
  "32": {
    "oracle_correct": 0,
    "oracle_gap_vs_loop1": 0,
    "rescued_vs_loop1": 0,
    "harmed_vs_loop1": 0,
    "stable_correct": 0,
    "stable_wrong": 0,
    "loop1_correct": 0,
    "loop2_correct": 0,
    "loop3_correct": 0
  },
  "64": {
    "oracle_correct": 1,
    "oracle_gap_vs_loop1": 1,
    "rescued_vs_loop1": 1,
    "harmed_vs_loop1": 3,
    "stable_correct": -3,
    "stable_wrong": -1,
    "loop1_correct": 0,
    "loop2_correct": -6,
    "loop3_correct": -1
  }
}
```

## Decision
```json
{
  "status": "capacity_signal_present",
  "best_oracle_rank": 64,
  "best_oracle_correct": 235,
  "best_loop1_rank": 64,
  "best_loop1_correct": 182,
  "any_depth_loop_beats_loop1": false,
  "any_rescue_count_improves_vs_rank32": true,
  "recommendation": "If rank64 improves rescued/oracle or produces loop benefit, run rank128 before unfreezing."
}
```

Review rank-localization deltas before escalating. Rank128 is justified only if rank64 shows rescued/oracle/depth movement; otherwise the next meaningful test is the unfreeze+Muon bundle.
