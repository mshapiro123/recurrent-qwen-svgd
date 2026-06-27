# Stage 5 Capacity Localization - stage5_capacity_localization_20260627_210858

- Baseline rank: `32`
- Target ranks: `128`
- Status: `completed`
- Decision: `capacity_signal_mixed_or_negative`

## Capacity Ledger

| rank | alpha | LoRA trainable M | stored params M | loop1 | loop2 | loop3 | oracle | rescued | harmed | loop8 tail ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 9.388 | 494.0 | 182 | 163 | 159 | 234 | 52 | 71 | 2.8300413441051058 |
| 64 | 128 | 18.776 | 494.0 | 182 | 157 | 158 | 235 | 53 | 74 | 2.732335107047776 |
| 128 | 256 | 37.552 | 494.0 | 182 | 158 | 159 | 235 | 53 | 73 | 2.7015996666280033 |

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
  },
  "128": {
    "oracle_correct": 1,
    "oracle_gap_vs_loop1": 1,
    "rescued_vs_loop1": 1,
    "harmed_vs_loop1": 2,
    "stable_correct": -2,
    "stable_wrong": -1,
    "loop1_correct": 0,
    "loop2_correct": -5,
    "loop3_correct": 0
  }
}
```

## Decision
```json
{
  "status": "capacity_signal_mixed_or_negative",
  "best_oracle_rank": 64,
  "best_oracle_correct": 235,
  "best_loop1_rank": 64,
  "best_loop1_correct": 182,
  "any_depth_loop_beats_loop1": false,
  "any_oracle_count_improves_vs_rank32": true,
  "any_rescue_count_improves_vs_rank32": true,
  "any_harm_count_worsens_vs_rank32": true,
  "any_loop2_or_loop3_regresses_vs_rank32": true,
  "recommendation": "Higher LoRA rank produced only mixed/noisy oracle or rescue movement while harm/deeper-loop regression remained; stop rank-only escalation and review the unfreeze+Muon bundle."
}
```

Rank128 completes the clean rank-only capacity sweep. If depth still does not beat loop1 and rescued gains come with added harm, stop rank-only escalation. The next experimental fork is either close this line or run the unfreeze+Muon recurrence-curriculum bundle as a deliberately larger-capacity substrate test.
