# Stage 5 Surface Repair Assessment - stage5_surface_repair_assessment_20260624_011139

- Status: `surface_repair_no_easy_content_lift`
- Passed: `False`
- Source benchmark: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Repaired benchmark: `outputs/stage5/stage5_surface_alignment_repair_content_cyclic_20260624_010142_benchmark/summary.json`
- Reason: Repair did not improve ARC-Easy content versus the source recurrent checkpoint.
- Next step: Inspect training rows and consider score-level alignment rather than more SFT on the same shard.

## Decision Evidence

- `arc_easy_content`: repair_delta 0 (2/2/252 W/L/T, n=256, p=1.0), repaired_vs_base_delta -8
- `arc_easy_cyclic`: repair_delta 3 (3/0/253 W/L/T, n=256, p=0.25), repaired_vs_base_delta 5
- `arc_challenge_content`: repair_delta 4 (6/2/248 W/L/T, n=256, p=0.2890625), repaired_vs_base_delta 3
- `arc_challenge_cyclic`: repair_delta 0 (2/2/252 W/L/T, n=256, p=1.0), repaired_vs_base_delta -5

## Order-Sensitivity Repair

- Status: `order_sensitivity_reduced`
- Improved: `True`
- Candidate order-sensitive row delta: `-2.0`
- Order-sensitive content-loss delta: `-1.0`
- Total content-loss delta: `-2.0`
