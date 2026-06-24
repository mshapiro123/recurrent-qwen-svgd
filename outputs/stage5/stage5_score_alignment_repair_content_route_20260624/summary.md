# Stage 5 Surface-Alignment Repair - stage5_score_alignment_repair_content_route_20260624

- Status: `surface_alignment_not_passed`
- Source summary: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Benchmark source: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`
- Order-sensitivity diagnosis: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/arc_easy_order_sensitivity_diagnosis.json`
- Diagnosis: `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/arc_easy_surface_mismatch_diagnosis.json`
- Repair objective: `content_cyclic_surface_alignment`
- Train rows: `48`
- Checkpoint: `outputs/stage5/stage5_score_alignment_repair_content_route_20260624/phase1_surface_align/phase1_step_75.pt`
- Assessment: `outputs/stage5/stage5_score_alignment_repair_content_route_20260624_assessment/summary.json`
- Surface repair assessment: `outputs/stage5/stage5_score_alignment_repair_content_route_20260624/surface_repair_assessment.json`
- Next step: Inspect content/cyclic deltas; if cyclic remains strong but content lags, add explicit score-level alignment.
