# Stage 5 ARC-AGI Progress Ledger - stage5_arc_agi_progress_20260621_233156

- Scan root: `outputs/stage5`
- Scanned files: `61`
- Parsed records: `14`
- Recommended next-plan source: `outputs/stage5/stage5_release_gate_20260621_233156/summary.json`

## Best By Arm

| Arm | Selected | Best-of-K | Examples | Label | Source |
|---|---:|---:|---:|---|---|
| `base` | 0 | 0 | 20 | `base_hybrid_program_first` | `outputs/stage5/stage5_claim_continue_lastlogits_20260621_150552_candidate_gate/base_hybrid_program_first_summary.json` |
| `phase1_start` | 0 | 0 | 5 | `phase1_model_only` | `outputs/stage5/stage5_arc_fast_smoke5_20260621_154412/phase1_model_only_summary.json` |
| `unknown` | 0 | 0 | 20 | `symbolic_only` | `outputs/stage5/stage5_claim_continue_lastlogits_20260621_150552_candidate_gate/symbolic_only_summary.json` |

## Recovered vs Base Gaps

- No complete recovered-vs-base benchmark summaries found.

## Gate 1 Assessments

- No Gate 1 assessment summaries found.

## Gate 2 Assessments

- No Gate 2 assessment summaries found.

## Selector Replication Gates

- No selector replication gates found.

## Same-Recipe Selector Conversion Gates

- No same-recipe selector conversion gates found.

## Same-Recipe Architecture Assessments

- No same-recipe architecture assessment summaries found.

## Release / Benchmark Gates

- `stage5_release_gate_20260621_233156` status `needs_benchmark_confirmation` passed `False` min ARC examples `100` failed criteria `arc_benchmark_confirmation, same_recipe_architecture_or_selector_conversion, hf_export_artifact`: Run or replicate recovered-vs-base ARC-AGI benchmark on a larger held-out slice.

## Broader Benchmark Suites

- `stage5_phase1_best_arceasy_full_20260621_184638` status `completed` checkpoint `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt` deltas: arc_easy/label/mean: -12 (W/L/T 15/27/528, p 0.08842954698775429)
- `stage5_phase1_best_gpqa16_20260621_184030` status `completed_with_failures` checkpoint `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt` deltas: none
- `stage5_phase1_step150_arcchallenge_full_20260621_194028` status `completed` checkpoint `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt` deltas: arc_challenge/label/mean: +2 (W/L/T 24/22/253, p 0.8829959121223965)
- `stage5_recovered_phase1_arc256_20260621_172908` status `completed` checkpoint `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt` deltas: arc_challenge/label/mean: -2 (W/L/T 24/26/206, p 0.887724827340783)
- `stage5_recovered_phase1_arcfull_20260621_173349` status `completed` checkpoint `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt` deltas: arc_challenge/label/mean: +1 (W/L/T 29/28/242, p 1.0)
- `stage5_recovered_phase1_particles_arc_20260621_174231_rep0_k4_arc256` status `completed` checkpoint `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt` deltas: arc_challenge/label/max: -9 (W/L/T 13/22/221, p 0.17546524899080396); arc_challenge/label/mean: -12 (W/L/T 12/24/220, p 0.06524533522315323); arc_challenge/label/vote: -13 (W/L/T 12/25/219, p 0.04703102743951604)
- `stage5_recovered_phase1_particles_arc_20260621_174231_rep2_k4_arc256` status `completed` checkpoint `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt` deltas: arc_challenge/label/max: -8 (W/L/T 12/20/224, p 0.21532714972272515); arc_challenge/label/mean: -4 (W/L/T 12/16/228, p 0.5715881884098053); arc_challenge/label/vote: -4 (W/L/T 13/17/226, p 0.584664711728692)
- `stage5_recovered_phase2_smoke_20260621_175538_arc128` status `completed` checkpoint `outputs/stage5/stage5_recovered_phase2_smoke_20260621_175538/phase2/phase2_step_25.pt` deltas: arc_challenge/label/max: -4 (W/L/T 7/11/110, p 0.480682373046875); arc_challenge/label/mean: -2 (W/L/T 7/9/112, p 0.803619384765625); arc_challenge/label/vote: -5 (W/L/T 6/11/111, p 0.332305908203125)
- `stage5_recovered_phase2_smoke_20260621_180336_arc128` status `completed` checkpoint `outputs/stage5/stage5_recovered_phase2_smoke_20260621_180336/phase2/phase2_step_50.pt` deltas: arc_challenge/label/max: -6 (W/L/T 5/11/112, p 0.210113525390625); arc_challenge/label/mean: -3 (W/L/T 6/9/113, p 0.60723876953125); arc_challenge/label/vote: -4 (W/L/T 6/10/112, p 0.454498291015625)
- `stage5_recovery_full_assessment_current_balanced_full` status `completed` checkpoint `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt` deltas: arc_challenge/label/mean: +2 (W/L/T 13/11/275, p 0.8388197422027588); arc_easy/label/mean: -9 (W/L/T 20/29/521, p 0.2528697301676033)

## Broader Benchmark Gates

- `stage5_benchmark_assessment_20260621_183952` status `needs_recurrent_recovery` passed `False` source `outputs\stage5\stage5_recovery_full_assessment_current_balanced_full\summary.json`: Return to deterministic recurrent recovery before GPQA Diamond or release claims.

## Claim Readiness Packets

- No claim-readiness packets found.

## ARC-AGI Baseline Registries

- No ARC-AGI baseline registry validation artifacts found.

## ARC-AGI SOTA Comparisons

- No ARC-AGI SOTA comparison artifacts found.

## ARC-AGI Candidate Gates

- `stage5_arc_fast_smoke5_20260621_154412` ARC `1` split `evaluation` limit `5` symbolic exact `0/5`, phase1 hybrid best delta `0`, base hybrid best delta `0`

## ARC-AGI SFT Recipe Gates

- No ARC-AGI SFT recipe gate artifacts found.
