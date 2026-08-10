# Strategy Handoff to Coding Agent — P3.1 Completion and P3.3 Preparation

Date: 2026-08-10. Governing: policy resolutions r2 (Drive `1-_bYB2IGvoZoIglSfS01-MfyXdKl1aOJ`), observatory adoption (Drive `1x3a1HlGBshiLZ350R-uW0iCzYGNJGY1G`), build response (Drive `1-7DU8aleh-tbkV9sR9Eh7VTERH5jBnsy`), amendment t1 (Drive `1kcSXssN6sVmQFQ2toJoMiQitMU7AcbMl`, ratified), reasoning-scope addendum (Drive `1JFUpF6wbIM_ZmrggaRp1sp_KIbONyY5_`). This is the execution order for the two deliverables that stand between the banked P3.2 receipts and the P3.3 lock. **P3.3 training remains unauthorized.** On delivery of items 1 and 2, strategy drafts the lock.

## Item 1 — P3.1 currency assembly, complete and sealed

**1a. Dataset/reader manifest.** Pin exact dataset revisions, reader configurations, and content hashes for: GSM8K (main, final-number reader) and MBPP (sanitized, unit-test execution reader) as primary targets; ARC-Challenge (Paper One reader) as secondary target; MMLU seeded slice + Paper One Tier-1 set + ARC-Easy as the floor/retention group. Battery-role enforcement per the addendum: no floor battery in any headline numerator.

**1b. DEV/CONFIRM splits and seals.** Document-stratified splits, seed 20260809. CONFIRM halves hashed and sealed under the atomic lease machinery **before any model — base, teacher, or augmented — is scored on them**. Deliverable includes the seal hashes.

**1c. Verified stratum.** Training splits of the same families with programmatic verification (gold answers; unit-test execution), document-disjoint from DEV and CONFIRM. Report counts per family and per label class (teacher-right/student-wrong, teacher-wrong, both-correct).

**1d. Sentinel panel.** 2,048 examples, stratified across the six observatory cohorts (consensus/no-op, stable missing knowledge, procedural reasoning, mixed, paired counterfactuals, paraphrase/OOD), drawn from training-split and DEV-side material only. Frozen, versioned, hashed beside the CONFIRM seals. Include the per-loop diagnostic-coda protocol stub (audit-only coda after each loop k on this panel).

**1e. Reference scores.** Frozen base and 14B teacher scored on the DEV halves with the same reader protocol. Reference table with gap_closed arithmetic, document-bootstrap CIs, raw deltas beside every ratio, and delta-only reporting where the teacher−base denominator CI touches zero.

## Item 2 — Guardrail recalibration, three-tier design

Rerun the paired sequential simulation on **empirical paired discordance and checkpoint autocorrelation from DEV** (the reference scores of 1e supply the paired rows). Report:

- **Tier S (stop):** search for Δ_cat = the smallest sustained drop detectable with power ≥ 99% within two consecutive evaluations at familywise false-stop ≤ 1e-4 over the campaign's look schedule. Report Δ_cat, the operating α, false-stop bound, and power at Δ_cat, −5, and −3.
- **Tier W (demote + review flag):** the −3-point rule at one-sided 90% upper confidence bound, two consecutive evaluations. Report expected false-demotions per campaign under the null and power at −3 and −5.
- Simulation code, seed, and the panel-size sensitivity (256/512/1,024 rows) so the lock can trade eval cost against Δ_cat if warranted.

## Item 3 — P3.3 data preparation (build, do not train)

Implement against the banked cache, per policy resolutions §3:

- Write-candidate admission at **T ≥ 0.70** on the strict concurrent stratum; report the realized count from the threshold table.
- **4,096-row held-out audit slice**, stratified by horizon and teachability decile, drawn and hashed before any training artifact exists.
- Gate negatives by confidence rank — min(student top-1 p, 14B top-1 p) over agreement rows — to a **3:1** negative:positive ratio; report the realized confidence cut and freeze it. Inverse-class weights at the realized ratio. All remaining rows labeled ignored. Position zero excluded from all classes.
- Wire the Tier-1 observatory telemetry list into the P3.3 training loop scaffolding (r^B, gradient-dot-write, tortuosity, turning angle, fixed-point residual, effective rank/participation ratio per loop) and the A_state intervention harness (zero, norm-matched random, cross-example permutation, stale state, bridge bypass; paired execution from cached pre-intervention state).

## Report-back and sequencing

Deliver items in any order, hash-ledgered as usual, with the standard evidence-record discipline and one creativity slot. Strategy responds to the report-back with the drafted P3.3 lock binding every number above to its receipt field, the two-seed initialization (E1-confirmation checkpoints, hash-pinned), the migration equivalence tests as pre-run assertions, and the calibrated directional share contract. Training begins only after Mark ratifies that lock.

## Plain-language summary

Three jobs before the big experiment: finish building the test — the question sets, their sealed final-exam copies, and the fixed panel of examples we will watch closely all campaign; recalibrate the safety alarms on real noise measurements so the numbers in the plan are honest; and stage the training data at the agreed quality cuts, with the measurement plumbing installed. No training yet. When these come back, we write the binding run plan, Mark signs it, and the experiment that tells us whether trained aim works finally goes.