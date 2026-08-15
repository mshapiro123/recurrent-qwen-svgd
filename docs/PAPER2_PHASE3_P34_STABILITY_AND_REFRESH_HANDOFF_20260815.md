# Paper Two Handoff: P3.4 Directional Stability, Fixed-Ceiling Probe, and Oracle Refresh

**Date:** 2026-08-15
**Status:** Complete; CPU stability audit, paired fixed-ceiling probe, and L4 oracle refresh banked
**Registered P3.4 verdict:** `REPLICATED_POSITIVE_BELOW_TRIGGER_B` (unchanged)
**Sealed evaluations:** CONFIRM and EVAL-E remain untouched

## 1. Direct answer

The P3.4 instability is not well described as one implementation fault. The strongest measured explanation is a knife-edge evaluation surface created by the mechanism's intended operating regime: approximately 95 rows per seed are discordant, but the endpoint gain is the small difference between about 50 fixes and 45 regressions. On the exact adjacent row receipts that survived, changed outcomes had an average option-score margin of 0.0078, versus 1.812 for stable outcomes. GSM8K changed most often. This is direct evidence that small parameter movement can change discrete task accuracy without requiring a collapse of the underlying mechanism.

The second likely source is endpoint arbitrariness. Training used a constant learning rate through step 4,000, without a terminal decay or a weight-averaged primary checkpoint. The raw endpoint therefore samples the late trajectory rather than deliberately landing it. Score-space averaging puts the joint late read near +8 rows, but this is not a weight-EMA counterfactual and does not meet Trigger B.

One initial explanation is narrowed by the receipts: evaluation ceilings changed across seeds, but not between score looks within either seed. Seed 0 was always read at 0.08 and seed 1 at 0.02. Controller transitions can still alter the parameter path, but a changing ruler cannot explain within-seed oscillation. The authorized paired fixed-ceiling probe measures the cross-seed exposure directly.

The registered verdict does not change. These are post-hoc diagnostic reads over DEV and cannot promote the result or spend the sealed exam.

## 2. Experimental design

### 2.1 CPU directional-stability audit

The audit used the two completed main-arm P3.4 A2 runs. It replayed all 4,000 registered batch hashes for each seed, recovered every 200-step curriculum segment, joined all 20 task looks to the controller and objective-share telemetry, and used the retained row-level receipts where exact identity survived.

Four candidate sources were examined:

1. **Evaluation surface:** fix/regression counts, discordant volume, exact adjacent-row churn, option-score margins, and battery localization.
2. **Curriculum:** realized depth and workload mix in each 200-step segment, verified against the saved schedule hashes.
3. **Controller:** scored ceiling, training-time rung transitions, share demotions, and their timing relative to score changes.
4. **Optimizer/landing:** late-window score means under the actual constant-learning-rate trajectory.

The pooled regressions and Shapley decompositions are descriptive diagnostics, not causal variance partitions. There was no randomized intervention over learning-rate schedule, averaging policy, or controller state.

### 2.2 Paired fixed-ceiling score probe

The two scored step-4,000 checkpoints are evaluated on the same frozen 1,024-row DEV panel under ceilings 0.02 and 0.08, producing a 2 seed by 2 ceiling matrix. The probe is evaluation-only, checkpoint-selection-barred, and must exactly reproduce each registered endpoint condition before the counterfactual cell is accepted.

The receipt reports pooled and per-battery fixes, regressions, net correct, paired row transitions between ceilings, and the seed-1 floor/target reconstruction. It cannot touch CONFIRM or EVAL-E and constructs no optimizer.

### 2.3 Oracle-direction refresh

The endpoint audit distinguishes two notions that had been conflated:

- **Registered estimator direction:** the analytic LM-head direction from the frozen base token to the cached 14B target. If the frozen base reader still reproduces the cached source token, this direction is not stale.
- **Persistent-serving direction:** after a deployed intervention changes the current token, another recurrent or cross-token write may need a direction re-anchored from that new source token.

Both endpoints are reread with the existing 4,096 positive and 12,288 negative audit populations. The receipt repeats pi_dir and pi_dep, verifies base-token identity, measures how often deployed writes change the source token, and compares the old direction with a deployed-token-anchored direction where the target has not already been reached.

## 3. Results: directional stability

### 3.1 Endpoint arithmetic

| Seed | Scored ceiling | Fixes | Regressions | Net | Discordant | Net / discordant |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08 | 50 | 45 | +5 | 95 | 5.3% |
| 1 | 0.02 | 54 | 44 | +10 | 98 | 10.2% |

The result is therefore intrinsically sensitive to a small number of boundary rows. This does not make the result unreal, but it makes a raw endpoint a poor summary of the late trajectory.

### 3.2 Knife-edge evidence

Only five exact adjacent look-pairs retained complete row identity: seed 0 looks 1-2 and seed 1 looks 1-5. Across those pairs:

- 42 row outcomes changed.
- Changed rows had mean option-score margin 0.00781.
- Stable rows had mean option-score margin 1.81153.
- GSM8K had the highest pooled change rate, 1.14%, followed by MMLU at 0.90%.
- MBPP changed at 0.60%; Tier-1 did not change.
- The available fix sets were not wholesale replacements. Seed 0's fix-set Jaccard was 0.767; seed 1's adjacent values were 0.887-0.925.

The evidence supports a small marginal subset churning, not a globally unstable mechanism. It also supports the proposed sequence-level persistence probe because churn concentrates most in multi-step GSM8K, although the retained identity rows are early rather than late and cannot settle that architectural question alone.

### 3.3 Curriculum lottery

All 8,000 batch hashes reproduced exactly. Across 200-step segments, realized mean depth ranged from 2.77 to 3.13 around the registered expectation of 3.0. Signed score movement versus mean depth had Pearson r = 0.060 (p = 0.721); versus code fraction, r = 0.122 (p = 0.466). The descriptive incremental R-squared assigned only 0.003 of absolute movement to curriculum features.

This does not prove the curriculum has zero effect. It does justify ranking it below the evaluation surface and landing policy for the next intervention.

### 3.4 Controller

Every seed-0 score read used ceiling 0.08. Every seed-1 score read used ceiling 0.02. Seed 1 advanced after looks 10, 15, and 20, but the objective-share controller demoted it before the following score read at looks 11 and 16. Thus:

- The cross-seed endpoint comparison used different effective inference systems.
- Within-seed score oscillation was not caused by a changing evaluation ceiling.
- Training-time controller changes remain a possible parameter-path effect.
- Intervals containing a training-time transition had mean absolute movement 2.33 rows, versus 2.97 for stable intervals. That does not support the controller as the main swing source in these data.

### 3.5 Landing behavior

The joint mean endpoint was +7.5 rows. Joint score means over the final 3, 5, and 10 looks were +8.67, +7.8, and +6.95 rows. This supports adding an explicit landing protocol, but it does not show that weight EMA would have produced +10. An EMA checkpoint was not saved and cannot be reconstructed from score counts.

## 4. Paired-ceiling results

The paired probe reproduced both registered endpoints exactly and completed with zero optimizer steps:

| Seed | Fixed ceiling | Correct | Fixes | Regressions | Net vs base | Floor net | Target net |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02 | 505/1,024 | 48 | 45 | +3 | 0 | +3 |
| 0 | 0.08 | 507/1,024 | 50 | 45 | +5 | -1 | +6 |
| 1 | 0.02 | 512/1,024 | 54 | 44 | +10 | +5 | +5 |
| 1 | 0.08 | 505/1,024 | 46 | 43 | +3 | -2 | +5 |

Seed 0 gained two net rows when the ceiling increased from 0.02 to 0.08: 30 rows changed, with 16 gains and 14 losses. Seed 1 lost seven net rows: 49 rows changed, with 21 gains and 28 losses. The mean net effect was +6.5 rows at 0.02 and +4.0 at 0.08, but this diagnostic DEV reuse is checkpoint-selection-barred and cannot itself select 0.02 for P3.5.

There is no monotonic wider-is-better response. In seed 1, widening the ceiling left the target-half net unchanged at +5 but changed the floor from +5 to -2. Across both seeds, 79 row outcomes changed between ceilings even though the mean score moved by only 2.5 rows. This is direct evidence that ceiling choice changes which boundary rows are fixed and regressed. One pinned ceiling is therefore mandatory for registered cross-seed reads.

The battery decomposition again localizes the largest bidirectional churn to GSM8K. MBPP retained a positive net in all four cells, while Tier-1 remained unchanged at 20/25 in every cell. The paired matrix does not identify a universally dominant ceiling; the response is seed-specific.

## 5. Oracle-refresh results

The L4 audit re-read the full 4,096-row positive population and the disjoint 12,288-row negative population for both step-4,000 endpoints. It constructed no optimizer, took zero training steps, and did not score CONFIRM or EVAL-E.

### 5.1 Refreshed audit values

| Seed | Registered ceiling | pi_dir | 95% CI | pi_dep | 95% CI | Collateral chi |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08 | 14.75% | 13.09-16.38% | 15.70% | 13.50-17.89% | 0.0% |
| 1 | 0.02 | 15.50% | 13.85-17.14% | 27.59% | 22.87-32.71% | 0.0% |

The all-row values reproduce the P3.4 endpoint audit to rounding. They do not repair the estimator issue below and therefore remain descriptive repeats of the registered DEV reading.

### 5.2 Frozen-anchor mismatch

The BF16 source-token reader reproduced the cached frozen-base source token on `3,943/4,096 = 96.26%` of positive rows, leaving 153 mismatches in each seed. This fails the audit's required identity condition. On the matched subset only, pi_dir was 12.70% for seed 0 and 13.32% for seed 1; pi_dep was 11.51% and 12.31%.

The mismatch does not show that the mechanism disappeared. It shows that the frozen oracle cache and the deployed BF16 reader are not the same estimator on every row. Future causal gating must rebuild or canonicalize the source anchor under one pinned serving reader before it can use pi_dir as a registered decision statistic. The 153 rows may not be silently dropped to recover a cleaner number.

### 5.3 Deployed-token re-anchoring

| Seed | Deployed source changed | Target already reached | Mean old/new cosine | Fraction cosine < 0.99 |
|---:|---:|---:|---:|---:|
| 0 | 336/4,096 (8.20%) | 174/4,096 (4.25%) | 0.9708 | 5.51% |
| 1 | 99/4,096 (2.42%) | 88/4,096 (2.15%) | 0.9911 | 1.67% |

The old-versus-deployed-anchor comparison excludes rows where the deployed token already equals the target. Its minimum cosine was zero in both seeds. Therefore, once a deployed write changes the source token, blindly reusing the frozen-source direction is not a valid persistent-serving contract. A persistent or cross-token mechanism must re-anchor its direction from the current source token, or demonstrate an equivalent invariant construction.

This result promotes persistence to a justified architectural diagnostic, but not directly to training. The clean order is: repair the BF16 source-anchor estimator, run a no-training persistent re-anchor probe on identical rows, then decide whether a trained persistent mechanism merits a new lock.

## 6. Interpretation and P3.5 implications

The currently supported priority order is:

1. **Knife-edge measurement surface:** measured directly and likely the main source of discrete variance.
2. **Undamped endpoint:** not causally tested, but the constant learning rate and late trajectory make it the most actionable source of endpoint arbitrariness.
3. **Ceiling comparability:** a real cross-seed confound, now being measured; not an explanation for within-seed movement.
4. **Fresh-scratchpad coherence:** plausible from GSM8K localization, now strengthened by measured deployed-direction staleness and requiring a dedicated re-anchored persistence probe.
5. **Curriculum lottery:** measured and currently weak as an explanation.

The following P3.5 contracts should be written before training:

- score every registered look at one pinned evaluation ceiling independent of training-time rung;
- decay the learning rate from its registered value toward zero over the final 10% of steps;
- freeze controller parameters and pin the training rung during that landing window, while continuing to log the counterfactual controller decision;
- save raw and EMA checkpoints throughout the landing window, with either EMA or raw-after-decay declared primary before results;
- persist row identity, per-token correct-versus-best-alternative logit margin, and per-row minimum answer-token margin at every look;
- report fix/regression churn and continuous margins by battery, especially GSM8K versus MBPP;
- rebuild the frozen source-token oracle under the exact serving reader and assert 100% source-anchor identity before any registered pi_dir decision;
- re-anchor any persistent write direction from the currently deployed source token rather than reusing a frozen-source direction;
- retain the task inference contract unless a separately preregistered persistence arm is authorized.

The continuous margin is a companion metric, not a replacement for task accuracy. If margins improve while discrete rows churn, the mechanism is improving beneath a discontinuous reader. If both margins and accuracy oscillate, the instability is in the model rather than only the metric.

The landing protocol is not authorized merely by this audit. Its decisive comparison should hold the trained endpoint, evaluation ceiling, and row panel fixed while reading raw and EMA checkpoints under the same evaluator. If the late-window EMA was not saved in P3.4, score-space averages must remain labeled descriptive rather than presented as an EMA counterfactual.

The persistence question should also be isolated from the landing question. A no-training probe can compare fresh-scratchpad and controlled cross-token state carry on identical rows, but any trained persistence mechanism requires its own lock because it changes the inference graph. The first analysis should test whether regression churn concentrates in later answer tokens and in multi-step GSM8K rows relative to MBPP; only then should persistent state be promoted as the next architectural lever.

## 7. Limitations and do-not-claim boundaries

- Only five adjacent row-level pairs survived, all early. Late same-row churn is not identifiable from the archived receipts.
- Score averaging is not weight averaging and cannot estimate an EMA checkpoint.
- The attribution regressions are small-sample, correlated, and descriptive.
- DEV was reused for diagnosis; no threshold, checkpoint, or inference setting may be selected from this diagnostic for a confirmatory claim without a new lock.
- CONFIRM and EVAL-E remain sealed.
- Do not call the endpoint instability an implementation bug unless the paired ceiling or refresh controls fail their identity checks.
- Do not claim persistent state is superior from GSM8K localization alone.

## 8. Plain-language summary

The model is making roughly fifty useful corrections and forty-five harmful ones. The final score is the small difference between those larger numbers. The rows that change are almost exactly tied before the intervention, so small parameter movement can move several rows in either direction. This is expected for a mechanism trained to act on borderline mistakes, but it means the last raw checkpoint is an unnecessarily noisy way to judge it.

The cheapest repair is to make the next run land deliberately: lower the learning rate at the end, stop changing the controller during that landing, save an averaged checkpoint, and always measure with the same ceiling. We should also save a smooth margin metric and every row identity so the next analysis can distinguish genuine deterioration from a few borderline answers changing sides.

## 9. Receipt map

- Drive handoff: `1RsHqcuqyHIu327T0MrTjRDe1n-f8xYls` in the standing research folder
- Drive private oracle archive: `1aEIdnBz1DIW8YbJrXF4jEH3jDilX8Ono`
- Directional-stability summary: `outputs/stage5/stage5_paper2_phase3_p34_directional_stability_20260815/summary.json`
- Directional-stability artifact manifest: `outputs/stage5/stage5_paper2_phase3_p34_directional_stability_20260815/artifact_manifest.json`
- Figure: `docs/figures/p34_directional_stability_20260815.svg` and `.png`
- Fixed-ceiling summary: `outputs/stage5/stage5_paper2_phase3_p34_fixed_ceiling_probe_20260815/summary.json`
- Fixed-ceiling artifact manifest: `outputs/stage5/stage5_paper2_phase3_p34_fixed_ceiling_probe_20260815/artifact_manifest.json`
- Fixed-ceiling figure: `docs/figures/p34_fixed_ceiling_probe_20260815.svg` and `.png`
- Oracle-refresh summary: `outputs/stage5/stage5_paper2_phase3_p34_oracle_refresh_20260815/summary.json`
- Oracle-refresh artifact manifest: `outputs/stage5/stage5_paper2_phase3_p34_oracle_refresh_20260815/artifact_manifest.json`
- Oracle-refresh private archive: `p34_oracle_refresh_receipts_20260815.tar.gz`, SHA-256 `bab977d8936580dbfe6c92ec7b26494ab3cd521f0722d10f51bdc43e24d0f505`
- Source P3.4 handoff: `docs/PAPER2_PHASE3_P34_A2_RESULTS_HANDOFF_20260814.md`

## 10. Questions for strategy review

1. Should P3.5 preregister the EMA checkpoint as primary, or raw-after-decay as primary with EMA secondary? The next run must not choose after seeing both.
2. Should the pinned evaluation ceiling be 0.02, 0.08, or a separately justified value? The paired probe supplies the empirical sensitivity but must not choose the value by itself.
3. The refresh found both a frozen-reader mismatch and material deployed-token re-anchoring. Should the estimator repair and no-training persistence probe precede every P3.5 training lever, or may they run in parallel with landing-protocol work?
4. Should the answer-token margin telemetry be summarized per token, per row minimum, or both? Both is the recommended contract because one weak arithmetic token can determine a GSM8K row.
