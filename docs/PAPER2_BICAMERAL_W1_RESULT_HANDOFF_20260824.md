# Paper Two Bicameral W1 Ladder Result Handoff

Date: 2026-08-24  
From: coding agent  
To: strategy agent and Mark  
Status: W0, W1 Phase A, W1 Phase B, the authorized generative read, X-6, and D-M5 are complete. Step 2 remains blocked. CONFIRM and EVAL-E remain sealed.

## 1. Executive result

The W1 ladder establishes a clean causal boundary.

1. The deployed write interface is live. A row-specific student correction delta, L0a, changes teacher-token margins and changes complete generated answers.
2. The available population targets are not answer-grade. The only fixed, deployable target, the global mean L2, produced exactly the same prediction as its control on all 461 rows in both seeds.
3. Cluster routing does not rescue the target. L1 and L3 use a gold-answer-derived routing feature. L1 harms generation. L3 gains 9 to 10 rows over its own shuffled control, but remains 11 to 20 rows below the random-direction control.
4. The frozen k=2 partition is primarily a task-family split. Cluster 0 contains 263 ARC-Challenge rows and 39 MBPP rows. Cluster 1 contains 1,732 GSM8K rows.
5. The top shared residual directions fail. No tested L6 sign and direction is positive in both seeds. The registered residual read is `H-noise`.

The recommended integrated key is `TARGETS-NOT-ANSWER-GRADE`. The margin-only Phase B result is `GLOBAL-STEER-WITH-GLOBAL-ADVANTAGE`, which is outside the charter's simpler L1 approximately L2 branch and therefore requires strategy adjudication. The generative result resolves the practical question: positive teacher-token margins from fixed population directions do not transfer into useful complete answers.

## 2. Plain-language summary

We tested whether the model could be helped by nudging its internal state in several proposed correction directions.

When we gave the model a direction computed specifically for the current question and its correct answer, the nudge worked. That proves the pathway can affect behavior. It does not prove a deployable system can find that direction on its own.

When we replaced that oracle direction with reusable directions learned from groups of examples, the result disappeared. The global reusable direction changed token margins but produced no answer change relative to its control. The cluster-specific directions mostly separated math questions from other task families, rather than discovering general reasoning modes. The tested residual directions also failed.

The architecture is therefore not mechanically dead. The missing component is a deployable conditional map from the current student state to a useful correction. Fixed average directions are too coarse.

## 3. Authority and scope

The governing authorization is `STRATEGY_BICAMERAL_W1_LADDER_AUTHORIZATION_20260824.md`, Drive `1TQH3WUvdzp-ueriVaYcaOFnyD1GIU1dk`, 8,703 bytes, SHA-256 `a998a51ab497b89c67d44f57213ef12a5dc12dbc136897ef9397f5d793e0e7b1`.

The binding clarification is `STRATEGY_BICAMERAL_W1_RULINGS_20260824.md`, Drive `1HIimhh_-2DPMNBK-uydgXzIKF0fy2N5n`, 8,836 bytes, SHA-256 `7fa09f4b2dc1678f681962a160f754f934783ec3dc05ae98f44dba2c574af726`.

The run was score-only. It constructed no optimizer and took zero optimizer steps. It read DEV-2 and the authorized 461-row generative population only. It did not score CONFIRM or EVAL-E.

## 4. Experimental design

### 4.1 Phase A: row-level target families

Five target families were injected through the frozen bicameral graph at gamma 0.05 under the registered sequential schedule. Each target was compared with its own shuffled control and a random-direction control. A family could win only if its confidence interval was positive in both seeds and it beat both controls.

| Family | Target meaning | Mean margin change | Mean shuffled change | Registered outcome |
|---|---|---:|---:|---|
| L0a | Student correction delta | +1.8893 | +0.4685 | Eligible |
| L0b | Teacher late-state tensor | -0.1843 | -0.1892 | Failed |
| L0c | Teacher-forced student correction delta | +2.3515 | +0.9840 | Winner |
| L0d | Alternative state target | -0.0276 | -0.0341 | Failed |
| L0g | Alternative correction target | -0.1963 | -0.1937 | Failed |

The L0a and L0c target outputs are identical across seeds. These are two provenance replications, not independent learned-model replications.

### 4.2 Phase B: granularity and residual structure

The frozen Stage-0 k=2 transform and centroids were extended to all 2,048 DEV-2 rows without refitting. Both clusters exceeded the registered 5 percent minimum.

| Arm | Meaning | Seed 0 mean, 95% CI | Seed 1 mean, 95% CI |
|---|---|---|---|
| L1 | Own-cluster mean | +1.5021 [1.4408, 1.5642] | +1.5021 [1.4408, 1.5642] |
| L2 | Global mean | +1.5563 [1.5229, 1.5885] | +1.5563 [1.5229, 1.5885] |
| L3 | Other-cluster mean | +0.1759 [0.1651, 0.1864] | +0.1759 [0.1651, 0.1864] |
| Best L6 | Best tested residual sign | -0.0102 [-0.0224, 0.0023] | -0.0199 [-0.0320, -0.0074] |

The paired L1 minus L2 mean is -0.05425 in both seeds. Its 95 percent interval is wholly negative in each seed. Global is modestly but measurably better than own-cluster on the margin estimand.

### 4.3 Generative read

The registered 461-row population contains 369 GSM8K rows, 67 MBPP rows, and 25 Tier-1 rows. Every positive-margin family was scored with the same fixed target at every autoregressive step.

L0a and L0c are `oracle-target-assisted`. Their numbers are causal contrasts against same-row controls and are not capability estimates. L1 and L3 use population target values but select the target with a gold-answer-derived negative-CE correction-gradient feature. They are also not deployable-grade. L2 is the only fixed, deployable population target.

| Arm | Seed 0 correct | Seed 1 correct | Matched control, seeds 0/1 | Net rows versus control, seeds 0/1 | Scope |
|---|---:|---:|---:|---:|---|
| L0a | 170 | 170 | 85 / 83 | +85 / +87 | Oracle-assisted causal contrast |
| L0c | 104 | 104 | 100 / 99 | +4 / +5 | Oracle-assisted causal contrast |
| L1 | 69 | 69 | 77 / 83 | -8 / -14 | Oracle-routed population value |
| L2 | 78 | 78 | 78 / 78 | 0 / 0 | Deployable fixed target |
| L3 | 145 | 145 | 135 / 136 | +10 / +9 | Oracle-routed population value |
| L4 random | 156 | 165 | Not applicable | Not applicable | Random-direction control |

Paired bootstrap intervals and exact McNemar tests give the following key contrasts.

| Contrast | Seed 0 paired difference, 95% CI | Seed 1 paired difference, 95% CI | Exact p, seeds 0/1 |
|---|---|---|---|
| L0a minus own shuffle | +18.44 pp [13.88, 22.99] | +18.87 pp [14.10, 23.43] | 3.89e-14 / 8.36e-15 |
| L0a minus random | +3.04 pp [-2.39, 8.46] | +1.08 pp [-4.56, 6.51] | 0.304 / 0.757 |
| L1 minus own shuffle | -1.74 pp [-3.90, 0.43] | -3.04 pp [-4.99, -1.08] | 0.152 / 0.0043 |
| L2 minus own shuffle | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 1.0 / 1.0 |
| L3 minus own shuffle | +2.17 pp [0.22, 4.12] | +1.95 pp [0.22, 3.69] | 0.041 / 0.049 |
| L3 minus random | -2.39 pp [-5.86, 1.08] | -4.34 pp [-8.24, -0.65] | 0.235 / 0.033 |

The L2 shuffled control is intentionally degenerate because permuting one global mean does not change it. Its exact prediction identity is therefore a construction check, not independent evidence. The random-direction comparison is the informative harder control. L2 trails random by 78 and 87 rows.

## 5. Integrated interpretation

### 5.1 What passed

- The write interface is causally live.
- L0a establishes a large same-row target-versus-shuffle effect on GSM8K. The net is +89 and +90 GSM8K rows in the two seeds.
- The fixed cluster extension is numerically stable. Both seeds assign the same 313 and 1,735 rows with 100 percent cross-seed agreement.
- The common correction field is structured. The W3 desk read finds about 71.3 percent common-mode energy and a cross-fitted state-plus-trajectory map cosine near 0.885.

### 5.2 What failed

- L0c wins the margin ladder but does not produce a meaningful generative advantage over its own shuffle and performs far below random direction.
- L1 is worse than its control, especially on GSM8K.
- L2 produces no answer-level effect at all.
- L3 gives a small same-shuffle gain, but its routing is oracle-derived and it does not beat random direction.
- All tested L6 residual directions are non-positive.

### 5.3 Mechanistic reading

The data reject a simple fixed-direction correction model. A useful correction depends on the row. Averaging directions preserves a positive teacher-token margin signal but discards the conditional information required to improve a full answer.

The W3 map result supplies the credible next hypothesis. Student state and trajectory features may predict row-specific correction structure well enough for a learned conditional map. That is a different estimand from fixed mean directions and is not limited by the failed L1/L2/L3 comparison in the same way.

## 6. Limits and do-not-claim boundaries

1. Do not quote L0a or L0c generation accuracy as model capability. Both use oracle-derived same-row targets.
2. Do not call L1 or L3 deployable. Their values are population summaries, but their row routing depends on a gold-answer-derived correction gradient.
3. Do not call the two clusters abstract reasoning modes. They are strongly associated with battery family.
4. Do not treat identical target outputs across seeds as two independent functional replications. The initialized target states are functionally identical.
5. Do not interpret L2's shuffle equality as a successful negative control. It is algebraically guaranteed for a single global vector.
6. The generative population covers GSM8K, MBPP, and Tier-1 only. ARC and MMLU appear in the margin panel but not in the complete-answer read.
7. The result does not prove that conditional correction is impossible. It shows that these fixed population targets are not answer-grade.
8. CONFIRM and EVAL-E remain untouched.

## 7. Execution integrity and incidents

- All five checkpoint files for both seeds were restored from Drive and matched the registered SHA-256 values.
- The W0 sequential-schedule identity gate remained the controlling evaluator identity.
- The generative manifest is 331,329 bytes, SHA-256 `01457fe62207263d0f04a03e61a906289c34db55c2d929bfe6a841eeef2607af`.
- The generative config is 499 LF bytes, SHA-256 `d87943947de19de6e54e475da7462a02bd910f840a6642604bd6c56b31a78f85`.
- The replacement-runtime cost probe projected 3.3094 A100-hours for 22 cells, below the 8-hour cap.
- A Colab runtime token expired during scoring. The server assignment and scorer remained alive. Reattachment used the same endpoint and did not restart or duplicate the scientific process.
- The score process ran from approximately 13:58 to 17:10 local time based on durable artifact timestamps.
- The final generation bundle is 20,829,637 bytes, SHA-256 `41b03693bb06cb915f44e4a03cb0da5649fab69717e5650a405b04e3d880a186`. Its 59 internal files all pass the embedded size and SHA manifest.
- The A100 was released after bundle verification. Colab reported no active server sessions.
- Optimizer constructed: false. Optimizer steps: 0. CONFIRM scored: false. EVAL-E scored: false.

## 8. Questions for strategy adjudication

1. Ratify the integrated generation key as `TARGETS-NOT-ANSWER-GRADE`.
2. Ratify `H-noise` for the tested L6 residual family.
3. How should the margin branch be named? The observed result is global greater than own-cluster, not the registered global approximately own-cluster relation. The coding-agent receipt uses `GLOBAL-STEER-WITH-GLOBAL-ADVANTAGE` and flags it for adjudication.
4. Correct the R4 scope language so L1 and L3 are explicitly oracle-routed population values, not deployable-grade targets.
5. Decide whether W2 Step 1 should still run as written. My recommendation is to pause any fixed-direction Step 1 and instead lock a conditional state-to-correction map experiment using only deployment-available features.
6. Retain L0a as an interface-positive causal control, with the permanent prohibition against capability language.

## 9. Recommended next steps

1. Bank W1 and preserve the sealed partitions.
2. Resolve the branch-name and R4 scope corrections before any new GPU work.
3. Draft the next experiment around the conditional map estimand supported by D-M5. Require a deployable feature set and same-row shuffled and random controls.
4. Keep the global mean and cluster means as descriptive baselines, not primary target families.
5. Require a generative read at the selection stage. Margin-only selection favored L0c and L2, both of which failed to provide deployable answer gains.
6. Do not reopen residual eigen-direction injection without a new mechanism that explains why all tested signs were non-positive.

## 10. Artifact ledger

Primary machine-readable receipts:

- `artifacts/bicameral_w1_20260824/w1_phase_a_aggregate.json`
- `artifacts/bicameral_w1_20260824/w1_phase_b_analysis.json`
- `artifacts/bicameral_w1_20260824/w1_generation_analysis.json`
- `artifacts/bicameral_w1_20260824/w3_desk_summary.json`
- `artifacts/bicameral_w1_20260824/extracted/generation/generation_bundle_manifest.json`

Figures:

- `artifacts/bicameral_w1_20260824/figures/paper2_bicameral_w1_summary.png`
- `artifacts/bicameral_w1_20260824/figures/paper2_bicameral_w1_summary.svg`
- `artifacts/bicameral_w1_20260824/figures/paper2_bicameral_w1_cluster_composition.png`
- `artifacts/bicameral_w1_20260824/figures/paper2_bicameral_w1_cluster_composition.svg`

Transport bundles:

- Phase A seed 0: 35,002,614 bytes, SHA-256 `e465cc3252defde42a2a571b8d5e87352cbb6de4faa75231b5f91834cd63f7fc`
- Phase A seed 1: 35,001,389 bytes, SHA-256 `f60a2e20745601b8b291aecbeaea87497d7c510d964b083703be9631b220fcf6`
- Phase B: 14,409,425 bytes, SHA-256 `b9c22650f4137eee038fa84d42ef7d060aa7532da4bf1b675fddfccab1372344`
- Generation: 20,829,637 bytes, SHA-256 `41b03693bb06cb915f44e4a03cb0da5649fab69717e5650a405b04e3d880a186`

The separate receipt manifest records byte counts and SHA-256 values for the handoff, analyses, and figures at publication time.
