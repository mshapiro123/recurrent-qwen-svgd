# Strategy Handoff: D0 Causal Allocation Audit and the D1 Objective Decision

**Date:** 2026-07-27  
**Status:** Complete, read-only audit landed  
**Audit checkpoint:** post-D0 EMA, SHA-256 `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`  
**Canonical receipt:** `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/summary.json`  
**Figure:** `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/causal_allocation_audit.svg`  
**Landing commit:** `8f2c47fe`

## 0. Executive verdict

The audit finds real but sparse value from selective recurrent depth. An oracle that knows which forced loop first matches the cached 7B teacher raises agreement from 72.65% at fixed depth 1 to 78.26% while using only 1.073 loops on average. This is an absolute gain of 5.61 points, or 11,189 additional matched positions out of 199,529.

The trained D0 policy does the opposite. It uses 1.158 loops on average but reaches only 70.46%, 2.20 points below fixed depth 1. It continues on many positions where another loop is harmful or useless and stops on most positions where another loop would help.

The tested deployable linear scalar probe does not recover the oracle opportunity. Under five-fold source-row-grouped cross-fitting, its best point is 144,968 correct versus 144,966 for always stopping after loop 1, a difference of two positions. This is a failure of this feature and model class, not proof that routing is impossible.

The revised immediate-transition target is stable across label-train, calibration, and evaluation, but extremely imbalanced: only about 2.1% of transition labels are `continue`. More importantly, the immediate target is itself incomplete as a policy objective. A position can remain wrong at the next loop and become correct two loops later. The oracle uses depth 3 or 4 on 2,625 positions. A rule that stops on every immediately neutral transition cannot reach those rescues.

**Recommendation:** bank the D0 diagnosis, do not launch D1 from the immediate-help labels, and redesign D1 around a preregistered cost-adjusted return-to-go target. Before model training, run one bounded feasibility study on label-train plus calibration only. Use a fresh untouched partition for any registered D1 test because the present evaluation partition has now been analyzed post hoc.

## 1. Why this audit was run

D0 trained a binary controller against teacher disagreement. That objective did not ask the deployment question directly: whether spending one more recurrent loop changes the current token from teacher-mismatched to teacher-matched without destroying a match already obtained.

The audit therefore forced the post-D0 model through loops 1 to 4 and labeled each one-step transition as:

- `helps`: wrong at loop `d`, matched at loop `d+1`;
- `hurts`: matched at loop `d`, wrong at loop `d+1`;
- `neutral`: both other cases.

It then measured fixed-depth performance, perfect-information oracle depth, the deployed D0 policy, and a post-hoc deployable scalar probe. It also generated a deterministic 100,000-position label-train dry run to determine whether the target distribution was stable enough to support a future D1 design.

This audit was read-only. It performed zero optimizer steps, wrote no checkpoint, and cannot change D0's registered verdict of `not_recoverable_at_pilot_scale` for binary teacher-disagreement targets, 4,000 steps, and one seed.

## 2. Validity and lineage checks

The audit passed the required replay test on all 199,529 evaluation positions:

| Replayed field | Mismatches |
|---|---:|
| Loop-1 prediction | 0 |
| Loop-4 prediction | 0 |
| Selected-loop prediction | 0 |
| Adaptive answer | 0 |

The private reference-row SHA-256 was `4b82296fb7539520aac3a01f0430cecb65a38cd6c081516cced3448b0cc15328`. The checkpoint fingerprint was unchanged before and after extraction. Training was disabled, no optimizer existed, and `optimizer_steps=0`.

The first launch exposed a recurrent-state axis error in the new feature extractor. It failed before producing an interpretable row. The correction normalized captured states from per-loop `[batch, sequence, hidden]` tensors to `[token, loop, hidden]`, bumped the private cache schema, passed the complete 2,270-test repository suite, and then produced the exact replay above. This was an instrumentation defect only; it did not affect D0 or the landed audit.

## 3. Main results

### 3.1 Fixed depth degrades rapidly

| Forced depth | Correct | Accuracy | Mean loops |
|---:|---:|---:|---:|
| 1 | 144,966 / 199,529 | 72.65% | 1.000 |
| 2 | 123,522 / 199,529 | 61.91% | 2.000 |
| 3 | 90,616 / 199,529 | 45.41% | 3.000 |
| 4 | 65,672 / 199,529 | 32.91% | 4.000 |

The average position should not receive more depth. Any useful adaptive policy must identify a small exception set rather than infer that generally difficult positions deserve more loops.

### 3.2 One-step outcomes are asymmetric

| Transition | Helps | Hurts | Neutral |
|---|---:|---:|---:|
| 1 to 2 | 8,564 (4.29%) | 30,008 (15.04%) | 160,957 (80.67%) |
| 2 to 3 | 2,771 (1.39%) | 35,677 (17.88%) | 161,081 (80.73%) |
| 3 to 4 | 1,554 (0.78%) | 26,498 (13.28%) | 171,477 (85.94%) |

At every transition, harmful moves substantially outnumber immediate rescues. The imbalance grows with depth. A generic uncertainty heuristic or inverse-frequency objective can easily spend more compute while lowering agreement.

### 3.3 Oracle headroom is meaningful and compute-efficient

At zero through 0.20 penalty per additional loop, the oracle selects:

| Selected depth | Positions |
|---:|---:|
| 1 | 188,340 |
| 2 | 8,564 |
| 3 | 1,898 |
| 4 | 727 |

This produces 156,155 correct, or 78.26%, at 1.073 mean loops. Relative to fixed depth 1:

- additional correct positions: 11,189;
- absolute agreement gain: 5.61 points;
- mean additional loops: 0.073.

At penalty 1/3, depth 4 drops out under the shallow-tie rule. At penalty 0.5, only depth-2 rescues remain, producing 76.95% at 1.043 loops. At penalty 1.0, the oracle reduces to fixed depth 1. The frontier behaves coherently under the locked utility definition.

This oracle is a teacher-agreement oracle, not an answer-correctness oracle. Its result establishes allocatable teacher-match headroom only.

### 3.4 The deployed D0 policy is dominated by fixed depth 1

The deployed controller obtains:

- 140,586 / 199,529 correct, or 70.46%;
- 1.158 mean loops;
- 4,380 fewer correct positions than fixed depth 1;
- 15,569 fewer correct positions than the zero-penalty oracle.

It is therefore both slower and less accurate than always stopping after loop 1.

At transition 1, its `continue` decision has 780 true positives, 15,385 false positives, and 7,784 false negatives. Continue precision is 4.8% and recall is 9.1%. At transitions 2 and 3, precision falls to 1.1% and 0.7%, with recall near 8%. The controller mostly continues on the wrong positions while missing most immediate rescues.

### 3.5 Most accepted-position damage is policy-induced

Among positions accepted by the plain drafter, 5,928 are lost by the adaptive post-D0 system:

- 974 are already lost at loop 1 after D0 weight training;
- 4,954 are lost after loop 1 because of the selected recurrent path;
- 83.6% of these losses are therefore preventable under an oracle stop-on-nonhelp decision.

The first harmful transition localizes 2,799 losses to loop 1 to 2, 1,495 to loop 2 to 3, and 660 to loop 3 to 4. Code accounts for 3,779 losses and general text for 2,149.

This separates two mechanisms that a future design must guard independently: loop-1 weight regression and post-loop allocation damage.

### 3.6 The cheap deployable scalar router does not recover the opportunity

The probe used per-loop answer margins and logits, prediction change, recurrent-state RMS and update RMS, state cosine, control margin, sequence length, relative token position, and code/general stratum. It excluded teacher-derived features and hidden-state projections. A class-balanced ridge score was fit with source-row-grouped five-fold cross-fitting; thresholds were selected on a separate validation fold for each compute penalty.

Its best point is:

- 144,968 / 199,529 correct, or 72.655%;
- 1.0066 mean loops;
- two more correct positions than fixed depth 1.

For penalties of 0.02 and above, it selects loop 1 everywhere. The tested scalar-linear interface therefore carries no practically useful out-of-fold routing signal. This does not exclude nonlinear functions, bounded hidden-state features, or a controller whose representation is trained for the utility objective.

### 3.7 The immediate-help target is stable but sparse

The deterministic label-train dry run contains 100,000 positions from 228 source rows and 300,000 transition labels:

| Transition | Label-train continue | Calibration continue | Evaluation continue |
|---|---:|---:|---:|
| 1 to 2 | 4.270% | 4.245% | 4.292% |
| 2 to 3 | 1.329% | 1.319% | 1.389% |
| 3 to 4 | 0.713% | 0.765% | 0.779% |
| Pooled | 2.104% | 2.110% | 2.153% |

The distribution generalizes closely across splits. The pooled inverse-frequency ratio is about 46.5 stop labels per continue label on label-train and 45.4 on evaluation.

That stability supports further design work. It does not validate inverse-frequency class weighting as the deployment objective. Because false continuation is much more common and often harmful, a 45-fold positive weight may reproduce the over-computation failure in another form.

### 3.8 Teacher-derived rescue correlates are descriptive only

Of 12,889 evaluation rescue events:

- 8,245, or 64.0%, occur in the lowest quartile of the drafter token's teacher log-probability;
- 9,845, or 76.4%, occur in the upper two teacher-entropy quartiles;
- 8,419, or 65.3%, have the drafter token ranked 2 to 5 under the teacher.

These signals help characterize the target and stratify future data. They are not deployable router inputs because they require the teacher. The teacher top-1/top-2 margin was not cached and was not reconstructed; teacher reload remained prohibited.

## 4. The objective issue revealed by the audit

The immediate label asks whether loop `d+1` helps relative to loop `d`. That is suitable for a one-step causal decomposition, but it is not a complete sequential control target.

The oracle selects depth 3 on 1,898 positions and depth 4 on 727 positions. Some of these paths can require passing through a neutral intermediate transition before reaching the first teacher match. A controller trained to stop whenever the next transition is neutral cannot reproduce that part of the oracle frontier.

The natural policy target is cost-adjusted return-to-go. For a locked per-loop penalty `lambda`, define the value of stopping at loop `d` and the best attainable future value over loops `d+1` through `D`. Continue only when the best future value strictly exceeds the stop value, with shallow ties stopping. In receipt notation:

```text
stop_value(d) = match(d)
future_value(d) = max over j>d of [match(j) - lambda * (j-d)]
continue_target(d) = 1 if future_value(d) > stop_value(d), else 0
```

Equivalent formulations may use an advantage or value head, but the penalty, tie rule, and horizon must be locked before labels are generated. This target aligns training with the oracle frontier and preserves multi-step rescues.

## 5. Recommended decision sequence

### Step 1: Bank the audit

Record the following bounded claims:

1. Selective recurrent depth has 5.61 points of teacher-agreement oracle headroom at 1.073 mean loops on this post-D0 evaluation.
2. The deployed D0 controller is dominated by fixed depth 1.
3. The tested cross-fitted scalar-linear router does not recover the oracle headroom.
4. Immediate utility labels are stable across splits but sparse and myopic for multi-step control.

Do not change D0's registered verdict.

### Step 2: Draft, but do not yet lock, D1

The D1 preregistration should specify:

- a return-to-go or advantage target rather than immediate-help classification;
- one primary compute penalty and any descriptive secondary penalties;
- shallow tie-breaking;
- the controller input interface;
- class weighting derived from policy utility rather than inverse frequency alone;
- an accepted-position guardrail that separates loop-1 weight regression from policy damage;
- fixed-depth-1, fixed-depth-4, deployed-D0, oracle, and cheap-router comparators;
- a fresh untouched evaluation partition and hashes;
- mean loops and agreement jointly, not accuracy alone;
- a single primary endpoint and seed policy.

### Step 3: Run one bounded feasibility study before model training

Use label-train for fitting and calibration for model selection. Do not inspect a new test partition. Compare:

1. the existing scalar-linear probe;
2. a small nonlinear scalar model;
3. a bounded projection or pooled representation from the current recurrent state;
4. an explicit value or advantage head if representation access is authorized.

The feasibility criterion should require a material positive utility gain over fixed depth 1 on calibration at the locked primary penalty, not merely balanced classification accuracy or AUROC. If no deployable interface clears that criterion, stop without D1 training.

### Step 4: Train only after a feasibility pass and preregistration lock

If a representation carries usable signal, train the smallest controller-compatible parameter set first. Protect loop-1 behavior explicitly and evaluate policy-level outcomes at every checkpoint. A class-balanced immediate-help BCE by itself is not recommended.

## 6. Questions for strategy review

1. Should D1 optimize the cost-adjusted return-to-go target above, or a learned value function with the same locked utility?
2. What primary per-loop penalty represents the paper's deployment claim? The audit supplies the full locked frontier but does not choose the normative tradeoff.
3. Should the feasibility ladder stop at nonlinear scalars, or include a bounded hidden-state projection before deciding that the interface lacks signal?
4. Is a new untouched corpus partition sufficient for D1, or should D1 freeze an entirely new corpus slice because the current evaluation family has been examined extensively?
5. Should loop-1 weight preservation and post-loop policy preservation be separate hard guardrails?
6. Is teacher agreement still the intended D1 endpoint, or should an independently scored correctness subset accompany it to bound teacher disagreement?
7. What minimum calibration utility gain should authorize training, given that the scalar-linear result differs from fixed depth 1 by only two positions?

## 7. Limitations and do-not-claim boundaries

- This is one checkpoint and one training seed.
- The oracle sees future teacher matches and is not deployable.
- Teacher match is not synonymous with semantic correctness.
- The scalar probe is linear after fixed feature construction; it does not test all routers.
- Hidden-state projections were not used by the cross-fitted probe.
- The evaluation partition was used for post-hoc analysis and should not serve as a blind D1 test.
- The audit does not authorize D1 training.
- The immediate-help dry run validates target prevalence, not the final sequential objective.

Do not claim that recurrence is generally useful on this corpus, that a router is impossible, that oracle headroom is achievable, or that D1 has passed a feasibility gate.

## 8. Plain-language summary

More thinking usually made this model worse, but not always. If an oracle knew exactly which tokens would benefit, it could improve teacher agreement by 5.61 percentage points while adding very little average computation. The trained controller did not find those tokens. It spent more computation and became less accurate than stopping immediately.

A simple router built from confidence, state-change, and structural measurements also failed to find the useful cases out of sample. The opportunity is therefore real but hidden from the cheap signals tested so far.

The next experiment should not repeat the original binary target or train on whether only the next loop helps. It should first test whether the current state contains enough information to predict the best cost-adjusted future stopping decision. Only a positive feasibility result should authorize another controller-training run.

## 9. Artifact map

- Aggregate JSON: `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/summary.json`
- Short receipt: `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/summary.md`
- Figure SVG: `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/causal_allocation_audit.svg`
- Figure PNG: `outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727/causal_allocation_audit.png`
- Frozen audit spec: `docs/PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_SPEC_20260727.md`
- Private evaluation feature cache: Drive path recorded in the aggregate receipt
- Private 100,000-position label-train cache: Drive path recorded in the aggregate receipt

