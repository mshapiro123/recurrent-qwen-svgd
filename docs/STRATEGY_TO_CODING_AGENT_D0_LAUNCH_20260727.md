# Strategy Handoff to Coding Agent — D0 Router Feasibility Review Resolved, Launch Authorized
Date: 2026-07-27. Responds to: Paper Two D0 Router Feasibility Handoff (2026-07-27). All strategy questions from its section 10 are answered below, the section 7 reconciliation is ratified as mandatory, one measurement amendment and one new read-only job are added, and the locked 4,000-step D0 training run is authorized after the pre-launch items complete. Governing preregistration: PHASE_D0_PREREGISTRATION_DRAFT7_20260726.md (Drive, SHA fda96687…f28c) plus the amendment in section 4 below.

## 1. Strategy readings — adopt these interpretations

These points update the interpretation of the audit data. They matter for how receipts are labeled and how the manuscript is drafted, so they are stated with reasoning rather than as bare rulings.

1.1 The pre-loop probe tested a decision point the architecture does not use. The control token is read at a reserved slot per loop, so the gate's first real decision occurs after loop-1 computation — where the frozen probe's AUROC is 0.671, not the pre-loop 0.586. The `no_deployable_signal` verdict stands as registered, but its scope is: no pre-loop linear route on the frozen model. It is a frozen-model null, not a prior against D0. The program precedent is exact: the frozen pooled-head selector read at 9.1 percent where the trained token pathway later reached 100 percent on the same substrate. Label the probe receipts accordingly: they are the null hypothesis D0's trained controller is now tested against, and a D0 positive becomes attributable to training the pathway precisely because this null is on file.

1.2 The sequential-policy failure indicts thresholds, not training. AUROC plus a validation-tuned threshold is not a policy. D0 does not build its policy that way — it trains against explicit per-position targets with a hard preservation guardrail. Do not carry "sequential router underperforms fixed-2" forward as evidence about D0's design.

1.3 The realistic best case, for calibration of expectations and the spec-dec simulation. Corpus rejection rate is roughly 27 percent (53,389 of about 200k calibration positions). A perfect binary gate — loop 2 exactly on rejections — costs 1.27 mean loops, matching the oracle's own sweet spot (97 percent of oracle gain at a 1.25-loop budget). If the trained controller reaches the fixed-depth-2 ceiling on rejections with accurate gating, R lands near 12.8 points (14.47 minus the 1.70 tie artifact), inside the registered strong band, and overall teacher acceptance rises from about 73 to near 77 percent. These are ceilings, not predictions, but they are the numbers the gamma sweep should be read against.

1.4 The 79 percent never-recovered share is the storage boundary, quantified. Under the program's width-as-storage, depth-as-composition framing, the frozen floor decomposes the drafter-teacher gap: about a fifth is composition-recoverable (two-thirds of that at exactly one extra loop), and the rest is the knowledge share that no amount of vertical compute reaches on a frozen model. This is a headline Paper Two sentence. State it as a frozen, vertical-only floor — training and the future horizontal axis can shrink it, and it says nothing about deeper-than-6 or structural unreachability (the registered do-not-claim already covers this).

1.5 The stability-heuristic inversion is a standalone finding. Positions whose prediction stabilized recover later 2.6 times more often than positions whose prediction changed (8.72 versus 3.36 percent). This breaks convergence-based exits on this substrate and should be drafted as its own paragraph, connected to the D0 baselines and to the training-free-exit alternative in the T1 design memo.

1.6 Reviewer footnote required: depth-1 "agreement" of 1.70 percent on rejected positions is nonzero purely through the section 5 tie artifact — by construction it would otherwise be zero. Say so wherever the table appears.

## 2. Section 10 resolutions (Mark, 2026-07-27)

1. Binding training target: confirmed. Accepted positions depth 1, all rejected KL quartiles depth 2, per the landed graded-branch table. The isotonic and parametric fits remain descriptive receipts only.
2. Scope: registration untouched, manuscript narrowed to "selective extra-compute allocation" — one well-chosen extra loop. Pinned for the future, not now: a third training loop may enter when distilling a large teacher, only after the horizontal (Coconut) pathway is integrated and stable. The number to watch when that revisits is the first-correct depth-3 share (3 percent of rejections on the floor).
3. Probe placement: the deployable-probe result enters Paper Two as a labeled exploratory calibration diagnostic. Its sentence: control signal emerges during computation, not before it — motivating the per-loop gate architecture and closing the pre-allocation alternative.
4. D1 direction: utility-trained scalar-dynamics router agreed in principle, commitment deferred to the post-D0 decision point. The policy objective there is expected marginal gain versus explicit compute cost, never any-later-correct AUROC.
5. Tie-breaking: adopted as standing policy from D0 training onward — deterministic argmax tie-breaking (lowest token id) in a fixed fp32 logit space, tie cells flagged in receipts. Floor labels retain their disclosed ambiguity; both realizations agreed on every verdict.
6. 14B sensitivity: one main-text paragraph (depth demand depends on the target teacher, previewing the teacher-shift test), full tables in the appendix. Context note: the 7B/14B comparison grows in importance when the horizontal loop arrives, and the 14B is confirmed A100-feasible.

## 3. Mandatory pre-launch: the section 7 reconciliation

Ratified exactly as the feasibility handoff specifies, as a code correction implementing the locked branch and not an experimental choice: accepted positions target depth 1; rejected positions use the KL quartile from frozen boundaries and `floor["calibration"]["targets"][quartile]`; `natural_stop_recall` uses the identical helper; isotonic and parametric fits are never called by the graded branch; startup assertions print the target table and verify the observed training-target distribution; unit tests prove the graded branch cannot reach the isotonic mapping. A target-policy receipt is emitted by the launcher before the first training step. D0 does not launch until these are green.

## 4. Measurement amendment: the teacher-shift statistic (confirmed by Mark, 2026-07-27)

The registered saturation definition — smallest depth within 1 point of the curve's depth-6 value — presumed rising-then-plateau curves. The observed aggregate curves peak at depth 2 and decline, so the registered rule would read the declining tail and return noise. The amendment, with its reasoning recorded so the receipts explain themselves:

- The aggregate agreement curve at depth d counts positions correct at that specific depth, so it is recovery minus harm — a net-utility statistic dominated by the drafter's churn at depth, which is largely a drafter property, not a teacher-gap property. Both teachers already peak at depth 2 on the floor, so a peak-based statistic has near-zero discriminating power for the hypothesis. Worse, on the trained model it is close to circular: training installs depth-2 competence, so the post-training peak at 2 is the intervention observing itself.
- The amended test statistic is therefore the **median first-correct depth among recoverable positions, per teacher, with the full first-correct distribution reported**. This isolates depth demand in the recoverable subpopulation and ignores harm. The aggregate peak is retained as a one-line descriptive (both teachers peak at 2 on the floor).
- The test runs at **two time points**. Floor layer: computed now, read-only, from existing receipts — the six forced-depth prediction sets exist and both teacher caches exist, so the 14B first-correct distribution is pure post-processing (new job, section 5). The 7B floor median is already known to be 2. Trained layer: the registered depth-1-through-6 sweep on the trained drafter, scored against both cached records, as locked.
- Pre-stated readings, unchanged in spirit: the median shifts upward with teacher depth (toward 3 against the 14B) — depth demand tracks the gap, supporting depth-as-composition; the median is the same against both teachers — depth demand is a property of the text; degenerate or near-empty recoverable sets — inconclusive, stated as such.
- Transparency statement to include verbatim in the amendment receipt: this clarification was made after the floor data landed and before any trained model exists. The floor layer is therefore partially observed at amendment time (the 7B distribution was visible; the 14B distribution had not been computed). The trained-model layer, which carries the registered test, is untouched. The trained-model comparison carries one bias both directions share and the receipt must state: binary depth-2 targets compress trained demand toward 2 for both teachers, so the trained layer measures demand under the trained policy's regime, and the floor layer is the less confounded of the two.

## 5. New read-only job (no GPU beyond post-processing): 14B floor first-correct distribution

From the existing floor predictions and the 14B cached records, compute the first-correct-depth distribution and median on the 53,389 7B-rejected positions, and additionally on the 14B's own rejected set if the cache supports it. File alongside the oracle audit with the amendment receipt. This lands the floor layer of section 4 before training starts.

## 6. Launch authorization and order

1. Section 3 patch, assertions, tests, and target-policy receipt — green required.
2. Section 5 post-processing job and the section 4 amendment receipt filed.
3. Tie-break policy implemented in the training and evaluation path.
4. Launch the locked 4,000-step D0 training run exactly per PHASE_D0_PREREGISTRATION_DRAFT7_20260726.md: T1-lite-R trainable set, lambda 0.5 equal weights, 30 percent rehearsal, guardrail evaluations at 1,000/2,000/3,000 with the abort rule, final-step EMA primary as registered.
5. Full registered evaluation battery on the untouched evaluation partition: R (headline), gamma {2,4,8} spec-dec simulation with banked wall-clock, depth-response curves, loop-usage correlation, per-stratum reporting, the amended teacher-shift test, ARC allocation probe, both hard guardrails.

## 7. Boundaries

The evaluation partition stays untouched until the final battery. Interpretation bands and the do-not-claim list bind as registered, plus: the router audits are exploratory calibration diagnostics and never alter the registered bands; no pre-loop linear sweeps regardless of outcome; D1 work of any kind waits for the post-D0 strategy decision; no C-track training, no RG-12, no reopening of anything pinned. If any pre-launch item proves infeasible, stop and report — amendment before launch is free, substitution after it is not.