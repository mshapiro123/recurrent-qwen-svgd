# Strategy Handoff to Coding Agent — Rung 0 Re-Scoring, Fresh Evaluation Slice, and the Turn to Composite Depth-by-Append

Date: 2026-07-28. Responds to: Paper Two Causal Allocation Audit handoff (2026-07-27). Supersedes nothing; extends STRATEGY_TO_CODING_AGENT_D0_BANK_AUDIT_20260727.md (Drive `1VWJuyfV4Dixquy7GgLQaCBmHggI58Z7e`, SHA `1e16439d…a7eb`). Three decisions from Mark, 2026-07-28, are carried here: rung 0 of the routing ladder is a GO; a fresh evaluation partition is approved; and **the next experimental direction is composite depth-by-append, not pure-vertical D1**. Concept basis: COMPOSITE_LATENT_LOOPING_CONCEPT_20260723.md section 5.5.

## 0. Why the direction changed, stated once so the receipts explain themselves

The causal allocation audit did not find a weak router. It found a hostile substrate. At the 1→2 transition, in-place vertical looping produced 8,564 helps against 30,008 hurts — a second loop is roughly three and a half times more likely to destroy a correct token than to rescue a wrong one. Any gate operating on that substrate is being asked to win a bet whose odds are set against it by the mechanism, and the oracle's own frontier (78.26 percent at 1.073 mean loops, +5.61 points over fixed depth 1, on 11,189 positions) is exactly what a perfect referee earns on a bad wager: real, but purchased almost entirely by *not looping*.

The geometric reading, developed with Mark and recorded in concept-note section 5.5: the recurrent block's native operation is attending over a sequence to compute a new position. Vertical looping violates that native operation — it re-runs the block on one position and overwrites that position's state in place. In-place refinement keeps no memory of what it replaced, so a loop that was not needed deletes a representation that was already correct. That is the mechanism generating the hurts class, and it is architectural, not a training deficiency.

Depth-by-append routes the same extra computation through the block's native interface instead. The prior state is never overwritten: it remains in the KV cache and the readout attends back over it. A wasted step then adds a distractor rather than deleting an answer. The prediction is falsifiable and cheap to test: **the hurts class should collapse toward zero under append while it stays near 30,008 under in-place**, measured by the identical helps/hurts/neutral transition decomposition the audit already defines.

If that prediction holds, the routing problem changes character entirely. Under in-place looping the router must be *right*; under append it need only be *affordable*, because the penalty for an unnecessary step drops from an accuracy loss to a compute cost. That is why this measurement comes before any further gate training.

## 1. Rung 0 — expert-choice re-scoring of the saved cross-fitted scores (GO, no GPU)

Pure post-processing on scores the audit already saved. The purpose is not to rescue D0's policy. It is to answer, at zero cost, a question the audit left open: was the deployable scalar signal *mis-thresholded*, or is it *absent*? D0's policy ran an absolute threshold and netted +2 positions out of 199,529 at 4.8 percent continue precision. Expert-choice routing — experts select their top-C tokens under a fixed capacity, rather than tokens crossing an absolute bar — converts that absolute-threshold problem into a relative-ranking problem, which is the one place the MoE literature has a genuinely better answer than we were using.

Specification:

1. Budget grid: allocate a second loop to the top B fraction of positions by cross-fitted score, sweeping B over {0.5, 1, 2, 5, 10, 20, 27} percent. The 27 percent endpoint is the corpus rejection rate, so it is the budget at which a perfect gate would loop exactly the rejected set.
2. Two blocking regimes, both reported. **Global**: rank across the whole partition — not deployable (a decoder cannot rank positions it has not reached), reported as the ranking-quality ceiling. **Local**: rank within a causal fixed window of 256 and of 1,024 tokens — deployable, and the number that decides anything.
3. Primary metric: net utility at the allocated set, computed as helps minus hurts against the cached 7B teacher, reported with the realized mean loop count, and plotted against three references — fixed depth 1, fixed depth 2, and the audit's oracle net-utility frontier at matched budget.
4. Secondary, and diagnostically the more informative pair: the score's AUC against the true helps-versus-hurts label (not against any-later-correct), and precision@B at every grid point.
5. Cross-fitting discipline unchanged from audit item 4 — fit on one fold of source rows, score held-out folds. No refitting on the evaluation partition.

Pre-stated reading, so the outcome cannot be renarrated afterward. If the locally-blocked expert-choice net utility is negative at every budget on the grid, the deployable scalar signal is absent rather than mis-thresholded, and the exotic-router contingency closes as a receipted negative — we will have measured it rather than assumed it. If some budget turns positive, record the budget and the margin; it becomes the fixed-budget comparator the composite has to beat, and it does not by itself reopen the D-axis.

## 2. Fresh evaluation partition, EVAL-B (approved)

The D0 evaluation partition has now been read for six post-hoc decompositions. Every one was legitimately labeled exploratory, and none of them are retracted — but the partition is no longer a clean surface for a measurement with a decision attached to it, and the composite comparison in section 3 has a decision attached to it.

Specification:

1. Size 0.2M tokens. Same locked sources and revisions as D0: FineWeb-Edu dump CC-MAIN-2025-26 and `bigcode/the-stack-smol` at the pinned revision. Same mixture proportions and same stratification scheme as the D0 evaluation partition, so the two are comparable by construction.
2. Documents disjoint from every prior partition (train, calibration, label-train, D0 evaluation) by document hash. Emit a manifest listing document ids with the SHA-256 of the id list, and assert empty intersection against all prior manifests before any scoring runs.
3. One cached Qwen2.5-7B greedy pass, banked with the same record format as the existing caches. The 14B pass is **not** commissioned here — the teacher-shift question is not what EVAL-B is for, and the cost is better spent later.
4. Handling policy: EVAL-B is read-once for the section 3 comparison. Log every read with its purpose. When it is spent, it is spent, and the next clean measurement gets a new slice — this is now the standing pattern, not a one-off.

## 3. DC0 — the composite depth-by-append measurement

Naming: the merge is neither pure D-axis nor pure C-axis, so it gets its own label. **DC0** is the first composite measurement. The D-axis is paused at D0-banked; D1 as previously scoped (utility-labeled pure-vertical training) is **not** cancelled but is deferred behind DC0, because a clean measurement of the destructive-substrate ceiling is worth much less than a measurement of whether we can leave the destructive substrate.

The decisive property of this experiment: **DC0 requires no training.** It is a forward-only forced-depth sweep on an existing checkpoint with a modified forward path. That is why it comes first.

### 3.1 Mechanism — M7, per-position append with transient eviction

Add to the composite design (COCONUT_INTEGRATION_DESIGN_20260725.md) as modification M7, in the same numbering as M1–M6 and inheriting their contracts.

At a decision position t, instead of re-applying the recurrent block to position t's state in place, append k latent positions after t. Each appended slot uses the `<|recur_readout|>` id per M3, with its embedding replaced by the final post-norm hidden state of the preceding position per M1, through the horizontal bridge at identity per M2 and the section-7 resolution. Position ids run continuous through the appended slots; attention includes them; the visible-generation mask on the control trio stays in force. The next real token is read at the last appended slot. Position t's own state is never overwritten and remains in the cache throughout — that is the entire point of the modification.

**Eviction.** After the token is emitted, the k appended slots are dropped from the KV cache before the next real token is processed, and position ids for subsequent real tokens are those of the unmodified sequence. Rationale: transient eviction keeps the visible sequence exactly aligned with the plain drafter's, so the k = 0 path is bit-comparable to the baseline and the helps/hurts decomposition compares like with like. A persistent scratchpad — appended slots left in the cache for all later tokens — is the more interesting long-run variant and reintroduces an additive cross-token channel we would then have to characterize. It is explicitly out of scope for DC0 and pinned for later.

**Accounting.** Extend M4's dual counters to the append grid: total block applications, feedback-producing applications, and now evicted-slot counts per decision position, asserted under forcing. An eviction that silently fails to evict turns DC0 into the persistent variant without anyone noticing, so the assertion is load-bearing rather than decorative.

### 3.2 Preconditions before the measurement is interpretable

DC0 is forward-only, so the full RG battery is not a precondition — forward-only measurement needs forward-only contracts. Required green before scoring:

1. RG-1 extended to M7: at k = 0 the composite forward must match the registered surgery within 1e-3, at L = 1 and at forced L, both budgets.
2. The M4/M7 counter assertions above, including eviction counts.
3. The **M2 RMS diagnostic**, which is now a precondition rather than a nice-to-have: report RMS statistics of final post-norm hidden states against embedding rows on this substrate. Coconut's raw identity feedback worked on GPT-2; Qwen2.5's scales are not guaranteed to match, and a scale mismatch would make an untrained append look catastrophic for reasons that have nothing to do with the geometry under test.

The RG-4 epsilon-stability sweep and the RG-11 precision comparison remain queued and remain **not** preconditions for DC0, because DC0 takes no gradients. They stay preconditions for anything with a training loop.

### 3.3 Arms

All arms are forced-depth passes over the same EVAL-B positions against the same cached 7B teacher.

1. **In-place vertical**, forced depth 1 through 4. This is the D0 mechanism re-run on the fresh slice. It is the matched baseline and simultaneously a pipeline validity check: it should reproduce the audit's harm asymmetry within sampling error. If it does not, stop and report — something is wrong with the slice or the scoring path, and nothing else in DC0 is trustworthy until that is resolved.
2. **Append, raw feedback**, forced k = 0 through 3, bridge at identity.
3. **Append, RMS-matched feedback**, same grid, with the fed state rescaled to match embedding-row RMS. Cheap, and it separates a geometry result from a scale artifact.
4. **Neutral-append control**, forced k = 0 through 3, where the appended slot carries the unmodified `<|recur_readout|>` embedding row with no fed state. This arm is what makes a low hurts class mean something: if hurts is near zero here too, then appending *anything* is harmless and the fed state has not been shown to matter. It separates harmlessness from usefulness, which are the two distinct claims section 5.5 makes.

### 3.4 Metric and pre-stated readings

Metric: the identical helps/hurts/neutral transition decomposition from audit item 1, applied at each transition (0→1, 1→2, 2→3 for append; 1→2, 2→3, 3→4 for in-place), with per-stratum breakdown and the same teacher-confidence stratification as audit item 6. The receipted comparator is the audit's 1→2 in-place figure: 8,564 helps against 30,008 hurts.

Readings, stated before the data exists. These are interpretation bands, not pass/fail gates — DC0 is a diagnostic, and gating a diagnostic is how D0's label got locked in.

1. Append hurts at the first transition falls below roughly one third of the matched in-place hurts, with helps at or above the in-place level and the neutral control showing materially less help: the geometry argument survives its first real test, the routing problem becomes a budget problem, and a trained composite gate is the next design.
2. Append hurts is comparable to in-place: check arm 3 before concluding anything. If RMS-matched also shows high hurts, the geometry argument is wrong on this substrate and section 5.5 is amended to record that.
3. Append hurts collapses *and* append helps collapses *and* the neutral control matches arm 2: the append is inert — the model ignores an input distribution it has never seen. This is the most likely outcome for an untrained feedback path and it is not a refutation. It is a statement that the measurement needs the fed state to be in distribution first. Contingency, held for Mark's markup and **not authorized here**: a short bounded adaptation run training only the horizontal bridge ΔW with everything else frozen, a few hundred steps, purely to bring the fed state into distribution, followed by re-measurement on a second fresh slice. Cost is small; the authorization is Mark's, because it is a training loop and this handoff does not authorize training loops.
4. Degenerate or near-empty recoverable sets at any transition: inconclusive, stated as such, same convention as the teacher-shift test.

### 3.5 What DC0 does not do

It does not test whether the composite reasons better, produces spec-dec speedup, or beats any teacher. It tests one thing: whether the append mechanism is gentler than in-place looping on the transition ledger. Every other composite claim waits for a trained run, and the do-not-claim list from D0 carries forward unchanged.

## 4. Program bookkeeping

The MoE exotic-router contingency is retired from the critical path. Section 1's rung 0 is its last measurement, and it runs because a budgeted-ranker number on the current substrate is the baseline that any composite improvement gets measured against — not because we expect it to rescue the D-axis.

The label-to-objective alignment section remains mandatory on every future preregistration, per the standing policy set in the bank-audit handoff. DC0 has no training label, so the section reads as a statement that the measured quantity is the deployment question itself: harm caused by a wasted computation step.

Still queued, unchanged, and independent of all of the above: RG-4 epsilon sweep, RG-11 three-policy precision comparison with the H-scaling diagnostic, S3/S5 Qwen3 screening probes, concept-note section 5.3 gate-token strategy (open), the P1/P2/P3 pairing decision (open), the WP1 oracle train-subset readout (status never confirmed), and the Arm G manuscript section (not started).

## 5. Boundaries

No training of any kind is authorized by this handoff. The D0 verdict, bands, and do-not-claim list are final and untouched. EVAL-B is read-once for section 3 and is not to be used for tuning, thresholding, or fitting anything. The persistent-scratchpad variant, trained horizontal bridges, L above 1 in the composite, RG-12, GRAM, width, and every existing pin stay closed. If any precondition in section 3.2 fails, stop and report before scoring — a red precondition makes DC0's comparison uninterpretable rather than merely noisy, and amendment before measurement is free.
