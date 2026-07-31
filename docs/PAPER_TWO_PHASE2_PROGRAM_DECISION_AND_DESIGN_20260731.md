# Paper Two Phase 2 — Program Decision and Design: Arbitration at Two Interfaces

Draft 3, 2026-07-31 (draft 1 revised same day for Mark's bounded-correction sidecar architecture and the gradient-characterization requirement; draft 3 folds in the theoretical foundations audit — amendments A1–A4 and verification computations V1–V5; section 9 recommendations approved by Mark). Companion and binding design context: THEORETICAL_FOUNDATIONS_AUDIT_20260731.md. Strategy lane, approved for Drive publication. Responds to: DC1 Stage A Result Handoff (Drive `1kaMAA7o9AJ3KR1wq53YhvL2RwowOBz3U`, verdict `none`, consequence `transient_append_retires`). This document records a program-level decision, defines two new experimental lines (DC2, D1) and the manuscript track, and declares a methodology addition (exploration/confirmation split). It authorizes no training; each line trains only under its own future locked preregistration.

## 0. Plain-language summary

The scratchpad experiment failed its safety test, but its own data shows why and what would fix it. If a perfect referee chose, position by position, between the model's original answer and the scratchpad's answer, the scratchpad would already be a net win — all of its fixes, none of its breakage. The missing piece is not the scratchpad; it is the referee. The same is true of the older in-place looping mechanism, where a perfect referee is worth even more. So the program's next phase asks one question at two places: can a referee be *trained*? One line (DC2) trains a small gate that chooses which answer to emit. The other (D1) trains the think-token controller that chooses whether to spend the extra computation at all. Both get a bounded exploration period to find the right design — with probes measuring how much referee-relevant information actually exists in the model's states — followed by exactly one locked, one-shot confirmation each. Meanwhile the paper gets written from everything already banked, which no new result can un-bank.

## 1. Program decision record

**Decision (Mark, 2026-07-31): the program does not retire.** The registered Stage A verdict (`none`) and its consequence (`transient_append_retires`) stand untouched in the ledger — they close *bridge-only linear transient append on the post-D0 substrate at the registered budget*, and nothing here renarrates that. By explicit program-level override, a new composite line (DC2, arbitrated append) opens under a new lock, and the deferred D1 line (utility-labeled in-place controller) activates. Three tracks run in parallel: DC2, D1, and manuscript consolidation. All existing pins not named here (persistent scratchpad, RG-12, GRAM/width, L above 1, retrieval-direction) remain pinned until the phase-2 verdicts are banked.

## 2. The unifying observation: measured oracle ceilings

From Stage A's immutable scoring cache (post-processing on saved arm predictions, labeled exploratory, no EVAL-C rescore): a per-position oracle choosing between the registered k=0 prediction and the trained-append prediction banks 7,799 helps and zero hurts — **u ≈ +3.91 points** over the 74.22 baseline. The audit's in-place oracle is **+5.61 points at 1.073 mean loops**. Both mechanisms therefore carry positive value gated behind the same missing function: arbitration. DC2 trains arbitration at the readout (which answer is emitted); D1 trains it at dispatch (whether extra compute is spent). This is one thesis with two independent tests, sharing label machinery. Confirmatory job for the coding agent, first thing: compute the exact composite oracle numbers (trained and untrained arms, and the cross-mechanism position overlap between append hurts and in-place hurts) from the existing caches, filed as exploratory receipts.

## 3. Methodology addition: the exploration/confirmation split (standing, both lines)

Each phase-2 line runs in two declared segments:

1. **Exploration window.** Dev-partitions only; explicitly unregistered; every number labeled exploratory; bounded by a GPU budget and a report-back cadence stated at the window's opening; architecture, objective, curriculum, and hyperparameters may all be iterated. No contact with any frozen evaluation slice. Nothing from the window is citable as evidence — its output is a *design*.
2. **Confirmation.** One registered run of the selected design under a locked preregistration with machine lock, fresh frozen evaluation slice, scripted verdict, bands with anti-degeneracy conditions. One shot, consumed on completion, exactly the Stage A machinery.

Rationale, recorded: every registered negative so far has been contaminated by a design variable (label, budget, architecture) that exploration would have caught. The split buys latitude without corrupting registration — the price is that exploration results prove nothing, and the documents say so.

**Curriculum principles binding on both lines** (distilled from the program's banked failures):

- *Safety before utility*: training stages order fallback/preservation first, selective improvement second. No objective ever again teaches "change your answer" without the ledger's cost of changing it wrongly.
- *Information-rate adequacy*: dense supervision (all valid positions per row), real batches, step counts sized so the training partition is seen multiple times. Undertraining may never again be a confound in a registered result.
- *Labels measured on the trained substrate* (standing D0 policy, unchanged).
- *Staged difficulty on label clarity* (T1-lite pattern): unambiguous cases first, widening on a stated schedule.
- *Anti-degeneracy bars*: every qualification band includes a condition that the trivial policy (gate never opens; controller never continues) cannot pass.
- *Instrumentation before interpretation* (added at draft 2): no training run, exploratory or registered, runs without the section 4.4 gradient-atlas telemetry. Stage A's record — 83 percent of logged gradients at the global clip, batch-1 loss noise, and no per-module norms — is the anti-pattern: we verified gradients were *correct* (RG-4) and *representable* (RG-11) but never measured whether they were *well-conditioned for learning*. A run whose gradients were not characterized cannot justify the next design.
- *Theory before registration* (added at draft 3, proposed as standing policy pending Mark's ratification): a first-principles audit precedes every future preregistration. The phase-1 record shows three registered negatives — the in-place harm ledger, Stage A's linear-bridge failure, and its starved supervision — were each predictable in advance from arguments now written down in THEORETICAL_FOUNDATIONS_AUDIT_20260731.md. The audit-then-register order converts hindsight into foresight at the cost of a document.

## 4. DC2 — arbitrated recurrence (draft 2: reference architecture is the bounded-correction sidecar)

**Question.** Can a trained, gated, anchored recurrent correction — arbitration made structural — deliver net-positive utility where replacement-style mechanisms could not?

**4.1 The interface invariant, and why the sidecar is the reference class.** The program's receipts now support one architectural generalization: every mechanism that *replaced* state was harm-dominated — in-place looping overwrites the position's state (net harmful on both checkpoints, parity ledger); transient append replaces the emission pathway with the slot readout (net harmful trained and untrained, DC0 and Stage A). The only structurally safe pattern is **preserved pretrained state plus a small, gated, learned correction**, and Mark's sidecar design (2026-07-31, from the attention/residual thread) makes that the invariant everywhere: an explicit identity highway through the inserted block; zero-initialized writeback (P_out = 0); scalar per-loop gates initialized near σ(−3) ≈ 0.05; and **anchored reentry** mixing the original h₀ back at every loop — the arbitration/fallback property diagnosed as missing after DC0, implemented structurally rather than hoped for from training. The design also dissolves the scale paradox in our probe record: RMS-matched feedback failed because it normalized a state that *replaced* the input, whereas the sidecar's update-normalized corrections (c ≈ 0.01–0.05 × RMS(h₀)) are safe because nothing is replaced. Unifying note: sequence-level loops with the anchor are vertical depth with the overwrite flaw removed; the separate low-dimensional scratchpad (4–16 slots, r ≪ d) with bounded cross-attention writeback is horizontal memory with the readout-replacement flaw removed; the multi-horizon speculative heads are the spec-dec deployment product itself. One object, both prior failure modes retro-fixed.

**4.2 Reference component list** (from Mark's design; each component enters only per the staging rule in 4.3; audit amendments A1–A2 binding): separate scratchpad state distinct from the residual stream, cross-attention read and bounded writeback; dual reentry (continuous plus anchored, ρ initialized 0.5–0.8, **capped at 0.9 in version one with the realized tube radius max_k‖h_k − h₀‖/RMS(h₀) logged — audit A1**, γ near 0); per-loop deep supervision with multi-horizon speculative heads (increasing loop weights with early supervision retained, geometrically declining horizon weights, starting point w = [0.1, 0.2, 0.3, 0.4], v = [1.0, 0.8, 0.6, 0.4]); feature-space targets against a stop-gradient teacher using normalized directional alignment plus a log-scale loss, multi-layer (EAGLE-3 lesson); margin-improvement loss across loops, **masked to positions where descent is possible — not already correct at loop k — so the Lyapunov constraint is never imposed on the infeasible (audit A2)** (KL-refinement as the gentle alternative, used sparingly); update normalization on both Δs and Δh; loop-state residual mixing with scalar softmax weights over [h₀ … h_k], initialized ≈ 0.5/0.5 on h₀ and h_k (short gradient paths across loops; recovery from a bad intermediate iteration); the **null-loop target** penalizing movement on low-teacher-entropy already-correct positions — the safety-before-utility principle as a loss term and the direct antidote to D0's reward-any-movement failure; staged unfreezing (bridge alignment → loop curriculum K 1→2→4 → local adaptation at 10–30× lower pretrained LR → optional LoRA in nearby upper projections). Per-module gradient clipping (scratchpad 1.0, bridge 0.5) replaces reliance on the global clip.

**4.3 Exploration stages, gated by telemetry rather than calendar.** Pre-window verification first (audit V1–V2, forward-only, dev material, before any training infrastructure is built): **V1, the expressivity check** — the margin distribution of oracle-help positions overlaid with the reachable-Δlogit bound Lip(F_{>L})·γM/(1−ρ), the Lipschitz gain measured by JVP on dev states, evaluated across c ∈ [0.01, 0.05]; V1 is the only computation that can reshape the design (its failure mode pre-names the remedy: a larger c bound, or E4's upper-layer LoRA as the Lipschitz-raising trigger). **V2, the block iteration-gain profile** — JVP gain distributions of the pretrained block at 1–4 iterates, quantifying the non-contraction account of the in-place harm ledger for the manuscript. Both land before E1 opens. Then E1: Mark's minimum-viable recipe at K = 1 — existing surgery cut retained for lineage continuity (insertion-point sweep deferred), token plus feature losses, full telemetry from step zero. E2: loop curriculum to K = 2 then 4, loop-state mixing, null-loop loss. E3: gate/anchor tuning and the margin-improvement loss, with the cheap G1 logit-blend readout gate (emit from α·logits_slot + (1−α)·logits_k0, α from scalar features, initialized closed) evaluated as the minimal comparator. E4: local adaptation (LoRA) only if E1–E3 plateau below the probe-reachable band. Rule against the co-debugging trap: every component addition is justified in the log by a named telemetry observation, never by enthusiasm. The information probes from draft 1 (cross-fitted state-to-label probes; reachable-utility curve under the oracle ceiling) run at E1 and steer the whole window — with their theoretical status upgraded by the audit (V3): probe discrimination is a Fano-style *ceiling* on any gate's utility, and the full-state-versus-scalar-feature comparison adjudicates D1's prior (that separability exists in the full state that scalar summaries destroy) in the same computation.

**4.4 The gradient atlas — the window's standing deliverable and the characterization the program has lacked.** Logged for every run: per-(module × objective × loop-index) gradient-norm trajectories, EMA-smoothed; loop-Jacobian gain distributions estimated by JVP probes (target: distribution centered near 1 in early training — persistently ≫ 1 flags exploding recurrence, ≪ 1 flags loops that erase gradient; the EMA audit's perturbation-compounding result is the vertical-axis precedent); clip-fraction per module (Stage A's 83 percent global-clip rate is the recorded anti-pattern); objective-balance ratios held within roughly one order of magnitude by EMA-adjusted loss weights (no GradNorm machinery in version one); per-objective signal-to-noise (batch gradient mean against gradient standard deviation); **inter-objective gradient cosines per module (audit A3** — magnitude balancing cannot fix direction conflicts; persistent cosine below zero between the token loss and the feature/null losses on bridge parameters reads as "the feature set cannot serve both masters," remedied by loss scheduling before any architecture change); and **the collapse metrics (audit A4)**: scratchpad effective rank (participation ratio of slot covariance), every gate's open-rate, and the A1 tube radius, with collapse thresholds pre-stated. Healthy ranges are pre-stated at window opening; the atlas is the evidence base from which the confirmatory design's constants are chosen, and it is the direct answer to the question "have we characterized how to learn from the scale and shape of the gradients" — after this window, yes, with receipts.

**4.5 Design decisions.** Feature-teacher source: **decided (Mark, 2026-07-31)** — frozen own-base Qwen hidden states are the geometric anchor for the feature loss, with the cached 7B tokens as the task signal; no cross-width projection of 7B features enters version one, per the teacher-ladder policy. Still held for Mark: speculative horizon J (proposed 4) and head parameterization; whether the DC2 confirmation registers the sidecar alone or sidecar-versus-best-G1 as arms.

**Confirmation.** DC2 preregistration locks the window-selected design; fresh EVAL-D slice (same sources, disjoint by hash); bands: **qualifies** = u > 0 with the Stage A CI machinery *and* helps ≥ a preregistered floor fraction of the measured probe-reachable utility (anti-degeneracy — a never-opening gate and a null-collapsed sidecar must not pass; floor held for markup, proposed 25 percent of the oracle helps); partial and negative bands with consequences stated at lock; and a preregistered compute accounting, since sidecar loops (scratch block plus read/write attention) cost far less than full-stack passes and the layer-equivalent arithmetic must be stated before any efficiency sentence is written.

## 5. D1 — utility-labeled in-place controller

**Question.** Can the control-token gate, trained on measured utility labels with an explicit compute cost, capture a meaningful fraction of the in-place oracle (+5.61 at 1.073 loops)?

**Design commitments** (from the standing D1 pre-commitments, updated by phase-1 evidence): substrate post-D0 EMA (better forced-2 ledger than pre-D0 per the parity result, T1 machinery intact, preservation cost erased); targets are RTG continue/stop labels built from forced-depth ledgers measured on this checkpoint, with registered per-loop cost λ chosen from the frontier; class weights measured, never defaulted; optional 14B-referee exclusion of teacher-noise positions decided at lock; policy-level guardrail on baseline-accepted positions; deployment-pair success framing (net agreement and spec-dec acceptance at matched loop cost), R descriptive.

**Exploration window contents.** Label construction and audit (class balance, stratum split, λ frontier); a pilot grid over class weights and curriculum staging (the P0 pattern); curriculum candidate: stage on label clarity — unambiguous stops and highest-margin continues first, widening by margin quantile on a stated schedule; rehearsal mix retained throughout. The rung-0 negative binds on cheap *post-hoc scalar* routing — the trained token pathway is a different class (the T1 precedent: frozen pooled-head 9.1 percent versus trained pathway 100 percent on the same substrate), and that distinction is the line's explicit bet, stated here so the result adjudicates it.

**Confirmation.** D1 preregistration, fresh EVAL-E slice, one registered run, anti-degeneracy: the never-continue policy (identical to baseline) cannot qualify; qualification requires positive net utility at a stated minimum continue rate or better, bands fixed at lock.

## 6. Sequencing, budget, and partitions

DC2's probe battery and oracle receipts first (no GPU beyond dev forward passes; steers both lines), D1 label construction in parallel (small GPU), then both exploration windows run concurrently with report-backs; confirmations lock independently whenever their windows produce a design. Fresh slices EVAL-D and EVAL-E generated under the standing partition discipline (disjoint by hash from all 2,218+ prior documents, manifests hashed, read-once). DEV partitions reusable within each window, never across a confirmation boundary. Manuscript consolidation does not wait on any of this.

## 7. Manuscript track (proceeds now, on banked results only)

Answers to the Stage A handoff's section 10, adopted: Stage A enters the **main narrative**, paired with T1-lite as the two-sided interface result; the synthesis is the handoff's own, with scope — *explicit control channels can causally select computation; raw latent-state injection resists preservation; both claims bounded to their tested interfaces*. The composite branch is closed **for the current manuscript** — phase-2 results decide their own placement later and nothing in the paper waits on them. The code/general asymmetry is reported descriptively (replicated across three partitions). One additional exploratory analysis is requested for the mechanism discussion: the section 2 cross-mechanism hurt-position overlap. Consolidation order: Arm G section (queued since July), methods/substrate/lineage, T1-lite–through–Stage A causal-control arc, limitations (single-seed policy stated plainly).

## 8. Boundaries

Nothing trains until a phase-2 preregistration locks. EVAL-B and EVAL-C stay spent; passive Stage A checkpoints are archival and are never promoted or mined. Exploration-window numbers are never cited as evidence, never enter receipts as findings, and never touch a frozen slice. Stage A's verdict and every do-not-claim from its handoff remain in force verbatim, including: the bridge learned but did not make append safe; no near-miss language. All other pins unchanged. The override in section 1 is the only pin lifted by this document.

## 9. Held for Mark's markup — each question with analysis and recommendation

### 9.1 DC2 anti-degeneracy floor

*In plain terms:* the success rule must stop a "coward" system from passing. A gate that never opens, or a sidecar whose null-loop loss collapses it to doing nothing, scores u ≈ 0 — technically non-negative, actually useless. The floor says: to qualify, the system must also capture at least some stated fraction of the helps a perfect referee would have banked.

*Analysis:* the floor trades off two failure modes. Too low (say 5 percent) and a nearly-inert system qualifies, handing the future policy stage an actuator with nothing worth routing. Too high (say 75 percent) and we demand near-oracle selectivity from a first confirmatory run, risking a fail verdict on a genuinely useful mechanism — repeating the T1-lite pattern of gates stricter than the science required. A subtlety: the oracle itself may be partly unlearnable (some helps may be indistinguishable from hurts given the information in the states), which is exactly what the E1 probe battery measures. A floor set against perfection punishes the mechanism for the probe's ceiling.

*Options:* (a) 25 percent of oracle helps, fixed now — simple, but blind to learnability; (b) 50 percent of the *probe-reachable* utility measured in E1 — self-calibrating to what is demonstrably learnable, but not settable until the probes land; (c) the lesser of the two — protects against both an unreachable oracle and an overly-generous probe.

*Recommendation:* **option (c)** — the DC2 preregistration locks the floor as min(25 percent of oracle helps, 50 percent of probe-reachable helps), computed from exploration-window receipts before any EVAL-D contact. *Rationale:* the floor should test selectivity against what the window proved learnable, not against perfection; both inputs are measured before the lock, so no degree of freedom survives to adjudication time.

### 9.2 Exploration-window budgets and report-back cadence

*In plain terms:* how much GPU time the free-experimentation period gets, and how often results come back for review, before we commit to the one-shot confirmation.

*Analysis:* too small a budget re-creates Stage A's central flaw — a negative that might just be undertraining. Too open-ended invites unfalsifiable tinkering and drift. Anchors from receipts: Stage A's entire training run cost 5.7 minutes; a properly-sized E1 run (dense supervision, real batches, the partition seen several times) is roughly 2 to 6 A100-hours; the window needs perhaps five to ten runs across E1–E4. Calendar-based cadence fits badly because stages gate on telemetry, not dates.

*Options:* (a) GPU-hour cap only; (b) report-back count only; (c) both, with reports at stage boundaries.

*Recommendation:* **option (c)** — approximately 40 A100-hours for the DC2 window and 20 for the D1 pilot grid, with a report-back at each stage boundary (E1, E2, E3/E4 for DC2; label-audit and pilot-grid for D1), each report a continue/stop decision point for you, and early closure whenever telemetry says converged or dead. Exact caps confirmed against the coding agent's resource note at window opening. *Rationale:* caps keep Colab economics bounded; stage-boundary cadence matches the design's own structure; and every report-back is a place to kill a failing line cheaply.

### 9.3 The 14B-referee exclusion for D1's labels

*In plain terms:* about one in six of the 7B's "the drafter was wrong here" judgments is contradicted by the 14B, which endorses the drafter's token. Should those probable-noise positions be excluded from the continue class D1 trains on, and when do we decide?

*Analysis:* including them teaches the controller to spend compute chasing targets that may not be errors at all — a milder cousin of D0's label misalignment. Excluding them costs a 14B scoring pass over the label partition (post-processing where the cache covers it, one cached pass otherwise) and shrinks the already-small continue class further, which interacts with class weighting — the exact trap P0 measured its way out of. Deciding now, either way, means building labels once but guessing; deciding at lock lets the measured class statistics decide.

*Options:* (a) exclude, decided now; (b) include, decided now; (c) build both label variants in the exploration window, compare class balance, label-noise statistics, and pilot-grid behavior, choose at lock.

*Recommendation:* **option (c)**. *Rationale:* this is precisely the kind of design variable exploration windows exist to resolve; the marginal cost is one extra labeling pass; and the P0 lesson — measure, never default — applies with force to anything touching class balance.

### 9.4 EVAL-D and EVAL-E sizes

*In plain terms:* how big the two fresh, one-shot test sets should be.

*Analysis:* Stage A's 0.2M-token slice gave 199,532 scored positions and a bootstrap interval of roughly ±0.5 points — comfortably sharp for a u ≥ 0 bar and for 50-percent ratio comparisons. Doubling to 0.5M narrows intervals by only ~40 percent while increasing teacher-caching cost and evaluation time; halving to 0.1M risks genuine ambiguity when a result lands near a band boundary, which is the one place resolution matters.

*Options:* 0.1M (cheap, risky at boundaries); 0.2M (house pattern, demonstrated adequate); 0.5M (sharper, costlier).

*Recommendation:* **0.2M each**, generated together in one data-preparation pass (with the frozen own-base feature caching for the sidecar's feature loss amortized into the same pass), frozen separately, disjoint by hash from all prior documents. *Rationale:* the power question is not hypothetical — Stage A's intervals empirically demonstrate 0.2M suffices for our bands.

### 9.5 Speculative horizon J and head parameterization

*In plain terms:* how many future tokens the model drafts per thinking episode, and what the drafting heads look like architecturally.

*Analysis:* the deployment target is speculative decoding, and our banked D0 simulations used draft lengths γ of 2, 4, and 8. Value per extra horizon declines geometrically (the design's v_j weights encode this), and a 0.5B drafter's acceptance beyond a few tokens collapses, so long horizons buy mostly noise and parameters. For the heads: Medusa-style independent per-horizon MLP heads are simplest — parallel, one clean loss term each, matching the v_j weighting exactly; EAGLE-style light autoregressive heads accept better but add sequential machinery and a second recurrence to debug; a shared trunk with horizon embeddings is parameter-cheap but couples the horizons' gradients, which muddies the atlas.

*Options:* J ∈ {2, 4, 8} × {Medusa-independent, EAGLE-autoregressive, shared-trunk}.

*Recommendation:* **J = 4 with Medusa-style independent single-layer heads** in version one; a head-class upgrade only if the gradient atlas shows head gradients starving or horizon losses failing to separate. *Rationale:* matches the middle of the banked γ grid, keeps one loss term per horizon for clean gradient attribution, and defers complexity to a telemetry-justified addition — the staging rule applied to heads.

### 9.6 DC2 confirmation arms

*In plain terms:* does the one-shot final test run only the sidecar, or also the simplest possible gate (the logit-blend G1) as a registered comparison?

*Analysis:* adding G1 costs almost nothing — the same evaluation pass scores it, exactly as Stage A scored five arms. What it buys is attribution: if the sidecar qualifies and G1 does not, the architecture's complexity demonstrably paid for itself; if both qualify, the simple gate becomes the preferred deployment object and the paper says so; if neither, the negative is not attributable to sidecar complexity alone. The one risk is adjudication ambiguity with two qualifying arms, which pre-stated precedence removes.

*Options:* (a) sidecar alone — cleaner narrative, weaker attribution; (b) both, sidecar primary and G1 secondary with identical bands and pre-stated precedence.

*Recommendation:* **option (b)**. *Rationale:* attribution is cheap here and the program has been burned before by results whose cause could not be isolated; the multi-arm single-pass discipline is already proven machinery.

### 9.7 Sidecar insertion point

*In plain terms:* where in the layer stack the sidecar taps in and writes back — keep the existing surgical cut, or search for a better split.

*Analysis:* the sidecar design says "middle third, not exactly the center by assumption," which is a fair challenge to our layer-6/layer-18 cut. But the cut is where every banked contract lives: the RG battery, the identity assertions, the checkpoint lineage, and all cross-phase comparability. Moving it in version one would change two things at once — architecture *and* location — making any E1 result unattributable, which is the co-debugging trap by another name. Meanwhile the E1 probe battery already sweeps probe placement per layer, so it produces exactly the evidence a later insertion sweep would need, for free.

*Options:* (a) existing cut for version one, insertion sweep deferred and probe-guided; (b) small insertion sweep inside the window (2–3 candidate cuts), doubling E1 cost.

*Recommendation:* **option (a)**. *Rationale:* one variable at a time; lineage continuity preserves every assertion and comparison we own; and the probe battery converts the deferred sweep from blind search into a data-guided decision.

### Decided ledger

- Feature-teacher source (Mark, 2026-07-31, draft 2 markup): frozen own-base Qwen hiddens as the geometric anchor, cached 7B tokens as the task signal (section 4.5).
