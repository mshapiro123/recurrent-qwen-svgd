# Program Track: Master Sequence

> **Current amendment:** Phase G reunifies the deterministic substrate and guided stochastic-width tracks. The abductive-injective gate now directly precedes frozen-block G-alpha; LPRM, adaptive halting, and SVGD remain downstream of a coverage win. See [PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md](PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md).

## 0. Purpose

This is the umbrella document for the recurrent-Qwen program. It updates the earlier program-track summary by ordering the whole effort into four phases by dependency, marking the current step at the head of the chain, and writing the gates between phases explicitly so the long-term arc and the place of the present re-entry work within it are visible on one page. Each phase is governed by its own handoff; this document is the sequence and the index, not a replacement for them. The map from phases to handoffs is in the last section.

## 1. Destination and the thesis under test

The destination is a compact model that reaches hard reasoning by trained latent depth, and where the task genuinely admits more than one valid approach, by multiple latent pathways, rather than by scale. The central thesis is narrower than that and is what the program actually tests: that recurrence can substitute for part of what scale buys, specifically the depth of sequential composition that a deeper stack provides, and not the width, the parallel features, or the stored knowledge that more parameters provide. That boundary is not a caveat, it is the design principle, because it determines which problems are worth training on and which gains would count as evidence. The decisive test of the thesis lives in Phase 1.

Where the program stands today, in one paragraph. The surgery is sound: the single-pass identity reproduces the base model exactly, so the regression is not architecture damage. A real depth signal exists on the harder ARC-Challenge content, but easy-item preservation is unsolved and the trustworthy debiased surface does not yet register the depth gains. Inference-time particle noise was a definitive negative: it produces no correct-bearing diversity, and the loop budget is ignored without training pressure, so both depth and breadth are training-time properties this checkpoint does not yet have. The dynamics are expansive rather than multistable, and the most recent analysis identifies the unreconciled loop closure as a prime suspect for both that expansion and the absent depth benefit. That is why the head of the chain is the re-entry fix.

## 2. Why the phases are ordered, not parallel

These were never parallel workstreams, though they felt like it while their structure was still being discovered. They are a dependency chain. The loop closure sits at the root, because depth cannot convert if the block computes on off-distribution input every iteration, so the depth adapter and the depth curriculum are operating on unusable input until the closure is fixed. Breadth sits above depth, because the effective-pathway diagnostic run on the current map measures closure-driven expansion rather than basin structure, so breadth cannot be assessed until the dynamics are bounded. Particles sit above breadth, because repulsion and selection have nothing to convert until there is correct-bearing diversity to work on. The sense of stepping back to fix re-entry is the cost of having found the root later than we would have liked, not a detour from the project. Fixing it now is doing the project in the only order that yields interpretable results.

## 3. Standing instruments, used in every phase

Four instruments are constant across the phases so that each phase is read with the same measuring tools rather than new ones.

The debiased multiple-choice surface, content scoring together with cyclic permutation, is the selection gate, not the content surface alone, because optimizing the surface that is easiest to move while the trustworthy surface stays flat is how a checkpoint is trained to look better only under the scoring that flatters it.

The Leinster similarity-sensitive diversity, computed with the model's kernel as the similarity matrix and read per trajectory rather than pooled, is the breadth metric throughout Phases 2 and 3. It is verified to recover the effective number of distinct pathways and to discount near-duplicates.

The standard-Qwen-same-curriculum control arm runs whenever a training comparison is made, so that any hard-tail gain is attributable to the architecture rather than to the data.

Expected value of perfect information is flagged as a standing discipline: when continued refinement or debate has passed the point of diminishing returns, that is surfaced as a flag, and the call to continue is the human's.

## 4. The sequence

### Phase 0, loop-closure re-entry. Current. Head of the chain.

Reconcile the recirculated state with the block's expected input. The work is the code read first, then the drift measurement, then the re-entry module that combines input injection, a renorm, and a targeted low-rank state map, then the disentangling diagnostic. Governed by the re-entry handoff.

In parallel, on the side that needs no GPU, prepare the depth-convertible curriculum: the capability ladder and constructed step-count problems, filtered by the chain-of-thought-rescue test so that only depth-shaped problems enter. This runs concurrently with the re-entry code so the curriculum is ready the moment the loop is fixed, which is what keeps the depth thread from starting cold.

The seam into breadth lives inside this phase. The disentangling diagnostic, a renormalization inserted on the loop path followed by a rerun of the spread measurement, is simultaneously the validation of the re-entry fix and the first real breadth measurement, so you leave Phase 0 already holding the answer to the first question Phase 2 would otherwise have to ask.

Gate to Phase 1: the single-pass identity still holds with the module inactive at one iteration, and the per-loop norm is bounded after the renorm. Off-ramp: if the code read shows the bridge already performs part of this, extend it rather than rebuild.

### Phase 1, depth. The decisive phase.

With the loop closing in-distribution, the depth-conditioned adapter finally has usable input to act on, and the depth-routing SFT trains the halting and the conditioning together against the ladder labels on the prepared curriculum. Read on the debiased surface, with the control arm, the test is whether trained depth converts the depth-shaped failures while preserving the easy items, which is the direct test of whether recurrence substitutes for scale. Governed by the wide-and-deep curriculum handoff and the depth-conditioned LoRA design.

Gate to Phase 2: depth converts depth-shaped failures while the easy items hold, and the gain survives the control, so it is attributable to the architecture. Off-ramp, and the cleanest in the program: if depth does not convert even with the loop fixed, the conditioning in place, and the data clean, then the limit is the block's single-pass capacity at this scale, and the fork is the scale decision rather than more architecture.

### Phase 2, breadth and multistability.

Rerun the effective-pathway diagnostic, now on the depth-trained, loop-fixed model rather than the current expansive one, so that for the first time it measures whether the dynamics support multiple basins rather than measuring closure expansion. If Phase 0 already collapsed the spread, you enter with the expansion handled and the only question is whether basins exist. If basins are present, the kernel and selector work is the right focus. If the effective count is still near one, regime shaping is what remains, informed by whether Phase 0 showed the residual instability to be magnitude or direction. Breadth is exercised on the multi-solution tasks the wide curriculum isolates, not on single-answer arithmetic. Governed by the breadth-mechanism handoff.

Gate to Phase 3: the effective pathway count is above one with correct-bearing diversity. Off-ramp: if basins are absent at this scale, that is a result to report rather than to force.

### Phase 3, particles, SVGD, and the selector.

This is where the kernel geometry finally belongs, as a soft regularizer toward the maximum-entropy-over-valid-pathways target on the identity coordinate rather than as the load-bearing force, with method-anchored supervision pinning the modes so collapse is structurally hard, the claim-level selector converting the surviving diversity into accuracy, and conditional invariance folded into the reinforcement phase as the nuisance-collapsing objective rather than run as a separate pass. Governed by the kernel-geometry and conditional-invariance handoffs, with the spectrum-to-signal recipe as the staging overlay.

Gate to a performance claim: the model beats base on the hard strata, the easy items are preserved, and the improvement survives held-out prompts and debiased scoring, which is the benchmark gate that precedes any public checkpoint or paper. Off-ramp: if diversity does not convert, pause the particle work and continue with deterministic depth.

### Standing alongside all phases, the scale probe.

The no-training identity and loop-preservation check at one and a half billion runs as cheap information about whether the ceiling is real, and it stays information. It becomes a commitment to move development only when one of the gates above sends you there, most likely the Phase 1 off-ramp.

## 5. The gates as off-ramps

The seams are decision points that can end or redirect the program, which is what keeps it honest and keeps you from being committed past the evidence. The four decisive forks, in order. After Phase 0, the spread either collapses under renormalization or it does not, which routes the breadth work and may dissolve much of it. After Phase 1, depth either converts or it does not, and the latter sends you to the scale decision rather than to more architecture. After Phase 2, basins either exist at this scale or they do not. At the benchmark gate, the model either beats base under fair scoring or it does not, and only passing it justifies a public claim. None of these is forced; each is a place the evidence is allowed to stop or turn the program.

## 6. Where the value is now

The binding uncertainty has shifted from architectural to empirical. The program is seven handoffs deep in design, and the single most informative thing available is not another design pass, it is running Phase 0, because the re-entry diagnostic will say more about whether the whole approach is viable than further planning will, and several open questions, including how much of the breadth problem is real, are waiting on that one run. The expected value of more design before Phase 0 is low and falling; the expected value of the Phase 0 run is high. The recommendation is to move to execution now and let the result, rather than more analysis, set the agenda for depth and breadth. The call to keep refining is yours; this is the flag, not a veto.

## 7. Map of handoffs to phases

- Phase 0, re-entry: the loop-closure re-entry handoff, with the depth-conditioned LoRA design for the architecture it sits beside.
- Phase 0 parallel, data: the curriculum data pipeline and the wide-and-deep curriculum handoff, for the depth-convertible problem set and the chain-of-thought-rescue filter.
- Phase 1, depth: the wide-and-deep curriculum handoff and the depth-conditioned LoRA design.
- Phase 2, breadth: the breadth-mechanism handoff, with its effective-pathway diagnostic and regime reading.
- Phase 3, particles: the kernel-geometry handoff and the conditional-invariance handoff, with the spectrum-to-signal recurrent handoff as the staging overlay across Phases 1 through 3.
- Standing: the scale probe as cheap information; the debiased surface, the Leinster diversity metric, the control arm, and the expected-value-of-information flag as instruments in every phase.
