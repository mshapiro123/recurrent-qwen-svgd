# Program Track: Master Sequence

## 0. Purpose

This is the umbrella document for the recurrent-Qwen program. It updates the
earlier program-track summary by ordering the whole effort into four phases by
dependency, marking the current step at the head of the chain, and writing the
gates between phases explicitly so the long-term arc and the place of the
present re-entry work within it are visible on one page. Each phase is governed
by its own handoff; this document is the sequence and the index, not a
replacement for them.

## 1. Destination and Thesis

The destination is a compact model that reaches hard reasoning by trained
latent depth, and where the task genuinely admits more than one valid approach,
by multiple latent pathways, rather than by scale. The thesis under test is
narrower: recurrence can substitute for part of what scale buys, specifically
the depth of sequential composition that a deeper stack provides, not the
width, parallel features, or stored knowledge that more parameters provide.

That boundary determines which problems are worth training on and which gains
count as evidence. The decisive test of the thesis lives in Phase 1.

Current standing:

- single-pass identity is exact, so regression is not caused by identity-path
  architecture damage;
- a depth signal exists on harder ARC-Challenge content, but easy-item
  preservation is not solved;
- trustworthy debiased scoring does not yet register the full depth gains;
- inference-time particle noise was negative: it creates no correct-bearing
  diversity on the current checkpoint;
- the loop budget is ignored without training pressure, so depth and breadth
  are training-time properties this checkpoint does not yet have;
- the dynamics are expansive rather than multistable;
- unreconciled loop closure is the current root suspect.

Therefore the head of the chain is the re-entry fix.

## 2. Why The Phases Are Ordered

These workstreams are a dependency chain, not parallel tracks.

Loop closure sits at the root. Depth cannot convert if the recurrent block is
fed off-distribution loop input. The depth adapter and depth curriculum operate
on unusable input until closure is fixed.

Breadth sits above depth. The current effective-pathway diagnostic measures
closure-driven expansion rather than basin structure, so breadth cannot be read
cleanly until dynamics are bounded.

Particles sit above breadth. SVGD, repulsion, and selection have nothing to
convert until there is correct-bearing diversity.

Stepping back to re-entry repair is not a detour; it is the dependency order
that makes later results interpretable.

## 3. Standing Instruments

Use the same instruments across phases:

- debiased multiple-choice scoring: content scoring plus cyclic permutation;
- Leinster similarity-sensitive diversity, using the model's kernel per
  trajectory rather than pooled;
- standard-Qwen same-curriculum control arm for training comparisons;
- expected value of perfect information as a standing discipline: when more
  refinement has low value, surface that explicitly.

## 4. Sequence

### Phase 0: Loop-Closure Re-entry

Current phase. Reconcile the recirculated state with the recurrent block's
expected input. The sequence is:

1. code read;
2. drift measurement;
3. re-entry module combining input injection, renorm, and targeted state map;
4. disentangling diagnostic.

In parallel, prepare the depth-convertible curriculum: capability ladder and
constructed step-count problems filtered by chain-of-thought rescue. This keeps
the depth thread ready when loop closure is fixed.

The seam into breadth also lives here. The renorm disentangling diagnostic is
both validation of re-entry and the first clean breadth measurement.

Gate to Phase 1:

- single-pass identity still holds when the module is inactive at one
  iteration;
- per-loop norm is bounded after renorm;
- loop-closure path is gradient-live and has passed a tiny repair smoke.

Off-ramp: if the code read shows the bridge already performs part of this,
extend it rather than rebuilding.

### Phase 1: Depth

This is the decisive phase. With loop closure in-distribution, train the
depth-conditioned adapter and halting/conditioning path against ladder labels
on the prepared curriculum.

Read the result on the debiased surface, with the standard-Qwen same-curriculum
control arm. The question is whether trained recurrence converts
depth-shaped failures while preserving easy items. This directly tests whether
recurrence substitutes for scale on sequential composition.

Gate to Phase 2:

- depth converts depth-shaped failures;
- easy items hold;
- gains survive the standard-Qwen control.

Off-ramp: if depth does not convert with loop closure fixed, conditioning in
place, and data clean, the limit may be the block's single-pass capacity at
this scale. That routes to the scale decision, not more architecture.

### Phase 2: Breadth and Multistability

Rerun the effective-pathway diagnostic on the depth-trained, loop-fixed model.
Now it can measure basin structure rather than closure expansion.

If basins exist, focus on kernel and selector work. If effective count is still
near one, move to regime shaping, informed by whether Phase 0 showed residual
instability to be magnitude or direction.

Exercise breadth on multi-solution tasks isolated by the wide curriculum, not
single-answer arithmetic.

Gate to Phase 3:

- effective pathway count is above one;
- diversity is correct-bearing.

Off-ramp: if basins are absent at this scale, report that result instead of
forcing particle work.

### Phase 3: Particles, SVGD, and Selector

SVGD and related kernels belong here: as soft regularizers toward
maximum-entropy-over-valid-pathways, not as the load-bearing mechanism.

Method-anchored supervision should pin modes so collapse is structurally hard.
The claim-level selector converts surviving diversity into accuracy.
Conditional invariance enters reinforcement as the nuisance-collapsing
objective.

Gate to a performance claim:

- model beats base on hard strata;
- easy items are preserved;
- improvement survives held-out prompts and debiased scoring.

Off-ramp: if diversity does not convert, pause particle work and continue with
deterministic depth.

## 5. Standing Scale Probe

The no-training identity and loop-preservation check at 1.5B runs as cheap
information about whether the ceiling is real. It remains information, not a
commitment to scale, unless a phase gate sends us there, most likely the Phase
1 off-ramp.

## 6. Current Value Judgment

The binding uncertainty has shifted from architectural design to empirical
execution. The most informative action is to complete Phase 0. More design has
falling expected value until re-entry results land.

Run Phase 0, read it, and let the result set the agenda for depth and breadth.

## 7. Handoff Map

- Phase 0 re-entry: loop-closure re-entry handoff, plus depth-conditioned LoRA
  design.
- Phase 0 parallel data: curriculum data pipeline and wide/deep curriculum
  handoff.
- Phase 1 depth: wide/deep curriculum handoff and depth-conditioned LoRA
  design.
- Phase 2 breadth: breadth-mechanism handoff and effective-pathway diagnostic.
- Phase 3 particles: kernel-geometry and conditional-invariance handoffs, with
  spectrum-to-signal as the staging overlay.
- Standing: scale probe, debiased surface, Leinster diversity metric, control
  arm, and expected-value-of-information flag.
