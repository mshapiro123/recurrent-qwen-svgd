# Program Track: Compact Frontier-Reasoning Recurrent Particles

## Purpose

This is the orientation document above the kernel-geometry and
spectrum-to-signal handoffs. It keeps implementation choices pointed at one
destination: a compact, owned model that performs strongly on verifiable
reasoning, with clean evidence for whether recurrent particles add value beyond
the training recipe alone.

## Destination

The target is a compact model that reaches frontier-level performance on
verifiable reasoning: mathematics, competitive code, STEM, and ARC-style exact
reasoning. The reference class is a VibeThinker-style small model, where a
training recipe rather than scale pushes a 1.5B-3B model into a much stronger
reasoning band.

## Proven Path And Project Bet

The proven path is the spectrum-to-signal recipe:

1. diversity-exploring distillation;
2. MaxEnt-guided or related verifiable reinforcement;
3. self-distillation;
4. claim-level test-time selection.

The recurrent-particle architecture is the project contribution and the bet. It
must prove that structural diversity in the computation survives later
reinforcement better than diversity encoded only in weights. A strong final
model is not enough; the decisive experiment must isolate whether architecture
adds hard-tail accuracy beyond the same recipe on a standard dense model.

There are two legitimate ordering modes:

- **Destination-primary:** prioritize the proven spectrum-to-signal recipe and
  layer recurrent particles on once base capability exists.
- **Contribution-primary:** prioritize mechanism evidence that architectural
  diversity is a real lever, accepting a slower path to a deployable artifact.

The working order is the hybrid: de-risk the mechanism cheaply, then scale
through the recipe only if recurrent particles show selector-convertible value.
When this fork affects a concrete decision, surface it explicitly rather than
silently assuming one branch.

Implementation note: Stage 5 now includes a dense standard-control SFT runner,
`colab/run_stage5_arc_agi_dense_sft.py`, plus dense base-mode LoRA loading in
`eval/eval_arc_agi.py`. This lets the project train an unmodified Qwen LoRA
adapter on the same ARC-AGI rows used for recurrent SFT, then route the planner
to a matched recurrent SFT arm. This is the first concrete scaffold for the
standard-vs-recurrent same-recipe comparison. Dense-control summaries include
paired comparison artifacts, so aggregate deltas are not used as the only
evidence.

Stage 5 also includes `colab/assess_stage5_recipe_control.py`, the explicit
same-recipe architecture gate. It reads the dense-control summary and matched
recurrent SFT summary, verifies that both arms use the same base model,
parameter count, ARC split, and recipe metadata, and asks whether recurrent
selected-answer accuracy beats the dense control in the hard bucket without
aggregate selected-answer harm. This is the local form of the decisive
experiment before larger recipe or scale-up runs. It also separates the
selector-conversion case: if recurrent improves hard-bucket best-of-K candidate
coverage without selected-answer conversion, the next move is selector/verifier
work on those candidates, not declaring the architecture dead. Selector
conversion itself must also convert into hard-bucket selected-answer lift; a
selector that only improves the aggregate is diagnostic evidence, not a pass.

## Decisive Experiment

Train both a standard version and a recurrent-particle version of the same
capable small base through the same spectrum-to-signal recipe. Add the same
claim-level selector to both. Test whether the recurrent version wins on the
hardest difficulty stratum, where diverse candidate sets should matter most.

This is the thesis experiment:

- structural diversity should survive the signal phase;
- the selector should convert that diversity into accuracy;
- the lift should concentrate on hard-tail problems. Aggregate-only lift is a
  diagnostic, not enough to pass the architecture gate.

The architecture earns its place in the hard-problem tail. Test-time scaling
matters most when single-answer accuracy is low, candidate diversity is real,
and the selector can extract a correct claim from several plausible paths.
Broad aggregate gains are useful, but the project claim depends on hard-tail
lift under a matched recipe.

## Near-Term Gates

### Gate 1: Measurement Before More Mechanism

Before more kernel geometry work, build the selector and a larger
difficulty-stratified evaluation suite. The selector metric defines what
``better`` means.

Required pieces:

- majority/self-consistency voting over saved candidate records;
- reliability-weighted voting over candidate claims;
- paired sign tests on selected-answer metrics;
- task-family or difficulty-stratified reporting;
- held-out checks for any calibrated projection or selector setting.
- a standard dense recipe-control arm whenever a recurrent training recipe is
  being evaluated for architecture lift.

### Gate 2: Mechanism Before Scaling

Only scale or begin reinforcement after the recurrent-particle mechanism shows
useful diversity that a selector converts to accuracy:

- non-negative selected-answer lift versus deterministic recurrent baseline;
- helped examples at least equal harmed examples;
- diversity that improves exact candidate coverage rather than only noise;
- held-out confirmation of the selected mechanism.

Implementation note: Stage 5 now has an explicit Gate 2 assessment artifact,
`colab/assess_stage5_gate2.py`, for recovery-particle summaries. It separates
replicated selected-answer lift from the weaker case where particles improve
best-of-K coverage but the selector does not convert that coverage into selected
accuracy. The next-action planner runs this assessment before treating particle
evidence as a scaling signal.

## Base Model Decision

Qwen 0.5B is likely below the floor for frontier verifiable reasoning. It is
still useful for surgery, identity, recurrence, and measurement plumbing. The
decisive experiment probably needs at least a 1.5B-3B base.

Prefer a base or instruct checkpoint that has not already had diversity
collapsed by reinforcement. Starting from an already heavily RL-trained
reasoning model can confound the survival-of-diversity question.

## Priority Order

1. Build measurement before adding mechanism when effects are small.
2. Judge comparisons with paired sign tests on selector-relevant metrics,
   stratified by difficulty or task family.
3. Keep standard-vs-recurrent under the same recipe as the decisive comparison.
4. Treat kernel geometry as a mechanism de-risking step, not the destination.
5. Scale only after Gate 1 and Gate 2 give a real signal.
6. Treat same-recipe recurrent-vs-dense assessment as required evidence before
   attributing ARC gains to architecture.

## Anti-Patterns

- Chasing more kernel variants before the selector and stratified eval exist.
- Scaling before the recurrent mechanism shows selected-answer lift.
- Treating oracle best-of-K as deployable performance.
- Treating a strong final model as success without isolating whether the
  architecture or the recipe produced it.
- Using derivative or repackaged checkpoints when a primary source model,
  teacher, or verifier is available.

## Current Implementation Link

As of this note, ARC candidate rescoring supports:

- `heuristic`;
- `self_consistency`;
- `reliability_vote`;
- `symbolic_priority`.

`reliability_vote` is the first target-free reliability-weighted selector for
Gate 1. It weights identical parsed-grid claims by candidate provenance,
program verification on demonstrations, symbolic source, shape consistency, and
source diversity. It is a selector baseline, not a final verifier.
