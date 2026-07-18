# Phase G Forced-Injection Causal Probe

**Status:** preregistered and implementation-complete before evaluation  
**Source:** `stage5_phase_g_multitarget_control_20260718`

## Question

The A0 posterior and prior behaved almost identically despite separated latent
head statistics and a nonzero posterior residual. This probe asks one narrow
causal question: can that learned posterior residual control the terminal
prediction if its magnitude is increased at inference?

This distinguishes a magnitude-limited additive channel from an additive
re-entry route that does not control the output on this substrate.

## Frozen design

- Evaluation only. No optimizer, backward pass, checkpoint mutation, or new
  training is permitted.
- Use both preserved A0 EMA checkpoints: KL `0.001` and the one permitted
  confirmation at KL `0.0001`.
- Use the exact held-out A0 control surface: 106 target variants in 32 repeated
  prompt groups.
- Reuse each row's published A0 trajectory seed.
- Multiply only the learned posterior residual at the high-level re-entry
  point by factors `1, 3, 10, 30, 100`.
- Record switching groups, selected-target fidelity, and K=1 validity at every
  factor.
- Factor `1` must reproduce every published A0 posterior K=1 prediction.
- The frozen deterministic lineage hash must be unchanged after evaluation.

## Locked readings

`CHANNEL-EXISTS`:

- at least 16 of 32 groups switch at any factor; and
- K=1 validity at that factor remains strictly above 0.50.

This authorizes a new preregistered successor using the same additive route
with a larger trained scale and preservation guardrails.

`NO-CHANNEL`:

- switching remains below 8 of 32 groups at every factor; or
- switching reaches at least 16 groups only where validity is strictly below
  0.50.

This closes additive re-entry injection. Any future successor must change
where conditioning enters, beginning with a FiLM-style conditioned route.

`AMBIGUOUS`:

- every intermediate outcome.

Ambiguous results are reported as measured and default to no authorization.

## Consequences

No outcome from this probe directly opens coverage, selection, learned
halting, LPRM, particles, or SVGD. A successor is authorized only by
`CHANNEL-EXISTS` and must add an auxiliary per-loop branch-choice loss. The
branching repeated-target task and gate order remain unchanged:

1. posterior control;
2. preservation;
3. coverage;
4. selection or particle mechanisms.

## Do-not-claims

This probe cannot falsify GRAM, stochastic recurrent width, or stochastic
reasoning generally. It tests only whether the preserved A0 posterior residual
has a magnitude-responsive additive route to terminal selection on this
retrofitted recurrent Qwen substrate.
