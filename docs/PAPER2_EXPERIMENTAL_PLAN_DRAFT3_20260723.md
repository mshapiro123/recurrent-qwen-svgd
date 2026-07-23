# Paper Two Experimental Plan - Draft 3 Pivot

**Date:** 2026-07-23
**Status:** program amendment; no new training authorization

## Program Goal

The positive-seeking program now targets a useful model trained on natural
reasoning traces and depth labels derived from speculative-decoding behavior.
The substrate remains in the Qwen series to preserve tokenizer compatibility,
the model-surgery implementation, and continuity with Paper One.

Paper packaging is deliberately unresolved. It will be decided after the D0
pilot, not before the evidence exists.

## Amended Queue

1. Finish P0 as an uncitable adapter-based loss calibration.
2. Lock and run T1-lite on one fresh full-block lineage.
3. Draft and lock D0 speculative-depth recoverability.
4. Run D0 only after the T1-lite verdict and the D0 lock.
5. Decide paper packaging, the natural-trace composite track, and whether any
   stochastic-width work reopens.

## T1-Lite Boundary

T1-lite asks only whether the explicit internal continue/stop token pathway is
a reliable causal actuator on this substrate. It retains the four registered
gates: forced-depth preservation, self-halted preservation, exact row-level
depth selection, and exhaustive logit-level causal override.

The registered lineage is full block, seed 0, with seed-1 confirmation under
the preregistered positive and near-threshold policy. The former R16 registered
lineage is descoped. No capacity comparison remains.

P0 remains R16 because it was already authorized and Mark elected to run it.
The complete ten-cell calibration grid runs before either control-loss
coefficient is locked. Among cells with stop and continue recall both at least
0.60, selection minimizes answer-accuracy loss against the lambda-zero
reference; ties prefer lambda 1 and then ratio 3.5. The selected lambda and
class ratio may become locked T1-lite constants, but P0 is not matched-lineage
evidence. T1-lite must independently clear every gate.

## D0 Boundary

D0 asks whether disagreement or acceptance behavior from a same-tokenizer
Qwen teacher ladder yields useful, recoverable depth supervision. It is a
positive-seeking pilot, not yet an authorized experiment. Before launch its
preregistration must lock:

- teacher checkpoints and hashes;
- corpus, licenses, split hashes, and leakage policy;
- candidate generation and verification;
- acceptance and rejection rules;
- disagreement-to-depth mapping;
- distillation targets and trainable parameter set;
- depth-recoverable-fraction metric;
- acceptance-rate uplift metric and matched baseline;
- natural-surface non-degradation guardrail;
- seeds, budget, stopping rules, and failure interpretations.

No D0 launcher may exist while any required field remains unresolved.

## Banked And Closed

Arm G remains banked under the registered `NO-CHANNEL` and `BOTH_FAIL`
readings. Additional KL, scale, optimizer, duration, particle, SVGD, selector,
or unchanged-interface variational runs remain prohibited. The coded
intra-block oracle probe stays unrun.

Natural-trace training remains closed until the post-D0 decision. Candidate
sources remain OpenR1-Math-220k as primary, s1K-1.1 and LIMO-v2 as small
high-quality candidates, Bespoke-Stratos as secondary, filtered NuminaMath as
rehearsal only, and Fable-5-traces excluded under the current strategy record.
