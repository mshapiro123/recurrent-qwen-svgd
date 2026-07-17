# Bounded Depth Selector Assessment

## Purpose

This is a closure experiment for the learned-depth pathway, not a general
study of adaptive computation. The random-table prompt explicitly states the
number of function applications. S1 therefore measures whether the installed
halting pathway can read and execute an explicit depth instruction. It does
not measure whether the model infers problem difficulty.

S2 asks the narrower original PonderNet question: when gold depth labels are
removed, can outcome loss plus a compute prior discover a useful depth policy
on this family?

## Immutable substrate

- Model: `Qwen/Qwen2.5-0.5B-Instruct` recurrent wrapper, split `6,18`.
- Mechanism checkpoint: N24 support-12 step 6000.
- Checkpoint SHA-256:
  `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`.
- Training rows: the frozen N24 `chain_mcq` rows, depths 1-12, 256 per
  depth, rendered through the same question-only full-symbol reader as
  evaluation.
- Evaluation rows: the frozen held-out N24 rows, depths 1-12, 64 per depth.
- Reader: question-only, full-symbol, one-token `A` through `X`.
- Maximum loops: 12.

The Prelude, recurrent block, bridge, re-entry adapter, Coda, and reader are
frozen. Only these existing selector parameters train:

```text
halt_predictor.proj.weight
halt_predictor.proj.bias
halt_predictor.loop_embedding.weight
halt_predictor.loop_bias
```

The target-loop router, target-loop embedding, and target-conditioned biases
remain frozen and unused. Passing gold depth through them would make S1 an
oracle-control test rather than a prompt-reading test.

## Exact frozen-feature method

The L4 first runs the immutable mechanism once over each prompt and caches:

- the pooled recurrent state at each loop;
- the target-symbol negative log likelihood at each loop;
- the forced-loop prediction at each loop for held-out rows.

Both selector arms train against these detached features. The mechanism is
therefore absent from the optimization graph. Every backward checks that no
frozen parameter has a nonzero gradient. The source checkpoint SHA and a hash
of all non-selector model parameters must match before and after both arms.

This is also the canary exemption: selector training can choose among fixed-T
outputs but cannot change any output at a fixed T.

## S1: supervised depth reading

Objective:

```text
CE(Ponder stopping distribution, stated row depth)
```

No answer loss and no target-loop control input are used.

Locked gates:

1. At every depth 1-12, at least 46 of 64 held-out rows select the stated
   depth.
2. At every depth 1-12, selected-depth final-answer accuracy is no more than
   3 percentage points below the same-row forced-depth reference.
3. Fixed-T accuracy is reported as a structural completeness control, not as
   intelligent allocation.

## S2: outcome-only Ponder objective

Objective:

```text
sum_t P(stop=t) * final_answer_NLL(t)
+ 0.02 * KL(P(stop) || truncated_geometric_mean_6)
```

There is no gold-depth loss. The geometric distribution is solved so that its
truncated mean over loops 1-12 is exactly 6.

Basic gates:

- final loss-window mean is at most 90% of the initial window mean;
- relative KL drift between halves of the final window is at most 25%;
- mean selected depth is strictly between 1.5 and 11.5;
- selected-answer accuracy is within 5 percentage points of S1.

Outcome bands:

- `STRONG`: basic gates pass and Spearman correlation between selected and
  stated depth is at least 0.8.
- `PARTIAL`: basic gates pass and Spearman is at least 0.3 but below 0.8.
- `COLLAPSE`: Spearman is below 0.3 or any basic gate fails.

If S2 collapses, the paper sentence is:

> Supervised routing works, but unsupervised depth discovery does not train on
> this family.

## Scope

The allowed claim is `depth_selection_control_pathway`: the existing halting
pathway can or cannot be trained under the two bounded objectives above.

The following claims remain prohibited:

- S1 demonstrates intelligent compute allocation.
- The selector infers unstated difficulty on random-table tasks.
- Learned halting on held-out hard reasoning is established.
- Fixed-T comparison is an intelligent-routing baseline.

If the independent PEFT closure produces a passing substrate, this exact
assessment may be repeated there as a second substrate without changing the
gates.
