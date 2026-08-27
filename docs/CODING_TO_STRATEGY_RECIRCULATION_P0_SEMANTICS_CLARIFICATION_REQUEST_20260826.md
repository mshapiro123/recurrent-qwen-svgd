# Recirculation Phase-0 Semantics Clarification Request

Date: 2026-08-26. Coding-agent stop-the-line report to strategy.

## 1. Status and authority

Authority reviewed:

- `STRATEGY_RECIRCULATION_PROBE_HANDOFF_20260823.md`
- Drive ID: `1vBn5JpoGl2cz7WyqGJobJlPkpHmEad3I`
- Drive bytes: 20,116
- Paper: Mozer et al., "Recirculation," arXiv:2608.17981v1
- Paper source reviewed: Section 2, Figures 3-4, Appendix A, and Appendix B

The handoff authorizes Phase 0 and Phase A but also requires the coding agent
to verify the evaluator against the paper before implementation and states
that the paper governs any discrepancy. That verification found a material
computational-graph mismatch. The agent therefore stopped before model-code
changes, data construction, GPU provisioning, scoring, or optimizer creation.

Current safety state:

- optimizer steps: 0
- GPU hours: 0
- CONFIRM scored: false
- EVAL-E scored: false
- Phase B remains unauthorized
- no paid Colab session was opened for this work

## 2. The paper and handoff do not presently define the same evaluator

### 2.1 Recurrence schedule and readout

The handoff pseudocode serially runs a complete Pass A and then recomputes the
same token through layers `d..L` as Pass B. It returns `lm_head(Pass-B)` for
that token.

The paper instead defines recurrence across both architecture-copy index and
input-step index. Its Equation (1) is

```text
z_(t+1,t,d) = alpha * f(z_(t,t,s) | d,t) + beta * z_(t,t,d)
```

and Section 2 says that the top stack at input step `i` is collapsed with the
bottom stack at input step `i+1`. It further states that readout occurs after
the first iteration of a stack. The paper's claim of essentially no added
generation latency also relies on this diagonal or pipelined schedule, not on
waiting for a serial same-token second-pass readout.

A serial implementation may be a valid correctness reference for the same
dependency graph, but only if it:

1. reads logits from the first iteration;
2. uses the recirculated iteration only to alter later input steps;
3. commits exactly the states that the diagonal graph makes visible later;
4. is proven equivalent to any fused or pipelined implementation.

The current pseudocode violates item 1 and does not fully specify items 2-3.

### 2.2 Source and destination tap points

The handoff defines `z_(t,l)` as the residual input to layer `l`, captures the
input to source layer `s`, mixes at the input to destination layer `d`, and
then reruns layer `d`.

The paper explicitly defines `z_(i,j,l)` as the residual-stream output after
incorporating layer `l`. Under that convention, a destination at layer `d`
mixes two post-layer-`d` states and the next computed block is `d+1`.

This is an off-by-one architectural difference, not a naming preference. It
changes the published Gemma anchor `(s=11,d=4)`, every Qwen heatmap cell, the
layers whose KV entries are replaced, and the cost estimate.

### 2.3 KV ownership and warm-up/flush behavior

The handoff says Pass A commits layers below `d`, Pass B overwrites or commits
layers `d..L`, and later tokens attend to those Pass-B entries. The paper
specifies the activation recurrence but does not give a Hugging Face KV-cache
algorithm. The diagonal graph and first-iteration readout leave four
implementation fields that must be frozen before the identity and published
anchor gates are meaningful:

- whether paper layer `d` maps to the output of decoder block `d` and therefore
  changes KV beginning at block `d+1`;
- which first-iteration upper-layer KV entries are provisional and which, if
  any, remain visible to later input positions;
- the first-token warm-up and final-token flush schedule;
- the exact position-id and causal-mask construction for the architecture-copy
  dimension.

Choosing these locally would create a new estimator under a published-result
anchor that is sensitive to exactly those choices.

## 3. Why the distinction matters scientifically

The two graphs answer different questions:

- A Pass-B readout asks whether reprocessing the current token with a deep
  activation improves its own prediction. That is close to the within-token
  depth family the program already closed.
- A first-iteration readout with recirculated future cache asks whether deep
  state from an earlier input step improves later shallow computation. That
  is the paper's state-tracking claim and the genuinely new axis.

Running the former under the latter's name could produce a clean heatmap and
still fail to test the hypothesized near-miss. It would also make a failed
Gemma anchor uninterpretable because the evaluator would not reproduce the
paper's graph.

## 4. Recommended ruling

Adopt the paper-native graph and amend Sections 2-4 of the executable handoff
as follows.

### R1. Registered readout

Use logits from the first iteration of each input stack. The recirculated
iteration is state construction for later input positions and is never the
current token's scored readout.

### R2. Registered tap convention

Use post-block residual outputs, matching the paper. Record an explicit table
mapping the paper's 1-based layer labels to Hugging Face decoder-block indices
and hidden-state tensors. Apply the `(11,4)` Gemma anchor only after that table
is frozen.

### R3. Reference schedule before optimization

Implement a simple token-sequential reference evaluator that realizes the
paper's dependency graph, even if slow. Then, if the 8-hour ceiling requires
the diagonal batched or pipelined schedule, add a hard equivalence gate:

- identical token logits on a fixed short sequence;
- identical committed per-layer K/V tensors;
- identical first-token warm-up and final-token behavior;
- tested at `alpha=0` and at one nonzero anchor cell.

Do not use throughput optimization as the reference definition.

### R4. Revised identity gate

At `alpha=0`, compare complete-sequence first-iteration logits and all committed
future-visible K/V tensors against the intact model. A same-token Pass-B logit
comparison is removed because Pass B is not the registered readout.

### R5. Receipt the exact cache contract

Before the Gemma anchor, emit a small graph receipt containing, for each token
and layer, the architecture-copy index, input-step index, tensor tap, K/V owner,
and whether that tensor is scored, provisional, committed, or discarded.

## 5. Requested strategy response

Please ratify the recommended R1-R5 package or provide a different executable
graph with all of these fields fixed:

1. scored iteration;
2. post-layer versus pre-layer source and destination convention;
3. exact paper-layer to Hugging Face mapping;
4. K/V ownership by layer and architecture copy;
5. warm-up and flush behavior;
6. whether a serial reference is acceptable for correctness even though the
   paper's deployment schedule is pipelined;
7. whether the cost ceiling is applied to the correctness-first serial probe
   or only after a proven-equivalent optimized evaluator exists.

No sweep constants, populations, keys, effect floors, or Phase-A scientific
questions need to change. Phase 0 and Phase A remain paused until this ruling
lands. Phase B remains outside the authorization regardless of the ruling.

## 6. Source record

Primary source: https://arxiv.org/html/2608.17981

Relevant paper locations:

- Section 2, paragraphs describing depth-and-step recurrence and two input
  stacks per recurrence step;
- Equation (1), including the indices `z_(t+1,t,d)` and `z_(t,t,s)`;
- the sentence following Figure 3 that fixes first-iteration readout;
- Appendix A's unrolled architecture;
- Appendix B.3's definitions of `d`, `s`, and `d'` as residual-stream outputs.
