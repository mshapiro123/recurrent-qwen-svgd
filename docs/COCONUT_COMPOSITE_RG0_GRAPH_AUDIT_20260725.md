# COCONUT Composite RG-0 Graph Audit

**Date:** 2026-07-25  
**Scope:** Qwen2.5-0.5B recurrent surgery before horizontal-path changes  
**Training authorized:** no

## Main forward path

The production recurrent forward in `models/recurrent_wrapper.py` contains no
`detach`, `.data`, `no_grad`, or `inference_mode` operation on the Prelude,
recurrent-state, Coda, normalization, or LM-head tensors. The detach calls in
that file are confined to diagnostic metrics, cached fixed projections, and
artifact loading. They do not sit on the normal loss path.

The vertical bridge performs dtype conversion, normalization, projections,
concatenation, and residual arithmetic without detaching. Its `no_grad`
contexts are initialization and projection-conversion operations only.

LoRA freezes base parameters with `requires_grad=False` but calls the base
linear layer normally. It does not wrap the frozen computation in `no_grad`,
so activation gradients remain able to pass through a frozen adapter budget.

## Cache and checkpointing

The wrapper accepts `past_key_values` and passes one cache object through all
decoder layers. It does not allocate a cache when the caller supplies none.
Any horizontal cache path must therefore construct the cache explicitly.

The wrapper permits KV cache only for one vertical loop and one trajectory. It
rejects cache for `max_loops > 1`. When the Qwen core has gradient
checkpointing enabled during training, the wrapper silently sets
`use_cache=False`. The composite contract must make that fallback explicit in
its receipt rather than claiming a cache run occurred.

The current registered trainers use `use_cache=False`. Their ordinary forward
and backward graph is continuous. EMA updates and checkpoint serialization run
after the optimizer step under `no_grad` or on detached copies, so they cannot
cut the live training graph.

## In-place operations

No in-place mutation of a live model activation appears in the recurrent
forward. In-place operations are limited to parameter initialization,
post-step EMA shadow updates, checkpoint restoration, fixed-token masking on a
clone, and diagnostic intervention tensors. Horizontal latent replacement
must preserve this property by rebuilding embedding segments rather than
assigning into `inputs_embeds`.

## Control-row behavior

The internal control-token installer initializes and splits the three new rows
outside the training graph. The resulting old rows are frozen and the three
control rows are shared between input embedding and LM head. A horizontal
placeholder using `<|recur_readout|>` must replace that position's embedding
functionally. The row can still receive LM-head gradient unless visible-token
labels and logits mask it; RG-8 must distinguish input-position leakage from
the intended shared output-row parameter.

## RG-0 decisions

1. Implement full recomputation as the reference and first training path.
2. Implement sliced cache only for `L=1`, with explicit cache allocation and
   a graph-preserving crop. Reject or fall back visibly for `L>1` and gradient
   checkpointing.
3. Expose the final post-normalization hidden state from the recurrent wrapper;
   no hook or detached logging tensor may supply horizontal feedback.
4. Keep the composite path in a separate wrapper and off by default. `H=0`
   delegates exactly to the existing recurrent wrapper.
5. Record both application counts: `H * L` feedback-producing grid cells and
   `(H + 1) * L` total recurrent-block applications including the final answer
   pass.
6. Interpret RG-8 as zero gradient from the **input placeholder path**. Because
   the row is tied to the LM head, its total parameter gradient need not be
   zero unless the control logits are excluded from the loss as specified.

## Result

RG-0 passes for construction. No pre-existing graph cut was found in the
recompute path. Cache reuse is not yet an established training path and must
earn equivalence under RG-5 before use.
