# Paper 2 Phase T0: Internal Control-Token Preparation

**Status:** preparation only. No Paper 2 training is authorized by this file.

## Purpose

The bounded selector study showed that the tested pooled hidden-state head did
not recover useful depth allocation. Paper 2 tests a narrower, causally cleaner
alternative: make continue/stop an explicit internal token decision at each
recurrent transition, intercept that decision before text decoding, and never
allow the control symbols into a user-visible answer.

This is not a selector sweep. It is one registered information-path change.

## Tokens And Resize

The planned symbols are:

- `<|recur_continue|>`
- `<|recur_stop|>`
- `<|recur_readout|>`

Launch must first confirm that Qwen2.5's tokenizer does not already contain
them. Exactly three rows are added to the input embedding and LM-head
vocabularies. The resize preserves the base model's global
`tie_word_embeddings` policy; each new input/output row receives identical
initial values and the added parameter count is recorded.

Qwen2.5 pads its model vocabulary beyond `len(tokenizer)`. The executable
preflight therefore registers inert administrative aliases for those existing
padding IDs before adding the controls. These aliases add no model parameters;
the three controls then occupy the three genuinely new embedding/LM-head rows.
Both the alignment count and the three control IDs are recorded in the receipt.

All three token logits are masked from visible autoregressive generation. The
continue/stop values are read directly at a reserved per-loop control position.
The readout token is an internal hidden-state anchor, not an answer token.

## Phase T0 Unit Contracts

1. Control tokens are never emitted into decoded output.
2. Requested, executed, and selected loop counts are recorded and agree when
   forced.
3. With control inactive and `max_loops=1`, the maximum absolute output-logit
   difference from the unmodified surgery is below `1e-3`.
4. In future width integration, control logits remain unaveraged per
   trajectory, and K=1 must preserve the control decision as part of parity.

## Phase T1 Gate

T1 remains blocked until Paper 1 experimental closure, Arm E, and the first
G-alpha verdict have all landed. When authorized, it has exactly two fresh-base
lineages: adapter-budget and full-block. Both use 30% rehearsal.

The preregistered gates are:

- chain-diagonal accuracy within 3 points of its matching non-think reference;
- self-halted accuracy within 3 points of forced depth;
- continue/stop selection accuracy at least 0.90 at every trained depth;
- a reported causal-override falsification run.

Only a green result supports: "token-pathway halting succeeds where the tested
pooled-head halting path failed."

## Natural-Trace Survey

Natural traces provide segmented step counts and independently verified final
answers. They never provide latent-state targets.

| Dataset | Card license | Verification fit | T2 disposition |
|---|---|---|---|
| `open-r1/OpenR1-Math-220k` | Apache-2.0 | Math Verify for most traces; judge for a minority | Primary candidate |
| `bespokelabs/Bespoke-Stratos-17k` | Apache-2.0 | Rejection-filtered math/code; reverify locally | Secondary candidate |
| `simplescaling/s1K-1.1` | MIT | Ground truth plus grader fields | Small high-quality candidate |
| `GAIR/LIMO-v2` | Apache-2.0 | Answer-bearing curated math; reverify | Small high-quality candidate |
| `AI-MO/NuminaMath-CoT` | Apache-2.0 metadata | Heterogeneous sources; source-aware audit required | Filtered rehearsal only |
| `Glint-Research/Fable-5-traces` | AGPL-3.0 | Agent/tool traces lack a uniform answer verifier | Exclude from primary curriculum |

Dataset-card URLs and the machine-readable acceptance policy are in
`training/internal_think_token_spec.py`. License metadata is not a substitute
for checking upstream source terms before redistribution.

## Width Boundary

This program does not displace G-alpha. G-beta has no dependency on token
halting. Any later integration is a separately preregistered replication:
expected-transition iso-compute is relocked, K=1 includes control-decision
identity, and control readouts remain per trajectory.

## Prohibited Work

No pooled-head halting, selector sweep, rank ladder, Phase T3, or Paper 2
training is authorized here.
