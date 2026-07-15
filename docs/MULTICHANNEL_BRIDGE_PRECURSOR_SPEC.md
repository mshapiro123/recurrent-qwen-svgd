# Multi-Channel Bridge Precursor Battery

## Status

F9 is **CLOSED** as of July 15, 2026. M1 was smeared on both checkpoints and
M2 failed the required backward-checkpoint replication, leaving at most one
possible vote even if M3 were perfect. The two-of-three activation gate is
therefore unsatisfiable. M3, alternative basis searches, and a multi-channel
bridge implementation are not authorized. This historical spec records the
eval-only precursor and changes no queue,
checkpoint, training parameter, or architecture. The preregistered activation
rule required both:

1. at least two of M1, M2, and M3 confirm specialization; and
2. the independent inverse-composition staircase returns reading one,
   per-position installation cost confirmed.

The landed staircase currently reports `experiment_stalled_at_matched_dose`,
not reading one. Therefore this battery can characterize the mechanism and
strengthen the paper, but cannot by itself activate a multi-channel rebuild.

## Channel definition

"Head channel" is not a residual-coordinate slice and not a KV-head channel.
Qwen2.5-0.5B has 14 query heads and 2 KV heads. The locked basis uses the 14
64-dimensional input-column blocks of the final recurrent attention layer's
`o_proj`. Each block maps one concatenated query-head output into the residual
stream and is independently orthonormalized. This preserves the useful
attention-head analogy without pretending that residual axes are head-local or
that the 2 GQA KV heads define the carried-state partition.

Each head-basis statistic is compared with at least 20 random orthogonal
partitions of the same `14 x 64` shape. The basis must also clear the random
null; anisotropy that any basis captures is not head structure.

## M1: Subspace drift

The collector records the masked-mean pre-bridge carried state after every
recurrent pass on 64 frozen rows per depth. A PCA envelope is fit only to
loop-1 carried states. For each later loop, the off-envelope residual is
projected into the query-head write subspaces and every matched random
partition.

Locked positive reading:

- loops 6 and later are eligible;
- at least 75% of eligible loops have head-basis top-3 drift share at least
  `2x` the random top-3 mean; and
- the same number of loops lie above the random top-3 p95.

The battery-level M1 vote requires this result on both the N24 step-6000
checkpoint and the recovered backward checkpoint.

## M2: Retrieval-head census

The evaluator runs eager attention and measures, for each recurrent layer and
query head, attention mass from the final `Answer:` query token onto the
rendered table token span. Forward tables are located between `Function table:`
and the next blank line; handoff tables occupy the prompt prefix through the
first blank line.

A layer-head identity qualifies when its mean table mass is at least `3x` its
layer median and that condition holds on at least 50% of rows. At least two
identities must qualify. Because scalar attention probabilities do not live in
the residual write basis, the matched null is an independent random orthogonal
rotation of each 14-head attention-mass vector; actual top-2 squared-energy
concentration must exceed the random p95.

The battery-level M2 vote requires replication on N24 step 6000 and the
recovered backward checkpoint. The fixed-boundary backward and natural-surface
keeper conditions are interpretation controls, not extra votes.

## M3: Injection sensitivity

An eval-only bridge flag projects the learned prelude contribution out of one
query-head write subspace while leaving the carried-state contribution intact.
The disabled branch retains the previous operation order and must be bit-exact
on a one-batch equivalence check. M3 scores active intermediate labels on the
forward N24 set for all 14 head subspaces and 20 random rank-64 subspaces.

Locked positive reading:

- maximum nonnegative accuracy damage divided by median head damage is at
  least `5x`; and
- maximum head damage exceeds random-ablation damage p95.

The intervention is never used for training or checkpoint mutation.

## Execution and recovery

The full target runs four conditions: N24 step 6000, the natural-surface
keeper, backward fixed-boundary, and backward recovery. Dynamics caches and M3
ablation progress are stored on Drive after bounded increments; GitHub receives
only lightweight JSON/Markdown receipts. Every checkpoint is restored by an
explicit path and asserted against its locked SHA256.

To reduce wall-clock without changing the design, conditions may run in
separate Colab instances under different run IDs. Results must be merged only
after all condition summaries exist; parallel jobs must not share a run ID or
write the same master summary.

## Architecture interpretation

Time-recurrent architectures provide strong analogical support for distinct
channel dynamics: RetNet uses multi-scale retention, GLA uses data-dependent
gating in recurrent linear attention, and Mamba-2 relates multi-head structured
state-space recurrences to attention. None of that establishes that Qwen's
depth-carried residual state should be split by attention head. This battery is
the needed substrate-specific test before introducing learned channel
projections, separate state/injection gates, or separate decays.

Primary references:

- RetNet: https://arxiv.org/abs/2307.08621
- Gated Linear Attention: https://arxiv.org/abs/2312.06635
- Mamba-2 / Structured State Space Duality: https://arxiv.org/abs/2405.21060

