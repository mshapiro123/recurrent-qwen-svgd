# Inverse-Composition Staircase Specification

## Purpose

This is the preregistered deterministic gate immediately following the Phase G
curriculum autopsy. It tests whether the inverse-composition failure was caused
by loop-label exposure starvation or by a real per-position cost for a
non-native reverse-search operation. It does not train stochastic latent heads,
learned per-trajectory halting, an LPRM, or SVGD. Phase G-alpha remains closed.

## Matched Arms

- **F, experiment:** the original forward table; each recurrent transition must
  search for a preimage.
- **C, control:** the same mapping, selected chain, target, row ID, split, and
  held-out row, rendered as an explicit inverse table. Each transition is then
  a forward lookup. Only rendering fields may differ.

Both arms initialize from natural-surface step 2000, SHA256
`0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
The N=20 train and test payload hashes are respectively
`4ab6377a15d64cf5e07c8855ed05f432feed75e512e196cbd53f648dc9fcb4a5`
and `4dd29d9fb7b4170390234646c7c1773377eea56145f6ae659e38f3ae443f2068`.

## Optimization Contract

The optimizer is AdamW in both arms. This preserves the recent deterministic
recovery lineage and avoids introducing Muon orthogonalization as a second
causal variable. AdamW state is preserved throughout a stage and is restarted
identically at stage boundaries. The next stage starts from the earliest
250-step checkpoint clearing its gate.

Each stage uses effective batch size 8 through gradient accumulation. If
`p_j` is the fraction of stage rows active at loop `j`, the base weight is
`1/p_j`; the newest loop receives a 2x multiplier; weights are normalized to
sum to the active cap. Loss is the fixed batch mean of the weighted active
loop-label cross-entropies. There is no per-row or active-label normalization.

Raw and weighted active labels are cumulative first-class receipts. After
removing the deliberate newest-loop 2x multiplier, realized loop masses must
remain within 0.8x to 1.25x at startup and every 200 optimizer steps.

## Stages And Gates

Phase 1 introduces caps 2, 3, and 4. Phase 2 introduces caps 5 through 8 only
if both arms pass every Phase-1 stage and the synthetic continuation guardrail.
Each stage is observed every 250 optimizer steps and stalls after approximately
1,500 weighted labels to the newest loop. The last 250-step checkpoint not
exceeding that dose is the stage limit; one checkpoint is the minimum
observable stage, so its dose may exceed 1,500 only when unavoidable (about
1,600 at cap 2). Phase envelopes are 4,000 and 6,000 optimizer steps per arm.
The stage gate is at least 46/64 held-out diagonal hits at the newest loop.

The active matrix is primary. Final diagonal accuracy is secondary. Every
transition receipt includes conditional success given the prior loop was
correct and ridge target decodability with a permutation null. At Phase-1 end,
CKA is reported separately for loop-2-correct and loop-2-incorrect rows only
when each stratum contains at least 32 examples.

## Safety And Readings

Active-supervision and nonzero-gradient assertions remain mandatory. A fixed
synthetic guardrail runs at each stage. A paired natural-surface accuracy
canary runs at each 1,000-step milestone with the standing hard-stop policy.
Selected checkpoints are SHA-checked and backed up to Drive; lightweight
receipts land after every arm-stage. Scientific stops use exit code 2.

The preregistered readings are:

1. F/C weighted-labels-to-bar ratio >=5x: non-native per-position cost.
2. Both arms cheap: prior failure was exposure starvation.
3. Both arms expensive or stalled: composition itself is difficult in this format.
4. F advances while C stalls: instrumentation alarm.

Post-run reporting clarification: when C reaches the bar and F stalls at the
same bounded dose, the receipt is `experiment_stalled_at_matched_dose`. That
result establishes a directional representation/operation asymmetry at the
matched dose, but it is not relabeled as the preregistered >=5x cost unless
both doses-to-bar are actually observed and their ratio clears 5x.

Muon is not part of this causal test. It may be evaluated later as a separately
preregistered optimizer ablation if the connected, dose-balanced AdamW job
stalls.
