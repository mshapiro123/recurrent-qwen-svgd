# Phase G Oracle Re-entry Interface Probe

**Status:** locked before training

## Purpose

This is the one bounded, terminal probe authorized after the additive Phase G
A0 route received the ratified `NO-CHANNEL` verdict. It does not reopen A0 and
does not train a stochastic prior or posterior. It isolates route capacity:
can an oracle command the next valid transition through the frozen keeper?

## Arms

Two parameter-matched conditioners are trained from the same frozen keeper,
with identical rows, sampling sequence, optimizer, learning rate, steps, and
seed:

1. `additive`: two learned command branches are folded into an additive
   residual.
2. `film`: the same two branches emit `gamma - 1` and `beta`, applying
   `gamma * h + beta`.

Both branches consume the pooled current loop-input state and the frozen token
embedding of the selected chain's true next symbol. Both output layers are
zero-initialized, so installation is an exact identity. The only experimental
variable is the combination rule.

## Frozen Contract

- Keeper SHA-256:
  `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
- The recurrent block, bridge, coda, embeddings, reader, halting modules, and
  all historical Phase G tensors are frozen.
- Only `oracle_reentry_conditioner.*` may be optimized.
- Frozen gradients are asserted zero after every backward pass.
- The frozen lineage hash must be identical before and after each arm.

## Data and Objective

- Training: the existing `1,899` repeated-prompt selected-chain variants.
- Held out: the existing `106` variants across `32` prompt groups.
- Held-out transition denominator: exactly `305`.
- Objective: per-loop commanded-chain cross entropy.
- No KL, latent sampling, coverage, selector, particles, or SVGD.

## Locked Gates

An arm passes only if every gate passes:

| Gate | Threshold |
|---|---:|
| Non-default branch control | `>= 0.85` |
| Overall transition control | `>= 0.90` |
| Transition legality | `>= 0.95` |
| Terminal validity | `>= 0.71` |
| Zeroed-conditioning keeper identity | exact |
| Frozen keeper lineage | exact |

Default versus non-default is defined against the unconditioned keeper's
same-reader prediction at the same loop on the identical prompt. Per-depth and
per-loop target-logit margins and control rates are mandatory localization
outputs.

## Terminal Readings

- FiLM passes, additive fails: feature-wise modulation localizes the interface.
- Both fail: re-entry conditioning on the frozen substrate is closed.
- Both pass: A0's failure localizes to its variational objective or
  amortization.
- Additive passes, FiLM fails: report the unexpected asymmetry and pause.

No outcome automatically authorizes a successor. Any later variational design
requires a new strategy decision and retains the gate order: posterior control,
preservation, coverage, then selection.
