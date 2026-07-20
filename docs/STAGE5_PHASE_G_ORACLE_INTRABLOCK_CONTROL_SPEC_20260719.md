# Phase G Distributed Oracle-Control Localization

**Status:** locked before training

## Question

The frozen keeper failed both parameter-matched additive and FiLM command
interfaces when each command was applied once before the recurrent block. This
probe changes one fact: the same FiLM conditioner is shared and applied before
every recurrent layer. Does command access throughout the transition make
non-default paths controllable?

## Single Variable

The historical single-entry FiLM arm is the control. The new arm matches its
keeper, 1,899 training rows, sampled-row seed, 1,500-step dose, optimizer,
learning rate, bottleneck, parameter count, objective, and 106-row held-out
set. The same conditioner is reused at every recurrent layer, so no layerwise
parameter bank is added. Applying the conditioner before, rather than after,
each layer ensures every intervention still passes through recurrent
computation before readout. The only variable is command access location.

## Frozen Contract

- Keeper SHA-256:
  `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
- Only `oracle_intrablock_conditioner.*` is trainable.
- Frozen gradients are asserted zero after every backward pass.
- Frozen lineage must be identical before and after training.
- Zero-initialized installation must be an exact output identity.

## Gates

The historical gates and denominators remain unchanged:

| Gate | Threshold |
|---|---:|
| Non-default branch control | `>= 0.85` |
| Overall transition control | `>= 0.90` |
| Transition legality | `>= 0.95` |
| Terminal validity | `>= 0.71` |
| Zero-condition identity | exact |
| Frozen keeper lineage | exact |

## Readings

- `DISTRIBUTED_INTERFACE_CONTROLS`: single-entry access was the binding
  constraint. This reopens design work, but does not automatically authorize
  variational training.
- `DISTRIBUTED_INTERFACE_FAILS`: the frozen substrate is not oracle
  controllable through either tested small high-level interface. Close this
  transplantation route and require co-adaptation of recurrent dynamics before
  another width attempt.

No KL, stochastic prior/posterior, coverage, selector, particle, or SVGD work is
performed in this probe.
