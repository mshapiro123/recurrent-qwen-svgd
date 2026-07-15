# Phase G-alpha Guided Stochastic Transition Specification

**Date:** July 12, 2026
**Implementation status:** Deterministic substrate gate passed; powered numeric margin remains to be locked before model implementation and launch

> **July 15 gate result:** The natural step-2,000 N20 verbal keeper passed the
> frozen branching gate at `389/512 = 75.98%`, with depth accuracies of
> `99.22%`, `74.22%`, `67.97%`, and `62.50%`. Its verified SHA-256 is
> `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
> The N24 symbolic keeper narrowly missed at `355/512 = 69.34%`, including
> depth 3 at `52.34%`. No adapter is needed. G-alpha is authorized after the
> preregistered power calculation fills the final numeric margin blank.

> **July 15 substrate amendment:** The closed inverse-task program showed that
> backward inference confounds multimodal coverage with installation of a
> non-native operation. G-alpha now uses multi-valued forward relations: two
> successors per state, exact reachable sets at depths 1-4, and the same-reader
> validity gate defined in the Part 1 closeout. The deterministic block is a
> frozen asset; only the prior head, posterior head, and injection scale train.
> The inverse-rendered amendment below is retained as historical lineage but no
> longer controls launch.

> **July 14 substrate amendment:** G-alpha remains gated, but its first deterministic validity assay is now inverse-rendered non-injective abduction on a demonstrated forward-lookup substrate. This preserves exact multimodal coverage while narrowing the claim: it tests guided branching over an explicitly rendered inverse relation, not canonical backward inference from a forward table. Frozen-set construction, validity gates, and the two-lane dependency order are defined in [TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md](TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md). The frozen-block architecture and comparator rules below remain unchanged.

## Objective

Test whether a learned conditional distribution over recurrent transitions discovers more distinct valid solutions than output-head sampling or additional deterministic depth. This is a transplantation test of GRAM's guided stochastic-width principle onto a repaired pretrained recurrent-Qwen substrate, not a reproduction of GRAM.

## Transition

Let `u_l` be the deterministic high-level state produced for loop `l`. The frozen recurrent transition remains unchanged. The stochastic re-entry is:

```text
z_l ~ q_psi(z_l | u_l, e(gold_next_l))       training
z_l ~ p_phi(z_l | u_l)                       inference
h_l = u_l + softplus(s) R z_l
```

`R` is a fixed seeded orthonormal projection from a 64-dimensional latent into the hidden dimension. It is a buffer, not a parameter. `s` is the trainable scalar injection scale and starts at `1e-3` after transformation. Stochasticity enters only at the bridge/re-entry level; no sublayer noise is allowed.

The project has gold intermediate **symbolic** states, not gold hidden
activations. `e(gold_next_l)` therefore means the frozen model embedding of the
next symbolic state, combined with the loop index. On a branching row the
generator samples one valid root-to-leaf path and stores the full chain. That
sampled chain supplies transition targets for the posterior and deep
supervision, while the exact reachable set supplies the coverage oracle.

## Heads

The prior head consumes a normalized high-level re-entry state and emits diagonal-Gaussian mean and log variance in 64 latent dimensions. The posterior consumes the same state plus the frozen embedding of the stored next symbolic state and emits its own mean and log variance.

Only these parameters may train:

```text
phase_g_prior_head.*
phase_g_posterior_head.*
phase_g_injection_scale
```

Everything else has `requires_grad=False`. Startup enumerates all trainable names and aborts on any unexpected parameter. Every backward pass asserts that every frozen parameter has either no gradient or exactly zero nonzero elements. Approximate zero is not accepted.

## Loss

For every supervised loop:

```text
L_l = CE_l + beta_kl * KL_balanced(q_l || p_l)
```

The task term is the existing per-loop symbolic-chain CE plus final valid-answer CE. KL is applied per transition, finer than GRAM's practical final-transition surrogate. KL balance starts at `0.8`. The preregistered coefficient sweep is `1e-4`, `1e-3`, and `1e-2`. Trainable weights are evaluated both raw and with EMA decay `0.999`.

Mandatory diagnostics by loop, depth, and reachable-set stratum:

- task CE;
- KL;
- prior and posterior variance;
- posterior/prior mean distance;
- posterior-collapse fraction;
- injection magnitude relative to deterministic state RMS;
- frozen-gradient assertion count;
- EMA versus non-EMA coverage.

## G-alpha Inference

- Fixed loop depth `T`; no learned halting.
- Independent prior samples for `K=1,2,4,8,20`.
- No particle interaction and no SVGD.
- Oracle scorer only; no selector is required.
- The same prompt, checkpoint, rows, and candidate validity function are used for all arms.

## Comparators

### Entropy-matched answer sampling

For every row, compute the categorical entropy of the latent arm's candidate distribution. Solve for the deterministic answer-head temperature that matches that entropy, then sample the same `K`. Record target entropy, achieved entropy, temperature, absolute error, and clamp rate. A fixed temperature is only a provisional diagnostic.

### Iso-compute depth

Compare `K` trajectories at `T` recurrent transitions each against one deterministic trajectory at `K*T` transitions. Report actual bridge calls and refuse the comparison if the requested depth is not honored.

The wrapper has no literal recurrent-loop cap, but its halting loop embeddings saturate above their trained index range. G-alpha disables learned halting and reads the explicitly forced final loop, so those embeddings do not select the output. A preflight artifact check must still prove that the requested `K*T` loops and expected bridge-call count actually execute, including the depth-4, K=20 cell at 80 loops.

## Guardrails and Lineage

The launch checkpoint must be one of the two exact frozen keepers that clears
the branching-relations gate, and it must match its explicit SHA. The gate is
pooled exact validity at least `0.70` and every depth at least `0.55`, with the
emitted symbol scored against the exact reachable set. The deterministic
synthetic guardrail and canary run before and after training. Because the
substrate is frozen, any guardrail change is treated as an implementation or
evaluation defect until disproven.

The frozen screen uses N20 verbal rows on the natural keeper and N24 symbolic
rows on the N24 keeper. Only one keeper must pass. If both narrowly miss, one
detachable rank-at-most-16 attention adapter may be considered after explicit
review; the base block remains frozen and the adapter-attached guardrail must
hold `0.93` in every registered stratum. No full-block continuation may cross
this seam.

G-beta opens only after latent coverage beats both comparators. LPRM selection, per-trajectory halting, and SVGD remain absent from this implementation.
