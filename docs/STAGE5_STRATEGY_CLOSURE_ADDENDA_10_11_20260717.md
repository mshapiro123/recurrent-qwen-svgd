# Stage 5 Strategy Closure Receipts: Addenda 10 and 11

**Date:** July 17, 2026
**Scope:** Receipts only. This file is not manuscript prose.

## Addendum 10: Guardrail battery

### PEFT installation arm

| Field | Receipt |
|---|---|
| Composition | 64 frozen arithmetic prompts |
| Baseline | `60/64 = 0.9375` |
| Hard-stop margin | absolute accuracy delta below `-0.03` |
| Correction method | none; this was a sequential operational hard stop, not a family of inferential tests |
| Step 1,000 | `61/64 = 0.953125`, green |
| Step 2,000 | `61/64 = 0.953125`, green |
| Step 3,000 | `61/64 = 0.953125`, green |
| Step 4,000 | `61/64 = 0.953125`, green |
| Step 5,000 | `61/64 = 0.953125`, green |
| Step 6,000 | `61/64 = 0.953125`, green |
| Dip details | no observed checkpoint dip below baseline |
| Identity control | one-loop maximum absolute logit difference `0.0` |
| Base-lineage control | pretrained base SHA unchanged |
| Permutation control | not part of this bounded canary; no permutation result is claimed |
| Evidence | `outputs/stage5/stage5_peft_ponder_closure_20260717_182113/summary.json` |

The canary supports preservation on the tested arithmetic slice only. It is
not evidence of broad natural-capability preservation.

### Deterministic keeper used by Phase G

| Field | Receipt |
|---|---|
| Composition | frozen N20 verbal branching rows, 128 per depth at depths 1-4 |
| Pooled gate | `389/512 = 0.7598`, floor `0.70` |
| Per-depth gate | `127/128`, `95/128`, `87/128`, `80/128`; each above floor `0.55` |
| Exactness control | reachable sets recomputed from each stored relation |
| Row identity | row SHA `eb80ef24637aee511a3e35607e87ae2530842ce11c551e6fa90ecda4d4115ef8` |
| Keeper identity | SHA `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f` |
| K=1 companion gate | pooled validity within `0.03` absolute and each depth/stratum cell within `0.08` before guided training is interpreted |
| Evidence | `outputs/stage5/stage5_part1_closeout_pivot_20260715/summary.json` |

## Addendum 11: Early-era telemetry archaeology

The recoverable early stochastic-era JSONL records contain expected-loop and
halting-entropy aggregates, but not a calibrated row-conditional depth policy.
After de-duplicating candidate copies by `(label, task, seed)`, the archived
settings show:

| Archive group | Unique task/seed cells | Mean expected loops | Mean halt entropy |
|---|---:|---:|---:|
| Extended fold 0 random-32 | 35 | 3.0535 | 1.1808 |
| Extended fold 0 within-group dim-8 | 35 | 3.0659 | 1.1726 |
| Extended fold 1 random-32 | 35 | 3.0631 | 1.1746 |
| Extended fold 1 within-group dim-8 | 35 | 3.0694 | 1.1704 |
| Recreated current random-32 | 70 | 3.0546 | 1.1798 |
| Recreated current within-group dim-8 | 70 | 3.0733 | 1.1681 |
| Original Stage 4 exact comparison | 39 | 2.8102 | 1.3024 |

Sources:

- `outputs/diagnostics/extended_fold*_*.jsonl`
- `outputs/diagnostics/recreated_*.jsonl`
- `outputs/stage4/stage4_opus_a100_20260620/exact_phase1_vs_phase2.jsonl`

These values show a moderately diffuse halting distribution centered near
three loops in the early stochastic evaluations. They do not show useful
row-conditional allocation. The later bounded selector result is stronger:
on a frozen executor with a nearly perfect forced-depth diagonal, the current
information path could not recover even a stated depth and outcome training
saturated at a boundary. No causal claim connects the early aggregate
telemetry to that later collapse.

## R16 bridge resolution

The R16 arm used `bridge_projection_mode="split"` while instantiating the
legacy tensors `bridge.proj.weight` (`[896, 1792]`) and `bridge.proj.bias`
(`[896]`). Split forward bypasses those 1,606,528 parameters in favor of
`bridge.prelude_proj` and `bridge.state_proj`. Report both:

- optimizer-marked: `7,613,953` =
  `4,399,104` recurrent LoRA + `3,214,849` bridge;
- forward-active: `6,007,425`.

The comparison against the full-block reference found no significant
per-depth difference, but it was underpowered to claim parity.
