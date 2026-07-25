# Handoff: COCONUT Composite Integrity Preflight

**Date:** 2026-07-25  
**Run:** `stage5_coconut_composite_rg1_rg11_20260725`  
**Substrate:** Qwen2.5-0.5B fresh recurrent surgery  
**Mode:** no training  
**Verdict:** `failed_integrity_contract`; RG-12 remains unauthorized

## Executive result

The COCONUT-style horizontal route is connected, causal at the differential
level, compatible with vertical recurrence, and transparent through a frozen
adapter backbone. Eight of eleven integrity contracts passed. Three failed:

- RG-4, one fixed-epsilon finite-difference check, missed its tolerance by
  `5.43e-4`.
- RG-5, the optional sliced-cache optimization, preserved gradient direction
  exactly but missed strict logit and absolute-gradient equivalence.
- RG-11, full-model bf16, produced a fed-state gradient cosine of `0.983584`
  against the fp32 reference, below the locked `0.99` threshold.

The result does not identify a detached horizontal path. It does block a pilot
under the current precision policy and forbids treating sliced cache as
equivalent to full recomputation.

## Design under test

The final post-norm hidden state immediately before a
`<|recur_readout|>` placeholder is reconstructed into that placeholder's input
embedding slot. An `I + delta` horizontal bridge is exact identity at step
zero. Full-prefix recomputation is the reference path. The first future pilot,
if authorized, uses fixed vertical depth L=1 and a frozen identity horizontal
bridge.

## Contract results

| Contract | Result | Measurement |
|---|---:|---|
| RG-1 composite-off identity | Pass | H=0 logit difference `0.0` at L=1 and L=2; adapter L=2 also `0.0` |
| RG-2 bridge identity | Pass | identity bridge versus raw feedback difference `0.0` |
| RG-3 chain reachability | Pass | fed-state gradient norm `0.140663`; prompt activation gradient norm `163.181` |
| RG-4 finite difference | **Fail** | analytic `-0.0110439`; finite difference `-0.0128746`; error `0.0018307`; tolerance `0.0012875` |
| RG-5 sliced-cache equivalence | **Fail** | logit max difference `8.51e-5`; gradient max difference `2.86e-4`; gradient cosine `1.0` |
| RG-6 frozen transparency | Pass | 84 LoRA modules; frozen base gradients absent; earlier feedback activation live |
| RG-7 two-axis accounting | Pass | H*L feedback cells `2`; total cells `3`; independent forward/backward counts both `3`; all cell gradients nonzero |
| RG-8 row and parameter coverage | Pass | replaced-slot activation gradient count `0`; optimizer and EMA hashes identical across 175 names |
| RG-9 anomaly detection | Pass | full forward/backward completed |
| RG-10 checkpointing | Pass | logit difference `0.0`; gradient cosine `1.0` |
| RG-11 precision boundary | **Fail** | all states finite; bf16/fp32 fed-gradient cosine `0.983584` |

## What is established

- H=0 is exactly identity-preserving for both budgets.
- The answer loss reaches the fed horizontal state and the prompt activations.
- Horizontal and vertical application counts are correct and every measured
  grid cell receives nonzero credit.
- Frozen adapter weights do not block activation-gradient flow.
- Replaced placeholder embeddings do not leak input-side gradients.
- Full recomputation remains connected under activation checkpointing.
- The horizontal bridge may remain frozen identity for the future integration
  pilot, as designed.

## What is not established

- Sliced-cache training is not equivalent to recompute under the locked
  tolerance. It must not be used in training without a separate repair.
- The current full-bf16 policy does not meet the precision contract.
- The single-epsilon RG-4 miss does not distinguish a bad derivative from
  fp32 finite-difference cancellation or an unsuitable epsilon.
- RG-12 causal use has not run, and no result shows that a trained model uses
  the horizontal state rather than merely permitting gradients through it.

## Recommended bounded follow-up

1. Retire sliced cache from the authorized training path. Keep recompute as the
   sole reference path; RG-5 remains a documented rejected optimization.
2. Run an fp32 epsilon-stability sweep for RG-4 at fixed direction and fixed
   weights. Require at least two adjacent epsilon values to agree with the
   analytic derivative under the original 10 percent relative criterion.
3. Compare precision policies on a small fixed prompt set: full fp32, current
   full bf16, and fp32 master weights under bf16 autocast. Keep the original
   per-example `0.99` cosine threshold. Select no policy from answer accuracy;
   this is graph numerics only.
4. Authorize RG-12 only if recompute passes the derivative diagnostic and one
   declared production precision policy passes RG-11. Do not relax the
   thresholds based on this run.

## Canonical artifacts

- `outputs/stage5/stage5_coconut_composite_rg1_rg11_20260725/summary.json`
- `outputs/stage5/stage5_coconut_composite_rg1_rg11_20260725/receipt.md`
- `docs/COCONUT_COMPOSITE_INTEGRITY_SPEC_20260725.md`
- `docs/COCONUT_COMPOSITE_RG0_GRAPH_AUDIT_20260725.md`
