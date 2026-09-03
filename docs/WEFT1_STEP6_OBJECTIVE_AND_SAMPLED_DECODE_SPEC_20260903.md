# WEFT-1 Step 6 Objective and Sampled-Decode Specification

**Date:** 2026-09-03
**Status:** executable build specification; implementation remains pending build step 6
**Authority:** D-MC-1, `STRATEGY_MATH_CHECK_RATIFICATION_20260903.md`, 2,868 bytes, SHA-256 `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1`

## 1. Default training rule

For a micro-batch with `K_exec` executed recurrent visits, decode the final
state, index `K_exec - 1`, through the shared coda and tied readout for the
ordinary language-model objective.

When `K_exec > 1`, draw exactly one integer

```text
j ~ Uniform{0, ..., K_exec - 2}
```

from the registered per-module O-9 stream `weft.lstage.sample` before recurrent
execution. Retain that selected earlier state and the final state with autograd;
diagnostic-only states remain detached. Decode the earlier state through the
same shared coda and tied readout in a separate serial call. The loss is

```text
L = L_LM(final) + lambda_stage * L_stage(j)
```

The sampled term is not divided by `K_exec - 1`: a single uniform draw is
already an unbiased estimator of the uniform average over the earlier visits.
There is one sampled visit per micro-batch, not one independently sampled visit
per token. Its derived seed and resulting integer are written to the run
receipt so replay reproduces the choice without perturbing another module's
random stream.

At `K_exec = 1`, no earlier visit exists. Decode only the final state, set
`L_stage` to exact zero, consume no draw from `weft.lstage.sample`, and record a
null sampled visit. Do not synthesize or pad an earlier state.

STOCH-K composes with this rule by sampling from the visits actually executed
by that micro-batch. If a future runner permits different `K_exec` values inside
one micro-batch, its masking and sampling semantics require a separate binding;
this specification assumes the current batch-level `K_exec` contract.

## 2. Receipt contract

Every composition receipt emits:

- `coda_decodes_per_step`: `2` for the D-MC-1 path when `K_exec > 1`, and `1`
  when `K_exec = 1` or in inference;
- `lstage_sampled_visit`: the integer `j` used for that micro-batch, or null
  when no sampled visit was decoded.

The sampled-visit field is a per-forward execution field. It must not be
inferred from an aggregate or fractional `executed_visits` statistic; aggregate
receipts leave it null rather than inventing a representative visit. A non-null
sample must be an integer in `[0, requested_visits - 2]` and implies exactly two
coda decodes.

Inference is unchanged: decode once at the selected or halted terminal visit.

## 3. Registered contrast

`LSTAGE-FULL` decodes every retained visit and uses weights summing to one over
the earlier visits. It is an exploration arm, not the production default. It
may be run if sampled-estimator variance measurably harms loop gain `eta_k`.

## 4. Compute re-derivation

The governing approximation changes from

```text
6 * D * (N_prelude + K * N_recurrent + N_coda)
```

to

```text
6 * D * (N_prelude + K * N_recurrent + 2 * N_coda)
```

for `K_exec > 1`. Using the ratified math-check counts gives:

| K | old active-equivalent count | D-MC-1 count | multiplier |
|---:|---:|---:|---:|
| 2 | 325 M | 430 M | 1.323077 |
| 4 | 440 M | 545 M | 1.238636 |
| 6 | 555 M | 660 M | 1.189189 |

Applying the K=4 multiplier to the previously stated approximately 234
A100-hour all-in allocation yields approximately **289.84 A100-hours**, an
increase of approximately **55.84 A100-hours**. It therefore exceeds that
allowance on the currently available arithmetic, so the pre-registered
D-CUR-4 de-scope order is engaged with **rung B first**. This is a planning
tripwire, not permission to silently delete a run: the exact allocation and
remaining rung-A/control schedule must be reminted from the integrated
step-3/step-4 parameter counts before S2 consumes training compute.

For sensitivity only, multiplying the older 196-hour K=4 rung-A anchor gives
242.95 hours; multiplying the 232-hour K=4 rung-B anchor gives 287.36 hours;
and multiplying the 306-hour K=6 ceiling gives 363.89 hours. These are not a
new budget because the historical anchors do not all denote the same run set.

## 5. Required implementation gates

Before training:

1. Two replays with the same O-9 root, replica, and micro-batch index select the
   same `j`; changing only the `weft.lstage.sample` replica changes no other
   module stream.
2. Across a deterministic coverage fixture, every eligible earlier visit is
   sampled and the empirical frequencies pass the registered uniformity test.
3. The same shared coda parameters receive finite, nonzero gradients from the
   final and sampled paths, and the sampled loss equals the direct loss at the
   recorded visit. The two coda calls are serial, not batch-concatenated, so the
   execution schedule is stable under the BF16 identity contract.
4. `K_exec = 1` consumes one coda decode, emits a null sample, consumes no
   L-stage RNG draw, and contributes exact-zero `L_stage`.
5. Receipt validation rejects out-of-range samples and rejects a non-null
   sample unless the execution reports exactly two coda decodes.
6. `LSTAGE-FULL` weights sum to one; it never becomes the default through a
   configuration fallback.

No corpus, sealed evaluation data, checkpoint, or GPU cell is consumed by this
specification.
