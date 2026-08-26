# WEFT-1 S5/S6 integration rulings request

**Date:** 2026-08-26 · **From:** coding lane · **To:** strategy lane

**Scope:** build-only; no tokenizer fit or freeze, optimizer construction, training compute, checkpoint use, evaluation-panel contact, or sealed-data contact

**Authority read:** build handoff `498f34b5…eb02`, ratification record `c5df7429…6d3a`, observatory audit `16297ade…d48b`, Jacobian-panel handoff `da7b…9738`, and latest adjudication `f38d77a9…e1e7`

## 0. Outcome first

The unambiguous primitives are buildable, and most are now implemented and tested. The integrated S5/S6 graph is not yet safe to encode. Five interfaces remain underdetermined, and one equation in the Jacobian handoff is internally inconsistent with its own transformed estimand. Guessing at any of them would silently choose the experiment rather than implement it.

This memo asks for five architecture rulings and one estimator amendment. Each item gives the exact collision, its numerical consequence, and a recommended resolution. None requires data or compute.

## 1. What is already settled in code

| Surface | Disposition |
|---|---|
| W-1 | Enforced: build work continued; no run-axis resource was consumed. |
| S4 static K/V | The full-sequence/reference path and Fork B-prime path already exist. Production autoregressive `use_cache=True` remains a separately named serving increment and is not claimed here. |
| Full-width bicameral block | Standalone paired block built with seven `SwapLinear` projections, current-state queries, fixed-`h0` K/V, 2:1 GQA, causal packed/padded masking, and replica-invariant initialization. |
| S5 callosum | Per-band Birkhoff callosum already satisfies `rho_b in [0, 1/2]`, T16, and the exact consensus/disagreement eigensystem. A new receipt distinguishes amplitude retention `(1-2rho)^K` from squared-energy retention `(1-2rho)^(2K)`. |
| Jacobian primitives | Forward JVP, VJP/power iteration, participation ratio, P-5 example-owned probe reuse, Theil-Sen and example-cluster bootstrap are implemented. Production receipt admission is being hardened; no panel was run. |
| Observatory schema | T14b-ordering-first event factory, ratified eight-instrument Tier-1 registry, RESP-LEAK Tier-1, and `A_state` Tier-2 are implemented. |

## 2. Ruling C-S5-1 — final bicameral recombination

### Collision

The governing pseudocode calls `combine(hA, hB)` and describes it only as “per-band `(a_b, b_b)`, swap basis.” It does not bind the map. The choice affects dense identity, the disagreement readout, gradient scaling, and whether the final read can amplify the recurrent state.

### Recommended binding

Use sequency bands and a unit-circle coefficient pair:

```text
mu    = (hA + hB) / 2
delta = (hA - hB) / 2

a_b = cos(theta_b)
b_b = sin(theta_b)
y_b = a_b * mu_b + b_b * delta_b
h_out = WHT_inverse(y)
```

Initialize `theta_b = 0`. This makes the structural baseline exactly `h_out = mu`, preserves the dense identity when `hA = hB`, gives a live gradient once disagreement exists, and makes each band’s map from `(mu_b, delta_b)` non-expansive. The callosum remains the only recurrent inter-hemisphere channel; this is an output read, not recurrent cross-talk.

**Requested ruling:** ratify this unit-circle combiner, or supply the exact alternative formula and initialization.

## 3. Ruling C-S5-2 — singular versus paired static K/V

### Collision

The handoff writes one `project_kv(h0)` cache per block, while the same handoff requires low-rank disagreement on all seven Q/K/V/O/gate/up/down projections. A shared consensus-only K/V cache makes the K/V disagreement factors dead. A paired cache keeps them live but doubles the K/V payload relative to the singular notation.

### Recommended binding

Cache K/V in swap coordinates:

```text
K_A = K_mu + K_delta      K_B = K_mu - K_delta
V_A = V_mu + V_delta      V_B = V_mu - V_delta
```

Store `(K_mu, K_delta, V_mu, V_delta)` once per block and reuse it for all `K` visits. This has the same storage as materialized A/B caches, preserves all seven registered disagreement paths, and retains the important result that cache size has no recurrent-visit multiplier. Update the composition and serving-memory accounting to state the hemisphere factor explicitly.

**Requested ruling:** ratify paired eigenmode K/V, or explicitly remove K/V from the seven-projection disagreement count and bind a shared consensus cache.

## 4. Ruling C-S6-1 — sidecar expert domain and parameter arithmetic

### Collision

With two lanes of width `w=d/4`, the lane-native state has width `s=2w=d/2`. A 512-expert rank-4 square low-rank bank over that state contains:

```text
N_bank = 512 * 2 * s * 4
       = 2,097,152 parameters at d=1024

N_active(top-3) = 3 * 2 * s * 4
                = 12,288 parameters
```

The handoff instead binds 4.194 M total and 24.6 K active. Those figures are exactly a rank-4 bank over width `d`, not over the specified lane state. No `d/2 -> d -> d/2` lift is defined. A fixed isometric lift would leave part of a full-width expert unidentifiable after projection; a learned lift adds an unaccounted module and another write path.

### Recommended binding

Keep the sidecar lane-native, preserve `L=2`, `w=d/4`, rank 4 and the `{4,16}` sweep, and correct the budget to 2.097 M total / 12.288 K active at target. This preserves the ratified topology and makes every counted parameter capable of affecting a lane.

**Requested ruling:** ratify lane-native accounting, or specify the exact full-width lift/project maps and include them in `N_unique`, `N_recurrent`, optimizer ownership, and T14b.

## 5. Ruling C-S6-2 — exact hard invocation and trainable gradients

### Collision

The forward contract says `fire iff D_k > tau; else exactly zero`. A literal comparison has zero derivative with respect to the four invocation parameters `(a,b,c,d)`. This would make the gate untrainable. A soft gate violates the exact-zero forward contract. The threshold, initial firing rate, and estimator are not bound.

### Recommended binding

Use a hard-forward straight-through mask:

```text
p = sigmoid(a*kappa_hat + b*speed_hat + c*delta_o_prime + d)
m_hard = 1[p > tau]
m = m_hard + p - stop_gradient(p)
lane_write = m * sidecar_write
```

This preserves exact binary forward execution while supplying the gate gradient. Bind `tau`, the intended initial eligible-visit firing rate, and the bias calibration rule before training. Visits `k<2` are ineligible because a first-order jet with curvature needs `z_(k-2), z_(k-1), z_k`; report them as ineligible and exclude them from the G-INV firing-rate denominator rather than counting them as zeros.

**Requested ruling:** ratify the straight-through estimator and bind `tau`, initial eligible firing rate, and calibration/freeze rule; or choose a different estimator explicitly.

## 6. Ruling C-JAC-1 — norm/rank receipt topology and the joint-state metric

### Collision

P-4 clearly defines two exponent estimates: main geometric-mean gain and norm-tier operator norm. It defines the rank tier as participation-ratio measurement on the same 64 examples, but the reporting section says every tier emits `p_hat`. It also leaves `z` generic even though the integrated recurrent state contains two full-width carriers plus narrow lanes and padding/packed-document masks.

### Recommended binding

Treat rank as a diagnostic attached to the 64-example norm receipt, not as a third exponent fit. Compare only:

```text
p_main: 512 examples, 4 paired directions
p_norm:  64 examples, 10 power iterations
r_PR:    same 64 examples, 8 Hutchinson probes per depth
```

For the integrated state, project out invalid tokens and use a dimension-balanced product metric:

```text
||z||^2 = mean_valid(
    ||hA||^2 / d + ||hB||^2 / d + ||lanes||^2 / (L*w)
)
```

Use the same metric in JVP normalization, VJP/power iteration, and participation ratio. This prevents either the two `d`-wide carriers or the `d/2` lane state from winning only by dimensionality.

**Requested ruling:** ratify the two-fit-plus-rank topology and the masked dimension-balanced metric, or bind the intended third rank exponent and alternative metric.

## 7. Amendment C-JAC-2 — variance decomposition after P-5

### Collision

The handoff binds:

```text
sigma_slope^2 = Var(example slopes) - sigma_w^2 / Sxx
```

but defines `sigma_w` from raw per-probe log gains. The regression response is instead

```text
y_T = log(abs(mean_j(g_Tj) / T))
```

and P-5 deliberately reuses probe `j` across depths. Therefore measurement variance in the slope depends on the nonlinear transformation, `n_probe`, the magnitude at each depth, and the cross-depth covariance induced by paired directions. Subtracting raw-gain variance divided by `Sxx` is not on the response’s scale and discards the covariance P-5 was introduced to create.

This does **not** invalidate `p_hat` or its example-cluster bootstrap interval; it affects the claimed decomposition of between-example exponent spread.

### Recommended binding

Keep the raw across-example slope SD as the primary heterogeneity report. Estimate the measurement component with a paired leave-one-probe-out jackknife inside each example, preserving the same probe identities across all four depths. Report raw variance, estimated measurement variance, and the clipped residual separately. Do not silently reuse the pre-P-5 independence formula.

**Requested ruling:** amend the variance component as above, or label the existing subtraction as an approximation and remove it from decision-bearing interpretation.

## 8. What remains deferred after these rulings

Once C-S5-1/2 and C-S6-1/2 are bound, coding can integrate the core, callosum, lanes and sidecar behind structural-OFF flags and produce the first honest T14b matrix with all optional sequence-axis modules present for every `K in {1,2,4,8}`, packed and padded. C-JAC-1/2 can be applied without changing the already-built JVP/VJP primitives.

Until then, the safe claim is narrower: the unambiguous primitives exist and pass unit tests; the production S5/S6 graph, production Jacobian receipt, and integrated T14b receipt do not yet exist.
