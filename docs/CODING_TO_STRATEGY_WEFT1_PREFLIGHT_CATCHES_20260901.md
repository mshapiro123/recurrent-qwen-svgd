# CODING → STRATEGY — WEFT-1 PRE-FLIGHT first-firing catches

**Date:** 2026-09-01  
**Status:** FAIL-CLOSED CATCH REPORT · CPU EVIDENCE ONLY · ZERO A100 SPEND  
**Authority verified:**

- `STRATEGY_PREFLIGHT_PROGRAM_20260902.md` — 15,575 bytes · SHA-256 `ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b`
- `STRATEGY_PREFLIGHT_RATIFICATION_20260902.md` — 2,233 bytes · SHA-256 `4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965`

The PRE-FLIGHT gate earned its existence on its first firing. The registered B1 positive control does not recover its planted exponent under the implemented, governing coordinate convention. A5 contains a gate-definition collision that can make the stated certificate smaller than the true operator norm. A6's literal K=1 liveness criterion is impossible for a ratified module that is intentionally ineligible on visit zero. A8 names a shared code path that does not exist.

No governed behavior was changed. No A100 runtime was allocated, no PRE-FLIGHT meter was started, no P-A artifact or worker was touched, and no receipt was minted. The existing relevant regression baseline remains green: **113 passed** across callosum, Clifford geometry, model, RNG, and Jacobian-panel tests.

Proposed catch numbering continues the programme ledger after catch #21. Strategy may renumber without changing the requested literals.

---

## Catch #22 — B1 detects a log-base error in the headline exponent

### Governing definitions

The Jacobian handoff defines

```text
log |lambda_T| = log(cL) - p log T
```

but then binds `x = log2(T)` with `Sxx = 5.0`. The implementation follows that literal:

- `analysis/weft1_jacobian_panel.py::theil_sen_slopes` documents `x` as normally `log2(T)`;
- `design_sxx` computes `np.log2(T)`;
- PT1 constructs its synthetic response directly from that same `x`, so it tests internal algebra rather than the physical planted law.

The measured response is a natural logarithm. For a planted law `lambda_T = -c T**(-p)`, regressing `ln|lambda_T|` on `log2(T)` returns `p ln 2`, not `p`.

### Reproduced CPU evidence

```text
planted p       current p_hat       p_hat with x = ln(T)
1.0             0.6931471806        1.0000000000
1.5             1.0397207708        1.5000000000

current Sxx     5.0000000000
ln-depth Sxx    2.4022650696
```

This is not a tolerance issue: the bias is exact and deterministic. B1 therefore fails before its 20-replicate coverage phase.

### Literal requested

Recommended amendment:

1. bind the canonical coordinate to `x = ln(T)`;
2. bind `Sxx = 2.4022650695910066` for `T = (1,2,4,8)`;
3. replace PT1's self-referential response with a physical `lambda_T = -c T**(-p)` plant;
4. keep `p` as the base-independent exponent in `alpha(T) = c T**(-p)`;
5. keep the n0=32 pilot diagnostic-only and the 20-replicate coverage gate as registered.

The mathematically equivalent alternative is to retain `log2(T)` and change the response to `log2|lambda_T|`, but that would change more of the current implementation and receipt surface.

One adjacent item remains governed by the earlier C-JAC-2 objection: the present `sigma_slope_hat` subtracts raw-probe variance after P-5 paired probes and a nonlinear response. B1 can proceed initially in a zero-measurement-noise control, or strategy can bind the paired leave-one-probe-out jackknife before the full noisy coverage claim.

---

## Catch #23 — A3 and A5 double-apply the sidecar gate

### Governing definitions

A3 writes:

```text
||DeltaW_k|| <= g_k * max_e ||A_e B_e^T|| * (top-3 sum)
```

so `DeltaW_k` is already the gated update. A5 then writes:

```text
Lambda_k = ||rotor|| * (1 + g_k ||DeltaW_k||) * ||A_rho|| * Lambda_core
```

which applies `g_k` a second time. For gates in `(0,1)`, this can understate the residual-map norm and therefore is not an upper-bound certificate.

### Reproduced one-dimensional counterexample

Take an ungated expert mixture with norm 1 and `g = 0.5`:

```text
A3 gated ||DeltaW||             0.50
true ||I + DeltaW||             1.50
A5 factor as written            1.25
certificate shortfall           0.25
```

The example commutes and has no estimation error; it directly violates the claimed bound.

### Additional provability collision

The only current runtime spectral routine is finite-iteration JVP/VJP power iteration. It is an empirical lower estimate of the leading singular value unless convergence is independently certified; it cannot be used as a provable upper bound merely by naming it `Lambda_core`. The current recurrent visit also composes re-entry, scratch update, scratch injection, loop embedding, and two residual sublayers per core block. The A5 product omits those live branches.

### Literals requested

Recommended amendment:

1. name the pre-gate expert mixture `U_k` and the applied update `DeltaW_k = g_k U_k`;
2. use spectral/operator 2-norm throughout;
3. require non-negative top-3 weights with a recorded L1 sum and use `|g_k|`;
4. define the sidecar factor as either `1 + |g_k| ||U_k||` or, if `DeltaW_k` remains the applied update, `1 + ||DeltaW_k||` — never both gates;
5. bind a true upper-bound source for every live factor, separate from the empirical Jacobian estimate;
6. enumerate re-entry, scratch, loop-embedding, and each intra-block residual factor in the implemented-visit certificate, or explicitly scope A5 to a synthetic operator composition until those adapters exist;
7. keep `product Lambda_k > exp(cL)` as a flag only after `cL` and the certificate topology are ratified.

Until these literals land, a generic factor-composition utility can be built, but it cannot honestly be emitted as the production loop certificate.

---

## Catch #24 — A6's K=1 liveness requirement contradicts visit eligibility

### Governing definitions

A6 requires one backward pass with every module ON at `K in {1,4}`, with no parameter tensor having an identically-zero gradient. The ratified re-entry bridge deliberately executes only when `step_index > 0`. At K=1 there is only visit zero. The trajectory-invoked sidecar is likewise ineligible until the required second-order jet exists.

### Reproduced CPU evidence

On a tiny active WEFT graph with recurrence, static K/V, Hadamard experts, engram, scratch, lane carrier, and re-entry all materialized:

```text
K=1 parameters with grad is None:
  reentry_bridge.layer_scale
  reentry_bridge.prelude_norm.weight
  reentry_bridge.projection.weight

K=4 parameters with grad is None: none
identically-zero present gradients at either K: none
```

This is correct execution, not a frozen-parameter bug. A literal A6 pass at K=1 is therefore impossible.

### Literal requested

Recommended amendment: bind an explicit parameter-eligibility matrix by `(module, K, visit)`. Require nonzero gradients for every **eligible and executed** trainable tensor; report structurally ineligible tensors separately. Keep K=1 for modules that execute on visit zero and K=4 for re-entry and jet-conditioned modules. The receipt should include per-module minimum eligible gradient norm, ineligible parameter names with reasons, and fail if an eligible tensor has `grad is None` or is identically zero.

---

## Catch #25 — A8's claimed shared delta-rule path is absent

A8 says the fast-weight/delta-rule successor arms share a code path. The current WEFT tree contains no delta-state update with `(beta, eta, lambda, S)`. `ReadOnlyLatentMemory` is immutable retrieval and is not that path. The programme also says PRE-FLIGHT adds no new arms.

The spectrum `[1-beta, 1-eta lambda]` and cap alone do not bind an executable update: the update equation, tensor orientation, normalization, parameter domains, norm, gate placement, and the intended "unnormalized-rate" positive-control variant remain unspecified.

### Literal requested

Choose one:

- **defer A8** until the successor-arm implementation is separately authorized; or
- authorize a standalone reference operator and bind its complete update equation, domains, norm, and positive-control variant. The result would be a mathematical certificate only, not a production-path receipt.

No delta-rule architecture will be invented under PRE-FLIGHT.

---

## Coverage disclosures, not additional catches

- **A1:** the per-band Birkhoff callosum and the integrated narrow scratch carrier both exist, but they are different modules. Endpoint/adversarial and bf16 coverage can be added; production per-band integration remains absent.
- **A2:** the Euclidean `Cl(2,0)` tensor primitive exists and can be composed at K=1…8. No learned production rotor carrier exists, so the learned-composition clause must remain deferred rather than inferred.
- **A7:** exact structural-OFF tests exist for parts of the graph. A complete module × K × dtype × backend matrix cannot claim the absent integrated rotor, per-band callosum, or sidecar.

These disclosures do not block the unambiguous standalone certificate tests. They do block language implying that the production integrated graph has passed them.

## Preserved execution posture

```text
PREFLIGHT_AUTHORITY_VERIFIED
CPU_BASELINE_GREEN
B1_POSITIVE_CONTROL_FAILED_AS_DESIGNED
A5_A6_CONTRACTS_NOT_EXECUTABLE_AS_WRITTEN
ZERO_A100_SPEND
P_A_UNTOUCHED
NO_PREFLIGHT_RECEIPT_MINTED
```

After an amendment binds catches #22–#25, the coding lane can implement the natural-log B1 control, the corrected factor certificate, the eligibility-aware liveness hunter, and the A8 disposition without another design round. Independently, the dedicated 5-hour PRE-FLIGHT meter will remain a separate receipt domain; the frozen 12-hour G-TOK meter and ledger will not be modified or reused.
