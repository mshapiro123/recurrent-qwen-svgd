# STRATEGY — D-NB-1 Ratification Record: Live K/V Becomes the Default; Static K/V Becomes the Control Arm

**Date:** 2026-09-03 · **Status:** RATIFICATION RECORD + AMENDMENT to handoff §5.3, to S-3 (S5/S6 rulings) and to A1's Q1 record. Mark ruled option (b) on 2026-09-03 — **an overrule of strategy's recommendation (a), recorded as coherent:** the only external evidence on this axis (Nanbeige4.2-3B, Tier-2, 3 B / 28 T) points to live K/V; the training-time cost of the flip is negligible; and the ratified design intent ("re-query a fixed reading with an updated question") is better tested as the *control* than assumed as the default. Not relitigated.

---

## The amended binding

> **§5.3 (amended) — K/V are recomputed at every visit from each hemisphere's current state.** For core block i at visit k: `K_A = W_K h_{A,k}`, `V_A = W_V h_{A,k}`, `K_B = W_K h_{B,k}`, `V_B = W_V h_{B,k}`, causally masked as normal. **The projections `W_K, W_V` remain μ-only (S-3 stands: K and V are not in the paired set — the paired set is Q, O, gate, up, down).** Hemisphere K/V therefore differ by *input*, not by weights: each hemisphere attends over **its own** keys and values. **No hemisphere ever attends over the other's K/V** — that would be a second inter-hemisphere channel, and §5.6 permits exactly one (the callosum). `h₀` still seeds the lanes (`bridge_in`) and the retention gauge; it no longer seeds the attention memory.
>
> **Design-intent statement, amended:** the loop *re-reads its own evolving state*; the lanes and carrier are additional persistent state alongside that re-reading, not a substitute for it. The former intent — a fixed reading re-queried with an updated question — is preserved as the **KV-STATIC control arm**.
>
> **Serving cache, stated plainly:** with per-visit, per-hemisphere K/V, an autoregressive server that caches every visit holds **2K× a standard transformer's KV cache** (8× at K = 4). Training is unaffected (no cache under multi-branch execution). **Fork B′ midpoint refresh** is re-registered as the cache-economy arm (2× per hemisphere); **KV-STATIC** (the former default) as the control. The **S2 contrast registered in the Nanbeige adjudication runs unchanged** — now as live (default) vs static (control) at the proxy rung, first in S2, both seeds, with the branches as written; if static matches live, the default reverts and the cache is recovered.

## Consequences bound with it

(i) **Integration step 2** wires the bicameral block with **per-visit K/V recompute**; the currently integrated static single-stream K/V path becomes the KV-STATIC arm's implementation (structural switch `kv_policy ∈ {live, static, midpoint}`, default `live`). (ii) The **T15 equivalence** tests that pinned the static path are retained for the static arm and a live-path equivalence test is added (at K = 1 the two policies coincide — that identity is the anchor). (iii) The **composition receipt** gains `kv_policy` and `kv_cache_multiplier_at_serving` (= 2K, 2, or 1). (iv) **C-JAC / PF-1.4:** the attention factor's Jacobian now depends on the current state through K and V as well as Q — `Λ̂_core` is re-measured under the live policy; no certificate claim changes (attention was already the empirical factor). (v) **KV-PAIR** (A1) is unchanged and orthogonal (it would pair the *weights*; this rules the *inputs*). (vi) The **EXTRAP-K** instrument gains a note: beyond-horizon behaviour under live K/V is the GRT/Nanbeige-comparable setting; the static arm's beyond-horizon curve is the differentiator.

## Ledger

The A1 record's Q1 outcome ("shared consensus from h₀") is **superseded on the h₀ half and preserved on the μ-only half**: consensus *weights*, live *inputs*. The Nanbeige adjudication's §3 registration stands with roles swapped (live = default, static = control).

---

**Strategy:** overruled and recorded; the argument for the flip is sound — evidence over intent, at zero training cost, with the intent kept alive as the control. The serving-cache figure (2K×) is the price and is now written where nobody can miss it. **Coding agent:** bind `kv_policy` with default `live` into step 2; keep the static path as the switchable control; add the receipt lines and the K = 1 identity test. **Mark:** recorded; nothing further.
