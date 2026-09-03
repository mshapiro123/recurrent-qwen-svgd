# STRATEGY — D-MC-1 Ratification Record: L_stage Decodes the Final Visit Plus One Sampled Visit

**Date:** 2026-09-03 · **Status:** RATIFICATION RECORD for the math check (16,587 B, SHA-256 `509cac8c…3dedcff`) §3. Mark ratified option (b) on 2026-09-03. Amends handoff §5.1 (the `step_logits` loop) and §6.2 in place; catch #38 is closed by this record.

| decision | Mark's ruling | binding text |
|---|---|---|
| **D-MC-1** | **(b) Final + one sampled earlier visit** (as recommended) | In training, per micro-batch: the **final executed visit** is decoded through the shared coda for `L_LM`, and **exactly one earlier visit `j ~ Uniform{0, …, K_exec − 2}`** is decoded through the shared coda for `L_stage`, drawn from the registered O-9 stream `weft.lstage.sample` (per-module generator, replayable). `L = L_LM(final) + λ_stage · L_stage(j)` — the single sampled term carries weight `λ_stage` so the estimator is unbiased for the uniform average over earlier visits (`Σ_k w_k = 1` in expectation). At `K_exec = 1` no earlier visit exists and `L_stage` is zero for that example — recorded, not padded. |

**Consequences bound with it.** (i) Training compute per token is `≈ 6 × (N_prelude + K·N_recurrent + 2·N_coda)` — **1.24× the previously accounted figure at K = 4** (1.32× at K = 2, 1.19× at K = 6); the compute allocation (D-CUR-4, ≈ 234 A100-hr all-in) is **re-derived by the agent with this multiplier** and reported before S2; the pre-registered de-scope order (rung B first) applies if the re-derived total exceeds the allowance. (ii) The composition receipt gains **`coda_decodes_per_step`** (= 2 under this ruling) and **`lstage_sampled_visit`** per micro-batch. (iii) The halting head and EXTRAP-K are unaffected: inference decodes once, at the halted visit. (iv) The STOCH-K arm composes with this rule unchanged — the sampled visit is drawn from whatever `K_exec` the step used. (v) Option (a), full per-visit decoding, is recorded as a **registered contrast** (`LSTAGE-FULL`) for the exploration allocation, not the default: if the sampled estimator's variance measurably hurts the loop gain `η_k`, that is the arm that says so.

**Also bound by the math check, unchanged by this decision:** catch #37's engram gate form (memory-space dot, `/√d_m`); T2 asserts nonzero `∂L/∂dU` and `∂L/∂dV`; T4 asserts `perm[k] = bitrev(gray(k))`; the `2⁻ᵖ` convention stays, the round-trip exactness claim is withdrawn.

---

**Strategy:** ratified as recommended; the objective keeps its purpose — a coda that can decode any depth — at a quarter more compute instead of double. **Coding agent:** bind the sampled decode into the step-6 spec with its O-9 stream, add the two receipt lines, re-derive the allocation, and report the engram-gate form and the T2/T4 assertions. **Mark:** recorded; nothing further.
