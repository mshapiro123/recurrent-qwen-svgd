# STRATEGY — D-HD-1 Ratification Record: Three Hemisphere-Divergence Receipts

**Date:** 2026-09-03 · **Status:** RATIFICATION RECORD for the hemisphere-divergence note (11,068 B, SHA-256 `687c2a78…c9f3b`) §4. Mark ratified option (a) on 2026-09-03. Receipt-schema additions only; no arm, no default change, no compute. **Addendum to the 2026-09-03 handoff packet (8,679 B, `07687b98…2b1f`): append as R8 and add the three lines below to §3 steps 2 and 6.**

| decision | Mark's ruling | binding text |
|---|---|---|
| **D-HD-1** | **(a) Add the three receipts** (as recommended) | **(1) `rho_hat_free`** — the hemisphere residual correlation `ρ̂(A,B)` as defined for `L_div` (§5.5), logged at every eligible step from the calibration gate onward **in every arm**, including `λ_div = 0`; the `λ_div = 0` value is the *free* point against which the `ρ*` grid is read. **(2) `delta_ratio`** — per paired matrix (`SwapLinear`), `‖dU dVᵀ‖_F / ‖μ‖_F`, logged per eligible step, with **tripwire SYM-COLLAPSE:** if any paired matrix's `delta_ratio` stays below its value at initialization for a window of 1,000 consecutive steps, the line stops and reports (the symmetric point is absorbing: all swap-symmetric gradients, `L_div` included, vanish at `δ = 0`). **(3) `lateralization_index`** — per callosum band `b`, `ℓ_b = sin 2θ_b ∈ [−1, 1]` from the S-2 combiner's `θ_b`, logged per eligible step; `0` = consensus read, `±1` = band read from hemisphere A / B alone. |

**Placement.** (1) and (3) live in the composition receipt family; (2) in the per-step parameter receipt with the existing tripwires (`cos(∇_A L, ∇_B L)`, §5.6.3). Step 2 (bicameral block, S-2 combine) carries (2) and (3); step 6 (objective stack, `L_div`) carries (1). The init value for (2)'s window is taken from the T7 receipt.

**Not bound.** Option (b) — anchoring the `ρ*` grid to the measured free value — is not adopted; the `ρ* ∈ {0.3, 0.5, 0.7} ∪ {λ_div = 0}` sweep stands as pre-registered and is *read* against `rho_hat_free`. NOISE-SHARED remains an offered WEFT-2 seed rule, not a WEFT-1 binding.

---

**Strategy:** ratified as recommended; three lines, no compute, and the question "do the hemispheres specialize?" becomes a number per band instead of an argument. **Coding agent:** add the three receipt fields and the SYM-COLLAPSE tripwire at steps 2/6; append this record to the packet as R8. **Mark:** recorded; nothing further.
