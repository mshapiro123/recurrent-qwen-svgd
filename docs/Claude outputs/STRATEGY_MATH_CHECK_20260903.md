# STRATEGY — Math Check, Work Shown: Four Claims From the Walkthrough, Verified — Two Catches Against the Handoff

**Date:** 2026-09-03 · **Status:** MATH VERIFICATION for Mark and the coding agent. Companion script `math_check_20260903.py` (7,586 B, SHA-256 `9dbe3724…08195b`) reproduces every number below deterministically (`python3 math_check_20260903.py`, float64, seeded).
**Scope:** the four claims not previously verified numerically — (1) bicameral symmetry breaking, (2) the engram gate's scale, (3) L_stage's effect on the shared coda and on compute, (4) the WHT's involution, sequency ordering, and round-trip exactness — plus the register of what was already verified this week.
**Ledger:** **catch #37** (engram gate divisor, handoff §5.11) and **catch #38** (per-visit coda decode not in the compute plan, handoff §5.1/§6.2 + D-CUR-4) — both strategy's. One sequency-ordering resolution the handoff explicitly asked for. One minor correction to a rounding claim. **One decision for Mark (D-MC-1, §3).**

---

## 0. Plain-language summary

All four claims hold in the form the design needs, and two of them exposed something wrong in my own handoff. The symmetry-breaking claim is exactly right: if the two hemispheres start identical they stay identical forever, because the gradient on the difference is the difference of two identical gradients — zero — and nothing else in the loop (shared keys and values, a symmetric callosum, a consensus-only combiner at initialization) can pull them apart. The design's rule that the difference must be nonzero at initialization is therefore load-bearing, not cautious, and the callosum's bound on its mixing strength is what stops it from erasing the difference once it exists.

The engram gate as written in the handoff divides a 1024-dimensional inner product by the square root of 64. That puts the gate's logit at four standard deviations instead of one, and about forty-six percent of gates start saturated at zero or one — a cold-start defect of exactly the kind the gating law warns about. The fix is to divide by the square root of the dimension you actually dotted, or to take the dot in the memory space; the formula meant the latter and now says it.

The staged-solution objective decodes every visit through the full shared coda. That is the right training signal — it is what lets the model be decoded at any depth — but the compute plan was built on a model whose coda runs once, and decoding it at every visit roughly doubles training compute at K = 4. That is a real hole in the allocation, and I have put the choice to Mark: pay it, or decode the final visit plus one randomly sampled earlier visit per step, which covers every visit in expectation at about a quarter more compute instead of double. The per-visit weights also need to sum to one, or the coda's gradient grows linearly with depth.

The Walsh–Hadamard facts check: the transform is an exact involution, and of the four candidate orderings the handoff said to test, the one that puts exactly k sign changes in row k is bit-reversal of the Gray code — not Gray code of the bit-reversal, which is what the prose said. The handoff anticipated this and asked for the passing variant to be reported; it is reported. One overclaim is trimmed: scaling by 2⁻ᵖ is exact, but the butterfly sums still round, so the round trip is not bit-exact in fp32 — both scalings give the same 7×10⁻⁷ error.

---

# 1. Check 1 — bicameral symmetry: δ = 0 is a gradient-dead fixed point; ρ < ½ cannot re-symmetrize

**Claim (handoff §5.5 rule 1):** "Symmetric start ⇒ identical hemispheres forever under any swap-symmetric gradient."

**Derivation.** Store `W_A = μ + δ`, `W_B = μ − δ`. By the chain rule, `∂L/∂δ = ∂L/∂W_A − ∂L/∂W_B` and `∂L/∂μ = ∂L/∂W_A + ∂L/∂W_B`. Suppose the hemispheres coincide: `δ = 0` and `h_A = h_B` at loop entry (true by construction, `h_A = h_B = h₀`). Every operation in the visit is swap-symmetric — attention against the *shared* K/V with `Q` from identical states, identical FFN weights, lane updates reading identical states, the callosum `A(ρ)` which fixes the consensus and scales the disagreement (zero stays zero), and `combine` at `θ = 0` which reads consensus. By induction `h_A ≡ h_B` at every visit, so `∂L/∂W_A = ∂L/∂W_B` and **`∂L/∂δ = 0`**. With `δ = dU dVᵀ`: `∂L/∂dU = (∂L/∂δ) dV = 0` and `∂L/∂dV = (∂L/∂δ)ᵀ dU = 0` — and note the trap the design already names: even if `∂L/∂δ ≠ 0`, `dU = 0` kills `∂L/∂dV` and vice versa. Hence **both factors nonzero at init** is necessary, not decorative.

**Second claim:** for `ρ < ½` the callosum cannot re-symmetrize. In the swap eigenbasis the callosum leaves `μ_state` fixed and multiplies `δ_state` by `(1 − 2ρ) ∈ (0, 1]`; it never touches the *weight* `δ`. So even if the state disagreement is contracted every visit, the next visit's paired weights regenerate it from the shared input; the only absorbing configuration is `δ_weight = 0`, which the gradient above shows is reachable only if it is already there. `ρ = ½` annihilates the *state* disagreement per band per visit (the collapse mode §5.6.1 names) but still not the weights.

**Numerical verification** (toy: d = 32, rank 4, T = 6, K = 3, shared K/V from h₀, callosum ρ = 0.3, combine at consensus; float64):

| δ init scale | ‖h_A − h_B‖ after K = 3 | ‖∂L/∂dU‖ | ‖∂L/∂dV‖ | ‖∂L/∂μ‖ |
|---|---|---|---|---|
| 0 | **0.000e+00** | **0.000e+00** | **0.000e+00** | 7.2e+02 |
| 1e-3 | 2.0e-04 | 3.3e-05 | 3.6e-05 | 7.2e+02 |
| 2e-2 | 8.1e-02 | 2.6e-01 | 2.8e-01 | 7.2e+02 |

Exactly zero gradient at exact symmetry while `μ` trains normally — the hemispheres would train as one model forever. Callosum sweep at δ-scale 2e-2: terminal ‖h_A − h_B‖ = 0.357 (ρ = 0), 0.111 (0.25), 0.016 (0.45), **0.000 (0.5)** — the closed form `(1−2ρ)` per visit, confirmed.

> **Verdict: holds.** Two consequences made explicit: (i) `σ_δ0 = 0.02` on *both* factors (§5.5) is the entire symmetry-breaking mechanism — the T2 liveness test must assert nonzero `∂L/∂dU` and `∂L/∂dV` at step 1 specifically, not merely nonzero `∂L/∂μ`; (ii) under separate weight decay `λ_δ` with a weak diversity gradient, `δ` drifts toward the absorbing point — `L_div`'s interior target is what holds it off, and the `ρ̂(A,B)` tripwire is the instrument.

# 2. Check 2 — the engram gate's scale (catch #37)

**Claim (handoff §5.11):** `g_t = σ( RMSNorm(h¹_t)ᵀ RMSNorm(W_K e_t) / √d_m )` with `d_m = 64` the memory width.

**Derivation.** RMSNorm output has per-coordinate RMS 1, so a vector in `R^n` has norm `√n`. The inner product of two *independent* such vectors has mean 0 and standard deviation `√n`. If `W_K e_t ∈ R^d` (the natural reading — `W_K` lifts the 64-dim engram row to the residual width so it can be dotted with `h¹_t ∈ R^1024`), the dot is over `n = d = 1024` and its standard deviation is 32. Dividing by `√d_m = 8` leaves a logit with **standard deviation 4** at initialization; `σ(N(0, 4²))` is saturated (`< 0.05` or `> 0.95`) with probability ≈ 0.46. That is the §5.13 cold-start trap in gate form: half the engram's gates start pinned, and a pinned gate's table rows receive gradient scaled by `σ′ ≈ 0`.

**Numerical verification** (20,000 draws, d = 1024, d_m = 64):

| divisor | logit std | gate mean | saturated fraction |
|---|---|---|---|
| `√d_m = 8` over a 1024-dim dot (as written) | **3.98** | 0.50 | **0.459** |
| `√d = 32` over the 1024-dim dot | 1.00 | 0.50 | 0.004 |
| dot taken in `R^{64}`, then `/√d_m` | 1.00 | 0.50 | ≈ 0 |

> **Catch #37 (strategy's; handoff §5.11).** The divisor must be the square root of the **dimension of the inner product**. **Bound:** `g_t = σ( ⟨ RMSNorm(W_Q h¹_t), RMSNorm(e_t) ⟩ / √d_m )` with `W_Q : R^d → R^{d_m}` a learned query projection (64 K parameters, hidden-class μP) and the dot taken **in memory space** — cheaper than a 1024-dim dot, and the reading the symbol `d_m` was evidently meant to express. The value path is unchanged: `h¹_t ← h¹_t + γ_m · cap(W_V (g_t e_t))`. **Agent:** report which form `CausalTokenEngram` implements today; if it is the as-written form, it is a bring-up defect with a one-line fix and a T2 liveness consequence.

# 3. Check 3 — L_stage: the shared coda's gradient and the compute plan (catch #38)

**Claim (handoff §5.1, §6.2):** in training, every visit's combined state is decoded through the **shared** coda to produce step logits, and `L_stage` aligns each visit toward the answer. This is functionally necessary — inference-controllable K and halting require a coda that can decode *any* visit's state — so it is not an auxiliary to be dropped.

**Derivation, gradient.** With `L = L_LM(final) + λ Σ_{k<K} w_k L_k`, the coda parameters receive `∂L/∂W_coda = ∂L_LM/∂W_coda + λ Σ_k w_k ∂L_k/∂W_coda` — `K + 1` decoding paths through the same weights. With `w_k ≡ 1` the coda's gradient grows like `K`; with `Σ_k w_k = 1` it is `K`-invariant in expectation. Verified (toy coda, per-visit states): unnormalized ‖∇‖ = 0.19, 0.28, 0.46, 0.77 at K = 1, 2, 4, 8; normalized 0.19, 0.14, 0.12, 0.10. Also worth naming: the step heads give the core a *direct* gradient at every visit that does not pass through later visits — a shortcut path that is stabilizing relative to the pure through-loop product `∏(I + α_T J_k)`, and part of why L_stage matters beyond "answer-readiness."

**Derivation, compute.** The composition accounting (§9) and the compute allocation (D-CUR-4, ≈ 234 A100-hr) price a forward as `N_fixed + K·N_recurrent` — prelude once, core K times, **coda once**. Per-visit decoding executes the coda `K + 1` times. With prelude ≈ coda ≈ 105 M and `N_recurrent = 57.5 M`:

| K | as accounted | full per-visit coda decode | final + one sampled visit |
|---|---|---|---|
| 2 | 325 M | 535 M (**1.65×**) | 430 M (1.32×) |
| 4 | 440 M | 860 M (**1.95×**) | 545 M (1.24×) |
| 6 | 555 M | 1,185 M (**2.14×**) | 660 M (1.19×) |

Training compute is `≈ 6 × D × N_ae`, so the multiplier applies to the whole training budget, not to a margin. **The compute plan does not contain it.** (Inference is unaffected: the coda runs once on the halted visit.)

> **Catch #38 (strategy's; §5.1/§6.2 against D-CUR-4).** The ratified objective's cost was never priced. Two bindings regardless of the decision: **`Σ_k w_k = 1`** over decoded visits (or `λ_stage` scaled to the same effect), and the **composition receipt gains a line `coda_decodes_per_step`** so the accounting can never again assume one.
>
> **D-MC-1 — decision for Mark.** *(a) Decode every visit through the full coda* — the text as ratified; ≈ 2× training compute at K = 4; the allocation (both rungs + dense control ≈ 234 A100-hr) is re-derived and the pre-registered de-scope order (rung B first) is likely to fire. *(b) Decode the final visit plus one earlier visit sampled uniformly per micro-batch (recommended)* — every visit is trained in expectation, the coda still learns to decode any depth, `L_stage`'s weight on the sampled visit is `λ` (unbiased for the uniform average), cost 1.24× at K = 4; variance is the price, and the STOCH-K arm already lives with exactly this kind of sampling. *(c) Decode the final two visits only* — cheapest that still trains "one step before done" (useful for halting), but visits `< K−1` are never decoded, so inference at small K is untrained; not recommended while inference-controllable K is a promise.

# 4. Check 4 — WHT: involution, sequency ordering, and the rounding claim

**Claims (handoff §5.6.2):** `W_d W_d = d·I`; sequency (Walsh) ordering is required and "the permutation is Gray-code-of-bit-reversal, but do not trust that description — build `W_d`, permute rows, assert row `k` has exactly `k` sign changes; one of the four obvious variants passes — report which"; and `wht(wht(x))·2⁻ᵖ` is the correct round trip because `2⁻ᵖ` is exact in binary floating point while `1/√d` "rounds twice."

**Verification.** Involution: `torch.equal(W @ W, d·I)` at d = 8, 16, 1024 — exact. Sequency: of the four variants, **only `perm[k] = bitrev(gray(k))`** — the **bit-reversal of the Gray code of k** — gives row k exactly k sign changes at d = 8, 16, and 1024; `gray(bitrev(k))` (the prose's description) and both inverse permutations fail. Bands at d = 1024, E = 8: band b = Walsh sequencies `[128b, 128b + 127]`. Rounding: in fp32 on 4,096 random vectors, the `2⁻ᵖ` round trip and the `/√d`-twice round trip have **identical** maximum error 7.15e-7, and neither is bit-exact — the butterfly's 10 levels of additions round; the *scaling* by `2⁻ᵖ` is exact, the *sums* are not.

> **Resolution (requested by the handoff itself):** the sequency permutation is **`perm[k] = bitrev(gray(k))`**; T4 asserts it. **Minor correction:** keep `2⁻ᵖ` as the convention (exact scaling costs nothing), but the handoff's implication that the round trip is exact and the `/√d` form is "WRONG" is withdrawn — both are correct to fp32 rounding, identically. Not numbered; it changes no design.

# 5. Register — everything verified numerically this week

| claim | where | result |
|---|---|---|
| μR product bound `‖∏(I + α_T J_k)‖ ≤ (1 + cL/T)^T ≤ e^{cL}` | PRE-FLIGHT A4 | holds; products 1.43 → 1.11 vs bound 1.50 → 1.64 |
| callosum `A(ρ)`: `‖A‖₂ = 1` on ρ ∈ [0, 1]; difference scaled by `(1 − 2ρ)`; mean invariant | S5/S6, this check | holds |
| rotor isometry; bf16 drift over K = 8 (plain orthogonal) | PRE-FLIGHT A2 | 3e-4 norm drift, cosine 1.0000 |
| combiner: unit-norm read, `∂y/∂θ = δ` at θ = 0, bound `‖y‖ ≤ ‖μ‖ + ‖δ‖` | S5/S6 | holds |
| loop Lipschitz certificate factors (corrected, single gate) | PF-1.4 | holds; sidecar bound 49.8 ≤ 80.4 |
| Falcon = SGD on ridge; transition spectrum `[1−β, 1−ηλ]`; cap `‖S̄‖ ≤ ‖y‖/(2√λ)` | Falcon adjudication | holds to machine precision |
| address-stability floor `cos ≥ √(1 − λ²)`; unnormalized mixture collapses under drift | Kathleen adjudication | holds; adversarial MC at d = 1024 |
| additive vs sequential operator composition, cross-term bound | Latent Skills adjudication | holds at three scales |
| base-shape attention scale `√(d_base)/d_head = 1/√64`; readout multiplier `α_out/m` | §8.1 amendment, PF-3.1 | holds; width-stable activations across m ∈ {¼ … 2} |
| Jacobian panel: ln-coordinate, `Sxx = 2.4023`, planted p recovered | PF-1, agent's run | four planted phases green |
| **bicameral symmetry; callosum non-resymmetrization** | **this check** | **holds** |
| **engram gate scale** | **this check** | **as written: 46 % saturated — catch #37, bound** |
| **L_stage coda gradient; compute multiplier** | **this check** | **normalize weights; 1.95× at K = 4 unpriced — catch #38, D-MC-1** |
| **WHT involution, sequency permutation, rounding** | **this check** | **holds; `bitrev(gray(k))`; rounding overclaim withdrawn** |

# 6. For the coding agent

Verify the script's hash and run it (float64, seeded; ~20 s CPU). Then: (1) report which engram-gate form `CausalTokenEngram` implements — the dot's dimension and its divisor — and, if as-written, apply the §2 binding and re-run T2 liveness on the engram tables; (2) confirm T2 asserts nonzero `∂L/∂dU` **and** `∂L/∂dV` at step 1 in the paired block; (3) bind T4 to `perm[k] = bitrev(gray(k))` and report the current implementation's variant; (4) add `coda_decodes_per_step` to the composition receipt and the `Σ w_k = 1` normalization to the L_stage spec (step 6), pending D-MC-1 for which visits are decoded. Nothing here touches P-A, the sealed corpus, or any GPU cell.

---

*Signature block*

**Strategy:** the walkthrough asked "does each piece do what we say," and the math says yes — while catching that my own formula would have pinned half the engram's gates at initialization and that the objective I wrote doubles the training bill I also wrote. Both were findable only by doing the arithmetic rather than reading the prose, which is the argument for this kind of check before every run, not after the first one fails.
**Coding agent:** §6 is the list; all four items are build-axis.
**Mark:** one decision, D-MC-1 — pay the full per-visit decode or sample it. I recommend sampling.
