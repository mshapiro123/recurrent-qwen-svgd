# STRATEGY — PRE-FLIGHT Amendment PF-1: Catches #22–#25 Adjudicated and Bound

**Date:** 2026-09-02 · **Status:** AMENDMENT to the PRE-FLIGHT program (15,575 B, `ceaa5338…f88a5b`) and its ratification record (2,233 B, `4a13054d…f66965`), adjudicating `CODING_TO_STRATEGY_WEFT1_PREFLIGHT_CATCHES_20260901.md` (9,685 B, SHA-256 `94ba96d7…1f04a4`, commit `04eb5137`). **Read in full from the Drive copy before ruling** (PS-1). Where PF-1 and the program conflict, PF-1 governs.
**Ledger:** catches **#22, #23, #24, #25 accepted with the agent's numbering. All four are strategy's** — two are outright errors in documents I wrote (#22 in the Jacobian handoff, #23 in the program), one is a criterion I wrote without checking eligibility (#24), one is a claim about the codebase I made without reading it (#25, a PS-1 violation against our own repository). **Eighth consecutive correct fail-closed stop.** The gate earned its existence on its first firing, exactly as the report says.
**Nothing here requires a decision from Mark.**

---

## 0. Plain-language summary

The pre-flight program was built to catch errors before training; its first run caught four, and all four were mine. The most consequential is #22: the Jacobian handoff mixed logarithm bases — it regresses a natural-log response on a base-2 depth coordinate — so the panel would have reported every exponent scaled by ln 2, turning a planted 1.0 into 0.693. That is exactly the headline number this program will compare against GRT's external prior, and it would have been wrong by a factor that happens to look like a plausible result. The fix is a coordinate change and a corrected design constant; the power calculation's binding conclusion survives, but its measurement-noise term grows by 44 percent and the agent will re-run it to confirm.

Catch #23 is a certificate that was not one: the loop Lipschitz bound applied the sidecar gate twice, which for gates below one produces a number *smaller* than the true operator norm — a one-line counterexample shows the "bound" at 1.25 against a true norm of 1.5. Worse, it named an empirical power-iteration estimate as if it were a provable upper bound, and it omitted live branches of the implemented visit. The corrected certificate separates what can be certified — every adapter factor has an exact or provable bound — from what can only be estimated — the core block, which is reported as an estimate with a convergence check and never called a certificate. Catch #24 is a criterion that could not be met: the re-entry bridge deliberately does not execute on visit zero, so at K = 1 its parameters correctly have no gradient. The criterion becomes an eligibility matrix, with the inverse test added — a tensor that is ineligible at K = 1 must become live at K = 4, or it is a frozen parameter hiding behind eligibility. Catch #25 is deferred outright: the delta-rule test referenced a code path that does not exist, and pre-flight invents no architecture.

The coverage disclosures deserve their own sentence: the production graph does not yet contain the integrated rotor carrier, the per-band callosum, or the sidecar. Standalone certificate tests may claim exactly what they test and nothing about the integrated graph. Strategy needs a build-status matrix of ratified modules against implemented ones, and asks for it as a receipt.

---

# 1. Catch #22 — the headline exponent's coordinate (accepted; strategy's error, Jacobian handoff)

**Verified:** regressing `ln|λ_T|` on `log₂T` returns `p·ln 2` (planted 1.0 → 0.693147; 1.5 → 1.039721); on `ln T` it returns `p` exactly. `Sxx` over `T ∈ {1,2,4,8}` is 5.0 in base 2 and **2.4022650695910066** in natural log. Exact, deterministic bias — not a tolerance question.

> **PF-1.1 — literals bound (the agent's recommendation 1–5, adopted verbatim).** (1) canonical coordinate `x = ln T`; (2) `Sxx = 2.4022650695910066` for `T = (1,2,4,8)`; (3) PT1's response becomes a **physical plant** `λ_T = −c·T^{−p}` — a test that constructs its response from the regressor tests algebra, not the law; (4) `p` is the base-independent exponent of `α(T) = c·T^{−p}`; (5) the `n₀ = 32` pilot stays diagnostic-only and the 20-replicate coverage gate stands. The GRT prior `p ∈ [1.2, 1.7]` (GV-1) is base-independent and unaffected.
>
> **PF-1.2 — power calculation re-run.** `SE(p̂) = √(σ_slope²/n + σ_w²/(Sxx·n))`: the measurement-noise term's SE contribution grows by `√(5/2.4023) = 1.443×` under the corrected coordinate; the `σ_slope` term is unchanged. The power document's conclusion (σ_slope binding, compute irrelevant, n = 512) is *expected* to stand — but expectation is not a receipt: **the agent re-runs `jacobian_power3.py` with `Sxx = 2.4022650696` and reports whether n = 512 stands.** If it does not, the new `n` returns through this amendment path.
>
> **PF-1.3 — the C-JAC-2 adjacent item, bound now.** `σ_slope_hat` may not subtract raw-probe variance after P-5 paired probes and a nonlinear response. B1 proceeds in **two phases**: **phase 1**, zero-measurement-noise control — validates the coordinate, the estimator, and coverage with `σ_w = 0`; **phase 2**, noisy coverage under the **paired leave-one-probe-out jackknife**: per example `i` with paired probes `j = 1…n_p`, compute the Theil–Sen slope `ŝ_i` on all probes and `ŝ_i^{(−j)}` on each leave-one-out set; `σ̂_{w,i}² = ((n_p−1)/n_p)·Σ_j (ŝ_i^{(−j)} − s̄_i)²`; then `σ̂_slope² = Var_i(ŝ_i) − mean_i σ̂_{w,i}²`, clipped at 0 with the existing clipped flag; cluster bootstrap over examples unchanged. Phase 2's coverage claim is the one that counts.

# 2. Catch #23 — the certificate that was not one (accepted; strategy's error, program A3/A5)

**Verified:** ungated mixture norm 1, gate 0.5 — true `‖I + ΔW‖ = 1.5`, A5-as-written 1.25, corrected `1 + |g|‖U‖ = 1.5`. The agent's three objections all hold: the double gate, the power-iteration-as-upper-bound conflation, and the omitted live branches.

> **PF-1.4 — corrected certificate, literals bound (the agent's 1–7, adopted, with bound sources supplied).**
> **(1)** `U_k` = pre-gate expert mixture; `ΔW_k = g_k U_k` = applied update. **(2)** Spectral 2-norm throughout. **(3)** Top-3 weights non-negative, L1 sum recorded. **(4)** Sidecar factor = `1 + |g_k|·‖U_k‖₂` — **never both gates.** **(5) Certified bound sources, one per live factor:** sidecar `‖U_k‖₂ ≤ Σ_e w_e‖A_e‖₂‖B_e‖₂` (exact SVDs of d×4 factors; verified: mixture norm 49.8 under bound 80.4 at d = 64); callosum `‖A_ρ‖₂ = max(1, |1−2ρ|) = 1` on ρ ∈ [0,1] (exact); rotor: exactly 1 **only if orthogonality is certified by construction** (Cayley/exponential parameterization of the Cl(2,0) primitive), otherwise its exact norm is computed — a rotor is small and structured, so exact is cheap; re-entry bridge, scratch update/injection, loop embedding: each is linear or gated-linear — exact spectral norm of its weight, times `|gate|` where gated; each core block's two residual sublayers: **no provable global bound exists** (softmax attention is not globally Lipschitz), so the core enters as `Λ̂_core` — an **empirical estimate, labeled as such, never called a certificate** — from power iteration with a convergence receipt (Rayleigh-quotient sequence, last-step relative change < 1e-3, iteration count) and a paired randomized lower bound. **(6)** The implemented-visit certificate **enumerates every live factor that exists in the graph today** (re-entry, scratch, loop embedding, core blocks); absent modules (integrated rotor, per-band callosum, sidecar) appear as **named placeholders with their bound formula pre-bound**, activated when the module lands. The production line therefore emits two numbers: `Λ_adapters` (certified) and `Λ̂_core` (estimate). **(7)** The `∏Λ_k > e^{cL}` flag waits for ratified `cL` and certificate topology; until then the receipt logs both numbers and flags nothing.

# 3. Catch #24 — liveness versus eligibility (accepted; strategy's criterion)

**Agreed:** `reentry_bridge.*` correctly has `grad is None` at K = 1 because visit zero never executes it; the jet-conditioned sidecar is likewise ineligible until the second-order jet exists. Correct execution, not a frozen parameter.

> **PF-1.5 — eligibility-aware liveness, literals bound.** An explicit **parameter-eligibility matrix by (module, K, visit)** is authored and committed with the test. Requirement: every **eligible and executed** trainable tensor has a non-`None`, not-identically-zero gradient; ineligible tensors are reported separately with the structural reason. K = 1 tests modules that execute on visit zero; K = 4 tests re-entry and jet-conditioned modules. Receipt: per-module minimum eligible gradient norm, ineligible names with reasons; fail on any eligible tensor with `grad is None` or identically zero. **Added — the inverse test:** every tensor ineligible at K = 1 must be **eligible and live at K = 4**; a tensor ineligible at every tested K is a fail, not a disclosure — that is the frozen-parameter case wearing eligibility clothing.

# 4. Catch #25 — the code path that did not exist (accepted; strategy's PS-1 violation)

I asserted a shared delta-rule path without reading the tree. `ReadOnlyLatentMemory` is immutable retrieval; no `(β, η, λ, S)` update exists.

> **PF-1.6 — A8 deferred.** Struck from PRE-FLIGHT. Its certificate tests (spectrum `[1−β, 1−ηλ]`, cap `‖S̄‖ ≤ ‖y‖_max/(2√λ)`, K-INV, and the unnormalized-rate positive control) travel with the **MEM-SYN-FW spec** when the successor queue reaches it, with the complete update equation, tensor orientation, domains, and gate placement bound there. Pre-flight adds no arms and invents no architecture — the agent's reading of the program is the correct one.

# 5. Coverage disclosures — accepted, and one receipt requested

A1, A2, A7 as disclosed: standalone certificate tests **claim exactly what they test**. **Standing language rule (PF-LANG):** no receipt or strategy document may say the *integrated production graph* has passed a certificate whose module is not yet integrated; the words are "standalone certificate passed; production integration absent." The disclosure that the **integrated rotor carrier, per-band callosum, and sidecar do not yet exist in the production graph** is material to the S2 timeline and is not a pre-flight matter — so: **receipt requested — a build-status matrix**: every ratified WEFT-1 module × {absent, standalone primitive, integrated, integrated + OBS-INV-tested}, with commit references. Strategy rules on the build queue from that matrix, not from inference.

# 6. What resumes

On verification of this amendment's bytes and hash: B1 phase 1 under PF-1.1/1.3, the power re-run (PF-1.2), the corrected certificate utility under PF-1.4 emitting `Λ_adapters` + `Λ̂_core` with placeholders, the eligibility-aware hunter under PF-1.5 with the inverse test, A8 struck. Week-1 order otherwise as ratified; the 5 A100-hr PRE-FLIGHT meter remains its own receipt domain; P-A untouched.

---

*Signature block*

**Strategy:** four for four against me, and the ledger says why that is the good outcome: #22 would have shipped a headline exponent wrong by ln 2 — a factor that looks like a finding — into a comparison with an external prior, and nothing downstream would have caught it. The certificate error is the subtler lesson: a bound that can be *smaller* than the truth is worse than no bound, because it is believed. PF-1.4's separation of "certified" from "estimated" is the honest shape, and it is the shape every future certificate in this program takes. #25 is a PS-1 violation against our own repository, which is the one source I can least excuse not reading.
**Coding agent:** the report was exemplary — reproduced evidence, exact literals requested, counterexamples that commute, and a disposition menu where a menu was the right answer. Verify bytes and hash, bind PF-1.1–1.6, and add the build-status matrix to the next receipt.
**Mark:** nothing to decide. The two-week program is doing precisely what it was funded to do, one day in.
