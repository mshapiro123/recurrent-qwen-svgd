# STRATEGY — What Makes the Hemispheres Diverge? Forces, a Micro-Experiment, and the Lateralization Dial

**Date:** 2026-09-03 · **Status:** DESIGN NOTE answering Mark's question ("what ensures the parallel paths deviate in their learning?"). Grounded in handoff §3.3, §5.5, §5.6.1, §5.6.3, T7 and the S-2 combiner ruling; **checked with a micro-experiment** (`symmetry_forces_20260903.py`, ten conditions, CPU, ~2 min). **No bindings**; one decision offered (D-HD-1) on cheap instruments. Critical path untouched.

---

## 0. Plain-language answer

Nothing *forces* the hemispheres apart, and it turns out nothing has to. The design has four mechanisms, and the micro-experiment says the third is doing almost all the work.

First, symmetry is broken at initialization at a registered magnitude (`δ ≠ 0`, both factors nonzero, T7). This is necessary and not sufficient: the perfectly symmetric configuration `δ = 0` is an *absorbing* state — every swap-symmetric loss, including our own diversity term, has zero gradient there — so if the hemispheres ever meet, nothing brings them back. Second, the optimizer works in `(μ, δ)` coordinates, so Adam normalizes the disagreement axis by its own small variance — a measured ~5× amplification of whatever gradient the disagreement receives. Third — and this is the finding — the combiner reads the *mean* of the two hemispheres at initialization, and the mean is blind to any disagreement that cancels. The loss does not merely tolerate divergence; it *rewards* it, because two estimates whose errors are anti-correlated average to a better answer than either. In the toy, with no diversity term, no noise and weight decay on `δ` at the full 0.1, the disagreement grows from 0.8 % of the consensus weights to 66 % and the hemispheres' errors go to correlation −0.8. Divergence is the default dynamic of an averaged pair, not something to engineer. Fourth, `L_div` with an interior target and the separate decay `λ_δ` are *brakes and steering*, and the callosum's `ρ < ½` bound guarantees the exchange can never annihilate the disagreement in the state.

What the model does **not** have is what stereoscopic vision has: two *different views* of the same input. Both hemispheres receive the same `h₀`; their difference is entirely learned. Two toy results speak to whether that should change. Independent per-hemisphere noise — a natural reading of the VAE analogy — is a *homogenizing* force: it pushed the error correlation from −0.8 to +0.8 and, combined with the other terms, drove the pair all the way into the absorbing symmetric state. A fixed structural parallax (each hemisphere sees a different 75 % of the input) produced strong divergence but a *worse* combined answer, because each side lost information a single path would have had. The biology lesson transfers with a caveat: parallax pays only when neither view alone contains the answer (depth needs two eyes); masking a view that a single path could have used just costs.

The last point is about what "specialization" means. Anti-correlated errors are ensemble diversity — two estimates whose disparity carries information — and the mean combiner gets that for free. "Separate expert focus" — one hemisphere handling something the other does not — is different, and the mean cannot express it. The S-2 combiner can: `y_b = cos θ_b μ_b + sin θ_b δ_b` is, at `θ_b = +π/4`, exactly `o_A/√2` and at `−π/4` exactly `o_B/√2`. **`θ_b` is a per-band lateralization dial**, and `sin 2θ_b ∈ [−1, 1]` is a lateralization index we can read off the trained model band by band. The toy's `θ` settled near 0.06–0.1 rad (weak lateralization) because averaging was optimal there; in the language model, bands where one hemisphere wins outright would show `θ_b` moving toward ±π/4. That instrument does not exist in the receipts yet, and it is the one that would answer Mark's question empirically rather than by argument.

---

# 1. The forces, in the design's own terms

| force | direction | where bound | note |
|---|---|---|---|
| `δ ≠ 0` init, `σ_δ0 = 0.02` both factors | breaks symmetry once | §5.5 (1), T7, PF-3.1 | `δ = 0` is absorbing: all swap-symmetric gradients vanish there (verified: `∂L/∂dU` at the symmetric point is 0 to numerical precision; `L_div` included) |
| mode-wise Adam on `(μ, δ)` | amplifies disagreement updates `√(1 + σ_μ²/σ_δ²) ≈ 5×` | §3.3, §5.5 (2)–(3) | noise on `δ` under AdamW equilibrates at `δ_rms ≈ √(η/2λ) ≈ 0.04` — comparable to init; Adam alone prevents collapse but does not make the difference meaningful |
| mean combiner's null space (`θ = 0`) | **rewards** anti-correlated errors; disagreement that cancels is free | S-2 | the dominant force in the toy: `‖δ‖/‖μ‖` 0.008 → 0.66, `ρ̂ → −0.8`, no `L_div` |
| `λ_δ` decay | toward symmetry, first order | §5.5 (4) | at 0.1 it is beaten by the null-space drift (C1 vs C2: 0.66 vs 0.69) |
| `L_div = λ_div (ρ̂ − ρ*)²`, `ρ* ∈ {0.3, 0.5, 0.7}` + `λ_div = 0` | pins `ρ̂` at the target | §5.5, §6 | pins exactly (C4: `ρ̂ = 0.49`, `‖δ‖/‖μ‖ = 0.08`); **all three targets sit on the *less-diverse* side of the free point if the free `ρ̂` is negative** — see §3 |
| callosum `ρ_b ≤ ½` | damps state disagreement by `(1−2ρ)^K`, never to zero | §5.6.1 | a bound on the exchange, not a driver |
| gradient-coupling arm A (stop-grad both directions) | prevents co-adaptation *through the gradient* | §5.6.3 | tripwire `cos(∇_A L, ∇_B L) → 1` |
| live K/V (D-NB-1) | each hemisphere attends over its own memory | R6 | removes a homogenizing tie the static policy had (both attended over identical K/V) — an unremarked consequence of the flip, positive for divergence |
| per-lane sidecar adapters `A_e, B_e` | parametric asymmetry in operator reads | S-4′ | same class as `δ`: learned, not structural |

# 2. Micro-experiment (two-layer MLP pair, SwapLinear, S-2 combiner, AdamW, nonlinear teacher, MSE)

| condition | loss | `‖δ‖/‖μ‖` | `ρ̂(A,B)` | `|θ|` |
|---|---|---|---|---|
| C1 as ratified (`θ₀ = 0`, `λ_δ = 0.1`, no `L_div`, no noise), 3k steps | 0.0228 | 0.59 | −0.73 | 0.08 |
| C1L same, 9k steps | 0.0213 | 0.66 | −0.80 | 0.06 |
| C2 `λ_δ = 0` | 0.0225 | 0.69 | −0.77 | 0.07 |
| C3 `θ₀ = 0.05` | 0.0227 | 0.65 | −0.82 | 0.09 |
| C4 `L_div`, `ρ* = 0.5` | 0.0243 | 0.08 | 0.49 | 0.05 |
| C4b `L_div`, `ρ* = 0` | 0.0248 | 0.12 | −0.02 | 0.06 |
| C5 independent entry noise `σ = 0.3` per hemisphere | 0.0271 | 0.35 ↓ | **+0.81** | 0.01 |
| C6 C3 + C4 + C5 | 0.0558 | **0.007** | **0.999** | 0.004 |
| C7 structural parallax (fixed different 75 % input subsets) | 0.0343 | 0.56 | −0.87 | 0.08 |

Readings. (i) **Divergence is the default**: the free pair goes strongly anti-correlated and keeps going (C1 → C1L). (ii) **`θ₀ = 0` is not a trap**: `∂L/∂θ = ⟨g, δ_out⟩` is `O(‖δ‖)` and nonzero from step 1 because `δ ≠ 0` at init; Adam normalizes it; `θ` moves immediately (C1 vs C3 differ little). S-2 stands. (iii) **Independent stochasticity homogenizes** (C5) and, with other symmetric pressures, **collapses the pair into the absorbing state** (C6) — from which no swap-symmetric loss recovers it. (iv) **Structural parallax by masking costs accuracy** (C7): each view lost information a single path would have used. (v) `L_div` does exactly what it is told (C4, C4b) at a small accuracy cost in this toy — its value in the language model depends on where the free `ρ̂` sits, which is the point of §3.

Caveats, stated: a 2-layer regression toy with a linear path from the combiner to the loss; no callosum, no loop, no shared K/V, no coda. The *signs* of the forces transfer (they are properties of the averaged-pair objective and of swap symmetry); the *magnitudes* do not.

# 3. Two consequences worth acting on, and one rule for the VAE idea

**(a) The `ρ*` sweep is anchored to a retrofit number.** The handoff prices the ensemble gain "at the frozen 0.7446" — the hemisphere correlation measured on a frozen-substrate retrofit, where almost all weights were shared and `ρ̂` was necessarily high. In a from-scratch pair with the mean combiner the free `ρ̂` may be *negative* (toy: −0.8). If so, `ρ* ∈ {0.3, 0.5, 0.7}` are three ways of making the hemispheres *more alike* than they would be on their own, and the `λ_div = 0` arm is the most-diverse arm of the sweep, not the least. The sweep is still well-formed (the free arm is in it) but its *reading* inverts. **The fix is an instrument, not a change:** log the free `ρ̂(A,B)` from the calibration gate onward in the `λ_div = 0` arm, and interpret the `ρ*` grid relative to it.

**(b) The absorbing state needs a tripwire.** T7 checks `‖δ‖ > 0` at init only. Nothing watches for the pair *meeting* later — and C6 shows it can happen under ordinary-looking pressures. A `‖δ‖/‖μ‖` receipt per paired matrix, with a stop if it falls below its init scale over a window, is the cheap guard.

**(c) The lateralization dial.** `θ_b` per band, reported as `ℓ_b = sin 2θ_b ∈ [−1, 1]`, is the direct readout of "expert focus" per band: `0` = consensus, `±1` = one hemisphere owns the band. It is one line in the composition receipt and it is the instrument that answers Mark's question on the real model.

> **Rule offered for the WEFT-2 seed (NOISE-SHARED):** any stochastic bottleneck in the VAE sense — a sampled latent, dropout on the state, a noisy carrier — applies **the same sample to both hemispheres**, or sits **after** the combiner. Independent per-hemisphere noise is a homogenizing force (C5/C6) and is registered as such; it is not the mechanism by which stochasticity would help a bicameral pair. The biology reading that survives: the two eyes see *different, deterministic* views of *one* world, not one view with two independent noises.

# 4. D-HD-1 — decision for Mark (instruments only; no arm, no default change)

*(a) Add the three receipts now — recommended.* `rho_hat_free` (from the calibration gate, all arms), `delta_ratio` per paired matrix with the collapse tripwire (stop if below init scale over a 1k-step window), `lateralization_index` `ℓ_b` per band. Step 2/6 schema lines; no compute. *(b) Also anchor the `ρ*` grid to the measured free value* (e.g. `ρ* ∈ {ρ̂_free + 0.2, +0.4, +0.6}` clipped to `[−1, 1]`, set at the calibration gate) — makes the sweep meaningful whichever sign the free point has, at the cost of a grid that is not fixed pre-launch (pre-registered *rule* rather than pre-registered *values*). *(c) Nothing now; revisit after S5's first `ρ̂` read.*

---

**Strategy:** the honest answer to "what ensures they deviate" is "the averaging objective, and it does so vigorously" — the design's job is to keep the deviation *informative* (the dial) and to keep the pair from ever meeting (the tripwire). **Coding agent:** nothing until D-HD-1. **Mark:** D-HD-1.
