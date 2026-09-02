# STRATEGY — PRE-FLIGHT: Two Weeks of Formalism and Micro-Experiments to Harden WEFT-1's Operators Before Training

**Date:** 2026-09-02 · **Status:** PROPOSAL pending Mark's ratification of two items (§6). Everything else is registration under standing authority.
**Gating statement, stated first so nothing fails closed on it (W-1, SEQ-1):** every item below runs on **synthetic data or public toy tasks**, on **CPU or a small explicit GPU cap**, and consumes **no frozen vocabulary, no sealed corpus, no training-budget compute, no checkpoint**. It sits entirely on the build axis. It reorders nothing: P-A → attribution → P-B → G-TOK is untouched. Pre-flight must never slow P-A's CPU workers — it runs on idle GPU time or a separate CPU.
**The organizing principle:** *plant the effect, then check the instrument sees it; plant nothing, check it reports null.* Two weeks is exactly enough to point every Tier-1 instrument at a synthetic ground truth before it is ever pointed at the real model, and to turn every operator bound we have derived on paper into a unit test that runs on every commit.
**Every bound quoted here was verified numerically before writing** (μR product bound at T = 1…16, loop Lipschitz certificate, bf16 rotor drift at K = 8).

---

## 0. Plain-language summary

Two weeks of waiting on data is an opportunity, and the most valuable way to spend it is not on new architecture but on making the existing one *unable to surprise us*. Three kinds of work fit the window.

The first is turning theory into tests. This program has derived a stack of operator bounds — the callosum is nonexpansive, the rotor is an isometry, the sidecar's update is norm-capped, the loop's Jacobian product is bounded by an exponential in the recurrence gain — and every one of them can become a unit test on random and adversarial inputs that runs on every commit. The most important new object is a **loop Lipschitz certificate**: a single per-step number computed from the modules' own norms and gate values, provably an upper bound on how much the recurrent step can expand, logged during training and alarmed when it crosses a threshold. It converts the contraction assumption behind the recurrence law from something we hope holds into something the run reports every step.

The second is calibrating instruments before they see real data. Eight Tier-1 instruments and the Jacobian panel are about to produce the program's headline numbers, and none of them has yet been shown to detect a known planted effect or to report null when there is none. On a toy WEFT at width 64 — a million parameters, minutes per run — every one of them can be tested against synthetic ground truth: plant a known contraction exponent and see whether the panel recovers it with calibrated error bars; plant a responsibility leak and see whether the leak instrument reports it; plant no leak and see that it does not. An instrument that has passed this is one whose result at scale can be believed; one that has not is a source of catches waiting to happen.

The third is the cheap catches — the ones that cost hours to find now and A100-days to find later: a parameter whose gradient is identically zero at initialization (the frozen-carrier mistake this program has already made once, now hunted automatically); a modules-OFF forward that is not bit-identical to the dense baseline; a bf16 recurrence that drifts across eight visits; a width change that breaks the parameterization; a seed that does not replay. Each is a one-line test with a pass criterion, and the discipline is that a failure on a ratified operator returns to strategy through the amendment path rather than being fixed quietly.

Two decisions are Mark's: the compute cap for pre-flight GPU time, and whether instrument calibration or operator certificates get the first week if the agent cannot run both in parallel with P-A.

---

# 1. Track A — Operator certificates: theory becomes tests

Each item is a pencil derivation already in hand (or short) plus a test on random and adversarial inputs, fp32 and bf16, run on every commit. Pass criteria are literal.

| ID | operator / claim | test | pass |
|---|---|---|---|
| **A1** | **Callosum** `A_ρ = I − 2ρu_δu_δᵀ`: nonexpansive on ρ ∈ [0,1], no sign-flip on [0,½], difference contracts as `(1−2ρ)` along `u_δ`, lane mean invariant, orthogonal complement untouched | random lane pairs, adversarial `u_δ`, ρ sweep incl. endpoints | all five properties to 1e-6 (fp32), 1e-2 (bf16) |
| **A2** | **Rotor carrier** is an isometry (Euclidean) / preserves the boost invariant (T5 variant); retention `r` well-defined | random inputs, K = 1…8 composition | norm drift per visit < 1e-6 fp32; bf16 K = 8 drift measured and recorded (baseline: plain orthogonal rotor at d = 512 loses 3e-4 in norm, cosine 1.0000 — the *learned* composition must be measured, not assumed) |
| **A3** | **Sidecar update cap** `‖ΔW_k‖ ≤ g_k · max_e‖A_eB_eᵀ‖ · (top-3 sum)`; gate-only near-identity init | random experts, PQ selection, gate sweep | bound holds; gate = 0 ⇒ ΔW = 0 exactly |
| **A4** | **μR product bound** `‖∏(I + α_T J_k)‖ ≤ (1 + cL/T)^T ≤ e^{cL}` | random `J_k` with ‖J‖ = L, T ∈ {1,2,4,8,16} | verified today: max product 1.43 → 1.11 vs bound 1.50 → 1.64 vs e^{cL} = 1.65 — becomes a standing test |
| **A5** | **Loop Lipschitz certificate** — *new object*. Per step: `Λ_k = ‖rotor‖ · (1 + g_k‖ΔW_k‖) · ‖A_ρ‖ · Λ_core` with the first and third factors provably 1, `Λ_core` from the core block's own spectral estimate. Certificate: `‖J_step‖ ≤ Λ_k` | random modules; verified today at three gate/norm settings: 1.037 ≤ 1.050, 1.181 ≤ 1.250, 1.230 ≤ 1.300 | bound never violated; **then shipped as a runtime receipt line** logged every N steps in every training run, with a pre-registered alarm threshold (proposal: `∏_k Λ_k > e^{cL}` with `cL` from the panel's own fit ⇒ flag, not halt) |
| **A6** | **Gradient liveness at init** (the catch-#11 hunter): no parameter tensor has identically zero gradient on a random batch; every gated module's *inner* factors receive gradient when the gate is near-identity | one backward pass, every module ON, K ∈ {1, 4} | zero tensors = none; report the minimum gradient norm per module |
| **A7** | **OBS-INV matrix**: modules-OFF ≡ dense baseline **bit-identical**, extended per-module (each module OFF alone) and per-K (structural-OFF at K = 1, 2, 4, 8) | fp32 and bf16, CPU and GPU-deterministic | bit-identical (not allclose) at every cell |
| **A8** | **Fast-weight / delta-rule family** (successor arms, but the code path is shared): spectrum `[1−β, 1−ηλ]`, cap `‖S̄‖ ≤ ‖y‖_max/(2√λ)`, K-INV (closed gate ⇒ S unchanged) | random keys, adversarial energies | bounds hold; unnormalized-rate variant *fails* (positive control for the test itself) |

# 2. Track B — Instrument calibration on synthetic ground truth (INST-CAL)

**The chassis:** toy WEFT, d = 64, 4/2/4 blocks (ten executing, per the ruling), lanes 2×8, K curriculum 1→2→4, all modules present and gate-able — ~1M parameters, minutes per run on CPU or seconds on GPU. Synthetic task families where recurrence provably matters: k-hop composition (answer requires k sequential lookups), iterated modular arithmetic, parity-with-carry, graph reachability at fixed depth. Each instrument gets a **positive control** (effect planted) and a **negative control** (effect absent); pass = detects the first, nulls the second, with the instrument's own error bars covering the truth.

| ID | instrument | positive control | negative control | pass |
|---|---|---|---|---|
| **B1** | **Jacobian panel** (μR exponent `p`, `σ_slope`, cluster bootstrap) | plant a linear recurrent map with known spectral radius ⇒ known `p`; also a planted `p ∈ {1.0, 1.5}` decrement law | white-noise recurrence (no power law) ⇒ sign-inconsistency rejection must fire | recovered `p` within 2 SE of planted across 20 replicates; **coverage test**: 95 % CI covers truth ≥ 90 % of replicates; the `n₀ = 32` pilot semantics exercised end-to-end |
| **B2** | **Loop marginal gain `η_k` vs reallocation control** (Tier-1 #5) | k-hop task with k = 4: K = 4 must beat K = 1 at matched params | k = 1 task: K = 4 ≈ K = 1 | signs correct; trained-horizon annotation present |
| **B3** | **T14a/b/c causality** (Tier-1 #1) | loop on the output path | loop's write path severed (structural-OFF) | ordering-first (O-6) output distinguishes the two |
| **B4** | **RESP-LEAK** `ΔQ_M^disabled` (Tier-1 #6) | train with sidecar ON, disable at eval ⇒ leak > 0 | sidecar never enabled ⇒ leak ≈ 0 | separation > 3× the negative control's spread |
| **B5** | **G-INV + R-DUTY** (Tier-1 #7) | planted gain distribution across experts | random duty cycle | G-INV recovers the planted distribution; R-DUTY null is flat |
| **B6** | **Carrier retention `r`** (Tier-1 #3) and **T5 rotor Euclidean-vs-boost** (Tier-1 #8) | rotor with planted retention 0.95 | retention 0.5 | measured within 0.02 |
| **B7** | **EXTRAP-K** + **STOCH-K-uniform** | train K ≤ 4 staged ⇒ evaluate K = 8: expect GRT-type degradation on toy | train K ~ U{1..8} ⇒ expect flat/monotone `L(K)` | instrument's two branches both observed on toy, so the branch logic is exercised before it matters |
| **B8** | **ADDR-STAB** (from the Kathleen adjudication) | address = `Ph` with planted length/K drift ⇒ ρ_k and paired-OOD cosine collapse | content-keyed address ⇒ stable; normalized mixture ⇒ floor `√(1−λ²)` respected | all three behaviours reproduced; the floor never violated |
| **B9** | **WORKSPACE-TRAJ / jet-descriptor trajectories** | planted multi-phase dynamics | stationary dynamics | trajectory statistics separate the two |

# 3. Track C — The cheap catches

| ID | check | pass |
|---|---|---|
| **C1** | **μP coordinate check** across widths d ∈ {64, 128, 256, 512} on the toy chassis: activation RMS per module at init and after 10 steps | < 2× drift across widths per module; a module that scales wrong is a parameterization bug found before S2 spends compute on calibration |
| **C2** | **bf16 loop precision**: K = 8 recurrence in bf16 vs fp32 masters on the *full* toy step (not just the rotor) | per-visit divergence measured; decide fp32 accumulation for the carrier on evidence, pre-registered threshold 1e-2 relative |
| **C3** | **Determinism and replay** (O-9): same seed ⇒ bit-identical run, CPU and deterministic GPU; STOCH-K sampling replays; per-module generators isolated (disabling one module does not shift another's stream) | bit-identical; stream isolation proven by test |
| **C4** | **Engram micro**: hashed n-gram table collision rate vs table size on a Zipfian synthetic stream; planted n-gram → target associations retrieved; Kathleen's crosstalk-vs-address-dim curve reproduced on toy | collision rate matches the analytic birthday estimate; crosstalk falls with dimension as predicted; **no sealed data touched** |
| **C5** | **Callosum bypass baseline** (CAL-BW-2): on the toy graph, lane→lane linear predictability with callosum OFF vs ON at init | a number, recorded — the first datum on how leaky the bottleneck is |
| **C6** | **Structural-OFF ≠ removal** regression test: the 4/2/4 proxy executes ten blocks under every OFF combination (catch #20 hunter) | block count asserted |
| **C7** | **Receipt schema dry run**: every receipt line the semantics chain and adjudications have added (ρ values, consumption fields, integer F\*, checkpoint indices, gate-rate-vs-K, realized ηλ, Λ_k certificate) emitted by the toy run | schema complete; nothing invented at G-TOK time |

# 4. Schedule — two weeks, two lanes

**Week 1 (certificates + the catches that block everything else):** A1–A8, A5's runtime certificate wired, C1–C3, C6–C7, and B1 (the panel's synthetic ground truth — the single highest-value calibration, because the panel produces the headline exponent).
**Week 2 (instrument calibration):** B2–B9, C4–C5, then a **PRE-FLIGHT receipt**: one table, every item, pass/fail/measured-value, bytes and hash. Any failure on a *ratified* operator or instrument returns to strategy through the amendment path — it is a catch, and catches get numbered, not patched.

**Cost:** overwhelmingly CPU-minutes. GPU only for the deterministic-GPU checks (A7, C3), bf16 precision (C2), and the widest μP width (C1). **Proposed cap: 3 A100-hr**, drawn from the observatory's exploration allocation (20 % of 25 = 5 A100-hr) — pre-flight *is* exploration spent on instrument validity — leaving 2 for later. The G-TOK tripwire (12) is untouched.

# 5. What pre-flight is not

Not architecture work: no new arms, no spec changes, no hyperparameter search. Not a substitute for S2: μP *calibration* still happens at S2; C1 only catches parameterization bugs. Not a use of the sealed corpus or of any public text for training — the engram micro uses synthetic Zipfian streams precisely so that nothing here can be mistaken for a data-provenance event.

# 6. Two decisions for Mark

**D-PF-1 — the GPU cap.** *(a) 3 A100-hr from the exploration allocation (recommended)* — keeps pre-flight inside an existing budget line; the exploration budget exists for exactly "is the instrument valid" work. *(b) A separate 5 A100-hr pre-flight line* — cleaner accounting, but a new line needs ratification and the BD-1 doctrine prefers not to mint lines for two-week efforts. *(c) CPU only* — zero GPU, but A7/C2/C3's GPU-deterministic and bf16 checks then wait for the first real run, which is the wrong time to discover them.

**D-PF-2 — priority if the agent cannot run both lanes beside P-A.** *(a) Track A + B1 first (recommended)* — certificates and the panel's calibration protect the headline result and the training run's stability; the remaining instrument calibrations can trail into the first days of S2 without cost. *(b) Track B first* — maximizes instrument confidence early, but leaves the Lipschitz certificate and liveness hunter unwired when the first GPU step runs.

---

*Signature block*

**Strategy:** the single most important line above is A5 — the loop Lipschitz certificate turns the recurrence law's contraction assumption into a number the run reports every step, which is the difference between a theory we hope holds and one we would notice failing. Second is B1: the Jacobian panel is about to produce the program's headline exponent against GRT's external prior, and it has never been shown to recover a planted one. Everything else is the cheap-catch discipline that the ledger's twenty-one entries argue for: the mistakes this program makes are found by tests written *before* the run, or by the coding agent at three in the morning.
**Coding agent:** this is build-axis work under W-1 — nothing here consumes frozen V, sealed data, budgeted compute, or a checkpoint, and it must not slow P-A. On Mark's ratification of D-PF-1/2: wire A5 and A6 first (they run on every commit thereafter), then the schedule as ordered. Return failures on ratified operators through the amendment path with the planted-vs-measured numbers; return everything else in the PRE-FLIGHT receipt.
**Mark:** two decisions, §6, neither urgent by more than a day. The honest framing of the whole program: two weeks of pre-flight cannot make the first training run succeed, but it can make almost every way it would *fail* a way we would see coming.
