# STRATEGY CHARTER + EXECUTABLE HANDOFF — W2′: The Conditional Mixer Program (Desk-Gated)

**Date:** 2026-08-25
**Status:** RATIFIED BY MARK 2026-08-25 and EXECUTABLE to the coding agent as written. Phase D (desk, CPU) starts on receipt. D4 (caching pass) and Phase G (scoring wave) are GPU-touching and are authorized **conditionally** per §6 — Phase G runs only if both desk gates pass, dry-run-costed, within cap; any gate failure, seed split, or surprise returns to strategy. **No optimizer is constructed anywhere in this wave. Step-2 remains blocked (D5). CONFIRM and EVAL-E remain sealed. No amplitude exploration: γ = 0.05 fixed.**
**Authority chain:** 2B-S charter `1xeAJYfq2lIvIm76nszshH4xmaeAkiDl2` (SHA `51083fa2…72ac`) → W1 result handoff `1pCW3126FB3GOyBXJhTcgtwHUjCPA1N6P` (14,072 B, SHA `630ad9ed7105a568b926a9a3d72ddac55b8bb8609d31f840f833ac19bea43d4a`, byte-verified) → W1 adjudication `1qqhKSzdaYEgc-uS0F_kWqTUOqEet8kJr` (15,383 B, SHA `25c0dd3c9fae501cb63f81e18933ca004ce44500573fe317deaacdee162b2815`; D-1/D-2 RATIFIED by Mark 2026-08-25, Step-1 fixed-direction pause in force) → Mark's six-option design note (relayed in-session 2026-08-25) → this charter, ratified same day.

---

## Plain-language summary

W1 proved the write interface can change complete answers and that every *fixed* direction we could deploy is worthless. This wave asks the only question left: can a map, reading nothing but what the deployed model can see about its own state, produce a *row-specific* correction that helps? The design spends almost nothing until the two cheapest possible tests are passed on cached tensors: does the map predict anything beyond the population mean (the mirage test), and do the two hemispheres carry any information about the correction that one ordinary pass doesn't (the necessity test)? Only if both pass does one forward-only GPU wave run — with the controls W1 taught us to want: a no-injection floor, a random control matched to the anisotropy of the space, shuffles that respect task families, and a single-stream twin of the mixer so a win can't be mistaken for "adapters help." Every decision rule is written down here, before any data exists.

## 1. Estimand and the leak boundary

**Estimand:** a conditional map u = g(features) from deployment-available quantities at the write interface to a per-row correction vector, evaluated at the answer level. Sizing basis: D-M5 (n·d observations). Motivation: W3 desk read — cross-fitted state-plus-trajectory map cosine ≈ 0.885 against the oracle field (undecomposed; decomposition is D1's job).

**Leak boundary (binding, the A-4 lesson made structural):** oracle quantities are legal **only as regression targets at fit time**. The map's *inputs* — at fit and at score — must be computable by the deployed student with no gold answer, no teacher forward, and no oracle-derived routing or selection anywhere in the input path. Every receipt states the input feature list explicitly.

## 2. Objects and definitions

- **h** — single-stream base state at the write interface. By T1 (bit-exact base path at all-gates-zero) this is the base model's own state; cached or cheaply cacheable.
- **h_A, h_B** — hemisphere states at the W0 operating point (gates at design values; branch correlation 0.7446, RMS difference 3.003, both seeds). **m = (h_A+h_B)/2, d = (h_A−h_B)/2** (swap eigenbasis).
- **Targets c\*(x)** — two registered supervision families, both `oracle-target-assisted` at fit time only: **c\*_a** = L0a loss-gradient target; **c\*_c** = L0c teacher-forced correction delta. Both are fit in Phase D; the **primary target is c\*_a** (generative pedigree from W1), registered now to foreclose post-hoc choice. c\*_c results are reported alongside; if only c\*_c passes the gates in both seeds, that is a named branch (`TARGET-FAMILY-SPLIT`) and returns to strategy.
- **The mixer (Mark's Option 1):** u(x) = s·(U_m V_mᵀ m + U_d V_dᵀ d), U,V ∈ R^{d×r}, fit as cross-fitted **reduced-rank ridge regression** (closed form — SVD of the ridge cross-covariance; no optimizer, D5-clean), with **separate rank budgets r_m, r_d and separate ridge strengths λ_m, λ_d** selected on held-out folds from a pre-registered grid (r ∈ {2,4,8,16,32}; λ log-grid, 7 points, agent to state it in the receipt before fitting). Init/gating law for any *future trained* variant, recorded now: random U,V with a scalar zero-gate s (E2); never zero factor matrices (F1). Irrelevant to this wave's closed-form fits.
- **Feature sets, registered:** **FS-1 (primary)** = (m, d). **FS-2 (secondary)** = (m, d, trajectory statistics exactly as in the W3 desk fit). **FS-0 (control)** = h alone, same total rank budget r_m+r_d — the parameter-matched single-stream twin. A gate may be passed by FS-1 or FS-2; whichever passes is carried into Phase G *unchanged* and named in every receipt.
- **Injection convention:** unchanged from W1 — v_inj = γ·(RMS(h_row)/RMS(u))·u, γ = 0.05, final-cell bridge write mask (all active positions), terminal write of the deferred schedule, `EV-LADDER-1` evaluator identity with the sequential execution schedule. Any mechanical deviation: stop and surface.

## 3. Phase D — desk battery (CPU except D4; starts on receipt)

**D1 — Mirage test (conditional cosine).** Decompose the W3 result: for each map (each target × feature set), compute the cross-fitted held-out cosine between (prediction − c̄_seed) and (c\*(x) − c̄_seed), where c̄_seed is that seed's population mean correction. Report pooled and per-battery, both seeds; secondary read with rank-8 R-S0-A nuisance deflation in place of mean removal. **GATE G-D1: pooled held-out conditional cosine ≥ 0.30 in both seeds (primary target or registered secondary per §2).**

**D2 — Hemispheric-necessity test.** Cross-fitted held-out risks R(c\*|h), R(c\*|m), R(c\*|m,d), R(c\*|h_A,h_B), all at matched total rank budget, both targets, with bootstrap CIs. **GATE G-D2: relative risk reduction of (m,d) over h ≥ 5% in both seeds.** Also report m-vs-h (does the masked mean differ from the base state at all) and the d-block share of fit energy (mode attribution).

**D3 — Desk analogs of the novel controls.** (i) Branch-pair shuffle: R(c\* | h_A(x), h_B(π(x))) — the row-level pairing test. (ii) Permuted-input risk: R(c\*(x) | features(π(x))). (iii) The A-3 composition desk item (per-battery decomposition of the L1−L2 = −0.054 margin deficit from the banked `w1_phase_b_analysis.json`) rides this phase.

**D4 — Site screen for Option 2 (GPU, forward-only, cap 0.25 A100-hr, both seeds).** One caching pass storing per-layer states at registered sites l ∈ {8, 12, 16} plus the interface, for the 256-row Stage-0 manifest. Then CPU: at each site, the incremental predictive value of h_A^l for the component of c\* orthogonal to what h_B^l predicts (and symmetrically). This screens Mark's Option 2 *before* any exchange module exists. **No injection at these sites this wave.**

**Desk-failure keys (each returns to strategy; Phase G does not run):** `MAP-MIRAGE` (G-D1 fails — the 0.885 was common mode) · `HEMISPHERES-UNINFORMATIVE` (G-D2 fails — Option 4's answer is no; the single-stream pivot decision goes to Mark with the receipts) · `TARGET-FAMILY-SPLIT` (§2). Seed split on any gate: escalate, no branch.

## 4. Phase G — one forward-only generative scoring wave (conditional authorization)

Runs only if G-D1 **and** G-D2 pass. Arms, per seed, all under the frozen injection convention, all scored generatively (SL-1):

| # | Arm | Question |
|---|---|---|
| G0 | **No-injection baseline** | the floor (SL-2, standing law) |
| G1 | Isotropic random, norm-matched | W1 continuity; the L4 anomaly re-read against G0 |
| G2 | **Covariance-matched random** — sampled from Σ̂ of the c\* population, norm-matched | fair generic-perturbation control in an anisotropic space |
| G3 | Within-battery shuffled c\* | row pairing, task identity preserved |
| G4 | **Branch-pair shuffle** — u = g(h_A(x), h_B(π(x))) | does row-level pairing of the two views matter |
| G5 | Permuted-input map — u = g(features(π(x))) | common-mode-only content of the map |
| G6 | **Mixer-from-h (FS-0)** — matched rank budget | parameter-matched single-stream control (Option 4) |
| G7 | **Mixer (winning FS from Phase D)** | the candidate |
| G8 | L0a oracle | ceiling; interface-liveness control (A-6 phrase rule) |

**Population:** preferred — a new **1,024-row battery-stratified generative manifest** drawn from DEV-2, frozen by name/bytes/SHA-256 *before any scoring* (the W1 461-row panel was underpowered for the L0a-over-baseline contrast). Pre-registered fallback: the existing 461-row manifest if the dry run prices the 1,024-row matrix over cap; never both, no hybrid comparisons. **Analysis:** paired per-row tables; McNemar exact tests and bootstrap CIs for every arm against G0, G5, and G6; per-battery breakdown; injection norm statistics; all margin numbers reported but decision-inert (SL-1). Tags per the three-valued taxonomy (A-4); G7's inputs make it `population-target` — the first deployable-grade candidate arm of the program.

**Compute-matched control (registered, not in this wave):** single stream with the middle run twice (weight-tied, sequential) plus a matched adapter — Mark's Option 4 compute-matched variant, which is also the original recurrent architecture. Deferred to its own cell because it requires a new graph identity; it enters the queue only if K-2 fires (§5).

## 5. Decision rules (registered branch map — decisions ride these keys only)

- **K-1 `HEMISPHERIC-CONDITIONAL-LIVE`:** G7 > G6, G7 > G5, G7 > G0, all both seeds → the hemispheric conditional channel is real. Next wave: Option 2 single-exchange site sweep (sites from D4; sender stop-grad law binding on any trained variant); the **Step-2 lock discussion opens** with these receipts.
- **K-2 `CONDITIONAL-LIVE-HEMISPHERES-DECORATIVE`:** G7 ≈ G6 > (G5, G0), both seeds → conditional correction works; the second hemisphere adds nothing to it. Pivot decision to Mark: single-stream + adapter carries the correction line; bicameralism retains only the ensemble-channel justification; the compute-matched control cell runs to complete the record.
- **K-3 `STEERING-DEAD-AT-GAMMA`:** no arm clears G0 and the controls, both seeds → conditional steering at γ = 0.05 is dead at every conditioning level tested. Return to strategy with two named options for Mark: the reserved γ probe (new safety receipts required) or the ensemble-only program.
- **Flat band:** W1-continuity ±9-row band applies to generative row counts; effect floor ≥ 20 rows for any "beats" claim. **Seed split on any key: escalate, no branch. Option 3 (repeated exchange) is unreachable except through a replicated K-1 and a subsequent single-exchange win.**

## 6. Budget, caps, and stop conditions

Phase D: CPU unrestricted; D4 capped **0.25 A100-hr**. Phase G: **dry run first** — measured per-arm generative cell time on the chosen manifest, reported before any full arm; cap **6.0 A100-hr**; wave total **6.5 A100-hr**. Pre-registered trim order if over cap: 1,024→461 fallback → G4 dropped → G2 dropped (G1 retained for W1 continuity). Stop-the-line: any cap breach, runtime mismatch, gate-seed split, degenerate Σ̂ sampling, injection-formula deviation, or unregistered analysis choice → return to strategy. Compute torn down after each session; summary-vs-file hash assertion on every archive.

## 7. Registered predictions (scored at the look; decisions ride §5, not these)

- **P-W2-1:** D1 conditional cosine lands in **0.35–0.55** both seeds — passes the gate, well below the raw 0.885.
- **P-W2-2:** G-D2 **passes**, but modestly: (m,d) over h relative risk reduction in **5–15%**.
- **P-W2-3:** In generation, **G7 ≈ G6** (K-2 over K-1) — my credence that the hemispheres add *generative* value this wave is low.
- **P-W2-4:** **G2 underperforms G1** — the covariance-matched control is a harsher perturbation than isotropic noise (the anisotropy hypothesis for the L4 anomaly).
- **P-W2-5:** No arm, G8 included, clears G0 by more than **5 pp** — the A-1 rider's ceiling is real at γ = 0.05.

## 8. Receipts required

Phase D: per-fit cross-fitted held-out risks and cosines with CIs; selected (r_m, r_d, λ_m, λ_d) and the full pre-stated grid; per-battery conditional cosines; mode-attribution shares; D3 outputs; D4 per-site tables; gate verdicts as machine keys. Phase G: frozen manifest (name, bytes, SHA-256) before scoring; per-arm × seed generative tables with all §4 contrasts; per-row tables into the fragility corpus; injection norm stats; dry-run costing. One result handoff per the wave rule: Drive + exact bytes + SHA-256, execution schedule and evaluator identity declared, optimizer-constructed/steps/CONFIRM/EVAL-E attestation lines.

---

*Signature block*

**Strategy:** charter drafted from Mark's six-option design note merged with the W2′ scoping and the banked W1 laws (SL-1, SL-2, three-valued tags, sequential identity, stop-grad, E2/F1 init discipline); predictions registered blind; desk-failure keys ensure the two cheapest refutations are bought first.
**Mark:** RATIFIED 2026-08-25 ("OK, ratified"). No further decision until the desk gates report or a key fires; K-2/K-3 outcomes return to you by name.
**Coding agent:** execute §3 on receipt (D1–D3 CPU immediately; D4 under its cap); report desk gates as a mid-wave receipt (gate verdicts + fit tables) before staging Phase G; Phase G only on double-pass, dry-run first; one result handoff; all seals and D5 unchanged.
