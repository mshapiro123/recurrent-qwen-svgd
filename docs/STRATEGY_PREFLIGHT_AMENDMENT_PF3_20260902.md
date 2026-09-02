# STRATEGY — PRE-FLIGHT Amendment PF-3: μP Bound (#33), Zero-Reference Rule (#34), #26/#27 Bound, the Build-Status Matrix Accepted — and the Critical Path Has Moved

**Date:** 2026-09-02 · **Status:** AMENDMENT to the PRE-FLIGHT chain, adjudicating the PF-2 implementation return (9,313 B, SHA-256 `c431f4bc…537677`, commit `5ebbefea`) and the **build-status matrix** (16,774 B, SHA-256 `79189268…443c06`), both **read in full from Drive before ruling** (PS-1). Where PF-3 conflicts with earlier documents, PF-3 governs. §1 fills gaps in handoff §8 under standing "exact semantics" authority; it contradicts no ratified line.
**Ledger:** #33 and #34 accepted, both strategy's (§8 under-specified; PF-2.2 silent on zero references). #26 and #27 now have their texts and are **bound below**. **Tenth consecutive correct fail-closed stop.** C7 stage 1 accepted; n = 520 recorded.
**One decision for Mark (D-PF-4, §5):** the matrix changes what the two weeks should be spent on.

---

## 0. Plain-language summary

Three things in this return. First, the coding agent could not run the width-scaling check because my build spec never wrote down the numbers μP needs — the base width, the base initialization scale, the base learning rate, and which rule each tensor follows. That is a real gap, and it is filled here in the base-shape form the §8.1 amendment established: the structural rules are binding, and the numeric base constants are bound as provisional pre-flight values that S2's calibration is explicitly entitled to replace, because replacing them at the base width is what μTransfer is for. Second, two smaller precision items — a rule for tensors that are exactly zero because the module hasn't executed yet, and the definitions that two older catches had been waiting on: what "the state" is when we measure how much a recurrent visit can expand it, and what exactly the structural-OFF invariance test compares against. Both are bound.

Third, and most important: the build-status matrix. It is an exemplary document and an uncomfortable one. The model that exists today is a well-tested recurrent bring-up graph — dense substrate, shared recurrence, static K/V, narrow lanes, token engram, read-only memory, diagnostics — and not the WEFT-1 that was ratified. The full-width bicameral core, the learned rotor carrier and its single write, the per-band callosum and final combiner, the loop sidecar, the occupancy router, the post-loop bridge, the halting head, and the objective stack are absent or standalone. The agent's own dependency ordering is correct and is ratified here. Two consequences follow. The instrument-calibration half of pre-flight cannot run on a toy WEFT that does not yet contain those modules, so continuing it as scheduled would calibrate instruments against a model that is not the one we will train. And the corpus is roughly two weeks out — which means the build queue, not G-TOK, is now the critical path after the freeze. The honest move is to pivot the remaining pre-flight time to integration, with each instrument calibrated as its module lands. That is D-PF-4, and it is Mark's call because it re-prioritizes what he ratified two days ago. Four of the integration steps also wait on design questions strategy owes — the recombination rule, the K/V representation, and two sidecar semantics — which get their own ruling document next, after I have read the handoff sections they live in rather than ruling from memory.

---

# 1. Catch #33 — the eight μP literals (accepted; strategy's; bound in base-shape form)

**Binding rule of construction:** every μP quantity is stated relative to the **base shape** `B₀ = (d_base = 512, d_head = 64, Q = 8, KV = 4, d_ff = 1408, lanes = 2×128, V = 32,768 provisional)` — the proxy rung, where S2 calibration runs — with width multiplier `m = d/d_base`. At `m = 1` the model **is** the standard-parameterization model. **Structural rules (1–5) are binding. Numeric base constants (6–8) are PROVISIONAL-PF: bound for pre-flight so C1 can run, and owned by S2 calibration, which may replace them at the base shape — that is μTransfer's contract, not a loophole.**

> **PF-3.1 — per-tensor class map and rules (Adam/AdamW μP, base-shape form).**
> **(1) Hidden class** — every weight matrix whose fan-in *and* fan-out scale with width: attention Q/K/V/O projections, SwiGLU up/gate/down, lane and scratch projections, re-entry/bridge projections, expert bank matrices, callosum/rotor parameter matrices when integrated. Init `W ~ N(0, σ_base²/fan_in)` (the same rule at every width); Adam LR `η_base/m`.
> **(2) Input class** — tensors whose fan-in is width-independent: the token embedding (tied), engram tables, any learned positional/anchor embeddings. Init `N(0, σ_emb²)` (width-constant); LR `η_base`.
> **(3) Output/readout** — **tied to the embedding**, so no separate output tensor exists; the readout applies a **multiplier `α_out/m`** to the logits (`logits = (h · Eᵀ) · α_out/m`), which realizes the μP output rule (init var ∝ 1/fan_in², LR ∝ 1/fan_in) for a shared tensor. Verified: at init the logit scale falls as `1/√m` with random `h` — the intended μP behaviour (readout is small at init, O(1) once `h` correlates with `E`).
> **(4) Vector class** — RMSNorm gains, biases, `layer_scale`, gate scalars/vectors, QK-norm gains: init as each module's design states (the gate-only near-identity rule governs gates); LR `η_base`; **no weight decay**.
> **(5) Weight decay** — decoupled, applied with the **base** rate on hidden-class tensors only: per-step decay factor `(1 − η_base·wd)`, i.e. the decay does *not* shrink with the per-tensor LR (torch's `lr × wd` coupling is corrected by setting `wd_tensor = wd·m` for hidden tensors, or by an explicit decoupled step — implementation's choice, receipt states which). Embeddings and vectors are not decayed.
> **(6) PROVISIONAL-PF numerics:** `σ_base = 1.0` (hidden init variance exactly `1/fan_in`); `σ_emb = 0.02`; `η_base = 3e-4` (the G-TOK AdamW rate, for continuity); `α_out = 1.0`; `α_emb = 1.0` (embedding-output multiplier; the input multiplier S2 may sweep); **residual multiplier = 1.0** (pre-norm residual, no depth scaling — depth is a rung parameter, not a μP axis in this program); **residual-branch `α`: none beyond the recurrence's own `α_T = c/T`**, which is a loop quantity, not a μP one, and is untouched. `wd = 0.1`, betas `(0.9, 0.95)`, `eps = 1e-8` as ratified.
> **(7) Attention scale:** `√(d_head,base)/d_head` per the §8.1 amendment — already integrated.
> **(8) The catch clause stands:** any tensor not assignable to classes (1)–(4) by the rules above returns as a catch with its name and shape; the implementer never assigns a class locally.
>
> C1 then runs under PF-2.1's topology with `m ∈ {0.25, 0.5, 1}`; pass unchanged (< 2× RMS drift per module across widths at init and after ten steps). The earlier 9× FFN drift is expected to vanish under rule (1); if it does not, the module's parameterization is a catch.

# 2. Catch #34 — zero-reference tensors in C2 (accepted; strategy's; bound)

> **PF-3.2.** Relative error is defined only where `‖x_fp32‖₂ > 0`. A tensor whose fp32 reference is exactly zero at a visit because it is **structurally ineligible** at that visit (PF-1.5's eligibility matrix — re-entry on visit zero is the known case) is excluded from that visit's relative-error population and reported as `ineligible (zero reference)` with the structural reason; **the check for such a tensor is that its bf16 value is also exactly zero** — an ineligible tensor that is nonzero in one precision and zero in the other is a fail. A zero fp32 reference for an *eligible* tensor is itself a fail (a live tensor with an identically zero gradient is the frozen case). Non-terminal exceedances (the visit-4 `engram.gate_bias` at 0.062) are retained as diagnostic lines with tensor name and both norms — the terminal-visit decision rule stands, exactly as the agent applied it. With this rule the pinned run's C2 receipt is complete: **pass.**

# 3. Catches #26 and #27 — now bound

> **PF-3.3 — #26 (C-JAC-1), the joint-state metric.** The recurrent visit's state is the concatenation `z = [h ; lanes ; scratch ; carrier-when-integrated]`, and the metric is the **plain Euclidean norm on the concatenation, no per-block reweighting** — so that block-operator norms compose as PF-1.4 writes them. `Λ̂_core` is power iteration (JVP/VJP) on the Jacobian of the *full joint-state* visit map, with the PF-1.4 convergence receipt. **C-JAC-1's prohibition on a production certificate/alarm stands** until the certificate topology is complete (it cannot be complete while the modules in §5 are absent); measurement and the two-number receipt line (`Λ_adapters`, `Λ̂_core`) are authorized now on the graph that exists.

> **PF-3.4 — #27 (A7), the comparison graph and eligibility.** Two comparisons, both **bit-identical** (`torch.equal` on logits and loss; fp32 and bf16; CPU now, deterministic CUDA under the PRE-FLIGHT meter): **(a) the anchor** — all optional modules structurally OFF at K = 1 equals the standalone dense transformer of the same configuration executing the same 4/2/4 block schedule (this is OBS-INV as ratified; the existing `assert_close(rtol=0, atol=0)` test is **promoted to `torch.equal`** and becomes the registered anchor cell); **(b) per-module OFF-idempotence at K ∈ {1, 2, 4, 8}** — for each module *eligible at that K* (per PF-1.5's matrix: re-entry needs K ≥ 2, second-order-jet consumers K ≥ 3), the graph with that module ON-then-structurally-OFF equals the all-OFF graph at the same K bit-identically, and the module ON differs from all-OFF (non-triviality — an OFF switch that changes nothing when toggled is a dead module). Cells for absent modules are typed non-passes until integration. The matrix `(module × K × dtype × backend)` so defined is the registered A7 matrix; it is minted only when every cell for every *integrated* module is green.

# 4. Accepted without further ruling

C7 stage 1 — emitted through the production builders, four families, base `n = 400` / confirmation `n = 399`, receipt `04b9c151…22abd6`, forgery-resistant by mutation test — **accepted**; stage 2 waits on the sidecar and PF-3.3 as recorded. Jacobian panel at **n = 520** with both frontiers passing (0.05092 ≤ 0.051; 0.03579 ≤ 0.036) — **recorded as the registered panel size**. The §8.1 implementation return (named constant, assertions at the ratified shape, no numerical change) — **accepted; #28 and #32 closed.**

# 5. The build-status matrix — accepted, and what it changes

**Accepted in full**, including the vocabulary, the PF-LANG discipline, and the do-not-claim boundary. **The implied queue is ratified as the build order:**

1. bind **C-S5-1** (final recombination `combine(h_A, h_B)`) and **C-S5-2** (production K/V representation: paired-eigenmode vs shared-consensus) — *strategy owes these*;
2. integrate the full-width bicameral block into the recurrent path;
3. integrate the learned rotor carrier, the single gated rank-8 write, the post-loop `bridge_out`, and the fitted retention gauge;
4. integrate the per-band callosum and final combiner;
5. bind **C-S6-1** (sidecar width/accounting) and **C-S6-2** (hard-invocation estimator/init) — *strategy owes these* — then integrate the conditional loop sidecar;
6. #26/#27 topology completion (now bound above; completion follows integration);
7. first production T14b and OBS-INV receipts.

**The four open design questions (C-S5-1/2, C-S6-1/2) get a dedicated ruling document next.** I will read the handoff's S5/S6 sections from Drive before writing it (PS-1 — I have been burned twice this week ruling from memory); the agent is asked to include the **exact open-question texts as it holds them** in its next receipt so the ruling answers the questions actually asked.

**The critical-path observation, stated plainly for Mark:** with P-A about two weeks from completion, the freeze and G-TOK follow it — but the model that G-TOK's winner will be trained into does not exist yet as an integrated graph. Steps 1–5 above are the critical path after the freeze. And PRE-FLIGHT Track B (instrument calibration on a toy WEFT "with all modules present and gate-able") **cannot run as written** until they land — it would otherwise calibrate instruments against a model that is not the one we train.

> **D-PF-4 — decision for Mark.** *(a) Pivot the remaining pre-flight window to integration (recommended):* the agent works the ratified queue steps 2–4 now on the build axis (ungated under W-1), strategy delivers S5/S6 rulings to unblock 1 and 5, and each Track-B calibration (B2–B9) runs **as its module lands** rather than on the original week-2 schedule. Pre-flight's Track A/C items stay wired and run on every commit. *(b) Keep the ratified schedule:* finish Track B on the bring-up graph as-is — measures the instruments against a model missing its defining modules, and delays integration by a week on a path that is already the long pole.

# 6. What resumes

On verification: C1 under PF-3.1; C2 completes under PF-3.2; A7 anchor promotion and the per-module matrix under PF-3.4; `Λ` two-number line under PF-3.3. On D-PF-4: the week's ordering. P-A untouched; no GPU cell promoted; the PRE-FLIGHT meter remains its own domain.

---

*Signature block*

**Strategy:** the matrix is the most useful receipt this program has produced, and the least comfortable: it says the design is ratified and the model is not built, in a table no one can argue with. The right response is not to defend the schedule but to re-cut it, which is what D-PF-4 asks. On #33 — a build spec that says "μP" without writing the base shape, the per-tensor map, and the numbers is a promise, not a parameterization; it is bound now, with the provisional constants labelled so S2 replaces them without ceremony. #26 and #27 were waiting on definitions nobody had written; they are written.
**Coding agent:** the matrix's vocabulary and do-not-claim boundary are adopted as standard. Verify bytes and hash; bind PF-3.1–3.4; on Mark's D-PF-4 ruling, take queue steps 2–4 or the original schedule; include the exact C-S5-1/2 and C-S6-1/2 question texts in the next receipt.
**Mark:** one decision, D-PF-4 — whether the second pre-flight week becomes an integration week. My recommendation is yes: the instruments can wait for their modules; the modules cannot wait for the instruments.
