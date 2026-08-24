# STRATEGY RULINGS — W1 Unblocked: Injection Position, Phase-B Contracts, and the Oracle-Assisted Generative Protocol

**Date:** 2026-08-24
**Status:** BINDING RULINGS W1-R1 through W1-R4. Phase A resumes on receipt; no changes to Phase A targets, constants, controls, winner rule, or the 8 A100-hr cap. D5 in force; Step-2 blocked; CONFIRM/EVAL-E sealed.
**Basis:** Coding-agent requests, both byte-verified on decode: `BICAMERAL_W1_INJECTION_POSITION_CLARIFICATION_REQUEST_20260824.md` (Drive `1hzGS4wPTRGMkdzuiaDdWBG8SxLvVCgKJ`, 2,422 B, SHA `55570cd7956e1b504fa36e7193e8cbf76f0e549e330a05ef7b8f307681bf0866`) and `BICAMERAL_W1_PHASE_B_GENERATIVE_CONTRACT_REQUEST_20260824.md` (Drive `1TkiTDV-oXPeAwjYLTyfU1iukEhHrY5zB`, 3,647 B, SHA `e32e0b1e7deebe27d0c071f93c7fcc91eb1133c5859a5e60ff3d6713c01a7859`). Seed-0 preservation acknowledged (archive 35,002,614 B, SHA `e465cc3252defde42a2a571b8d5e87352cbb6de4faa75231b5f91834cd63f7fc`; summary SHA `901829de…48b3`; 0 optimizer steps; seals untouched; commits through `47c725ce`).

---

## Plain-language summary

The agent stopped for the right reason again: my authorization contained two clauses that only agree if you already know what I meant. "Apply the final-cell write convention exactly" and "at the terminal position" pick out different tensor operations — the frozen final-cell write touches every active token, while a literal reading of my phrase touches only the last one. The ruling is the first reading: all active positions, exactly the operation whose harmlessness we measured, which also means the completed seed-0 run is valid and banks as the registered result, and seed 1 resumes under unchanged code. The wording was mine; the agent's refusal to guess is the sixth strategy-sourced ambiguity this discipline has caught before it could contaminate a result. The other three rulings complete the back half of the wave before it can stall again: cluster labels for the full margin panel come from the frozen Stage-0 centroids with no refitting and a sanity gate on the extension; the mystery residual directions are taken per seed, both signs, with a registered orientation convention so "both seeds agree" is even meaningful; and the generative follow-up runs with the oracle-derived target held fixed during decoding — labeled loudly as an oracle-assisted causal read, never a capability claim, with a phrase rule to keep it honest in every future document, and with the useful distinction that the cluster-mean and residual-direction arms carry no per-row answer leak at all, so their generative numbers are deployable-grade where the per-row arms' are not.

## W1-R1 — Injection position: ALL ACTIVE POSITIONS (option 1, ratified); seed 0 BANKS; language corrected

The controlling clause of the authorization is "the final-cell deployed-write convention, **exactly**" — comparability with the operation whose safety and delivery we measured (159-flat, ≥99.957% delivery) was the design intent, and that receipt attaches to the **full active-token bridge mask**. The phrase "at the terminal position" meant *terminal in the write schedule* (the single deferred write at the end of computation, as against per-loop writes), not the last token; as written it was ambiguous, **the ambiguity is mine, and it is owned on the scoreboard as the sixth strategy-sourced catch**. Corrected authorization language, binding: *"applied under the final-cell bridge write mask (all active token positions), at the terminal write of the deferred schedule."* A literal last-token intervention would have been a new estimator with no prior receipts — rejected.

**Consequences:** seed 0's completed Phase-A run (11 cells × 2,048 rows, all-active-position implementation) is **promoted from engineering evidence to the registered W1 seed-0 result**, subject to the standard byte-verification of its receipts in the wave's result handoff. Seed 1 resumes under the existing code, unchanged.

## W1-R2 — Phase-B cluster assignments: frozen-centroid extension to the full panel (recommended option, ratified, with one added gate)

Freeze each seed's Stage-0 feature transform and k=2 centroids; compute identical features for all 2,048 DEV-2 rows; assign every row to its nearest frozen centroid; **no refitting on DEV-2**; L1/L2/L3 computed from the winning family over the full panel. The 256-row alternative is rejected for comparability (the L0 result lives on the 2,048-row population; no hybrid comparisons). Required artifacts stand as requested: feature-transform, centroid, and assignment files by name, bytes, and SHA-256, both seeds, emitted **before** any Phase-B scoring, together with the full-population cluster counts and per-battery composition. **Added gate (registered now, before data):** if the frozen-centroid extension assigns fewer than 5% of the 2,048 rows to either cluster in either seed, stop and report before scoring — a degenerate extension is a finding about the clusters, not a nuisance to route around.

## W1-R3 — L6 residual directions: per-seed eigenvectors, both signs, with a registered orientation convention

Adopted as recommended: each seed's own cross-fitted R-S0-A residual eigenvectors, descending eigenvalue order; both signs scored as separate cells; sign retained in every arm name and receipt; no post-hoc sign selection; tensors and hashes provided before staging. **One completion the request correctly forces:** eigenvector sign is arbitrary *per seed*, so "both seeds agree on +u₁" is undefined without an orientation rule. **Registered orientation:** orient each seed's u_k so that ⟨u_k, ĉ_seed⟩ ≥ 0, where ĉ_seed is that seed's global correction-mean direction (a deterministic, relational convention — legal under the gauge law). With that orientation fixed, the both-seed rule applies at the rung level: rung (k, sign) clears only if it clears in both seeds, each seed using its own oriented direction. Directions are seed-specific gauge objects; cross-seed identity of the vectors is not required and not claimed.

## W1-R4 — Generative staging: the oracle-assisted protocol (recommended option, ratified, with tag semantics and a phrase rule)

Adopted: for each registered 461-slice row, compute the arm's target once (same construction as the margin panel), hold it fixed, and apply it through the frozen final-cell write convention **at every autoregressive decoding step**, with the same γ=0.05, serving reader, generation parameters, and execution schedule as the frozen depth-study config-2 evaluator. Omission is rejected: the ladder's question is causal — *does this direction fix answers when delivered?* — and the same answer-derivation already lives in the margin panel; crippling the generative readout while keeping the margin one would be asymmetric self-deception. The controls (own-shuffle and random on the same rows whenever a parent arm stages) net out generic-injection effects, which is what makes the causal contrast valid.

**Tag semantics (binding on all receipts and documents):**
- **`oracle-target-assisted`** — every L0-family and L5 arm (the row's own gold answer enters the target). These numbers are causal capability reads. **Standing phrase rule (R5-class): no `oracle-target-assisted` generative number may ever be quoted as model capability or accuracy; only contrasts against same-row controls are quotable.**
- **`population-target`** — L1/L2/L3/L6 arms (cluster means, global mean, residual directions: no per-row answer enters the injected vector). These generative reads are deployable-grade evidence and may be discussed as such, with their own controls.

Required inputs stand: the exact 461-row manifest and frozen generation configuration cited by artifact hash in the resumption receipt (the depth-study config-2 artifacts the agent already holds).

## Resumption

Phase A resumes on receipt of this document: provision A100, execute seed 1 under the unchanged all-active-position code, then Phase B under W1-R2/R3 and staging under W1-R4. All prior caps, keys, branch maps, trim order, and escalation rules unchanged. One result handoff per the wave rule. The tracker update for these rulings rides the W1 result adjudication.

---

*Signature block*

**Strategy:** W1-R1–R4 issued 2026-08-24; injection ambiguity owned (sixth strategy-sourced catch); seed 0 promoted to registered result conditional on handoff verification; orientation convention and tag semantics registered before any Phase-B or generative data exists.
**Coding agent:** resume per §Resumption; emit the named artifacts (assignments, residual tensors, generation config) with hashes before their phases score.
**Mark:** no decision required — these rulings interpret and complete the already-ratified W1 authorization without changing its scope, cost, or keys.
