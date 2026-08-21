# STRATEGY HANDOFF — 2B-S Preludes: Executable Authorization for the Coding Agent

**Date:** 2026-08-21
**Author:** Strategy agent (session bf36cdbb)
**Status:** AUTHORIZED FOR EXECUTION — both preludes launch together in one session
**Basis:** `STRATEGY_2BA_RESULT_ADJUDICATION_20260821.md` (Drive `1eJvHq5lhHgSjos7zkt42YFTNbpEPS72D`, 15,828 B, SHA-256 `28806eff10fa9a40258a8563243ffd177a51c06df1dfc3d5d81c4ecef5a838a7`), **ratified by Mark 2026-08-21** ("Ratified", recorded in-session) — R1–R5 and the prelude gate are now binding.

---

## Plain-language summary

Mark has signed off on the autopsy verdicts and the plan. Before we design the next training stage, two cheap checks run first. Prelude-1 asks whether the untrained wrapper's surprising loop-4 recovery is real, reusable computation or a fragile accident that happens to cancel out — we poke it three ways (add noise, remove the intermediate writes, swap states between questions) and see if it bends or shatters. Prelude-2 is the deferred desk audit of which weight matrices actually moved during the failed run, testing the audit's prediction that the multiplicative gate starved one matrix of gradient. Both are read-only: no optimizer, no training, sealed evaluation sets untouched. Their results decide the shape of the successor charter.

---

## 0. Ratification record

Mark ratified adjudication rulings R1–R5 on 2026-08-21 after strategy's audit-vs-data reconciliation. Effective now: R1 session deviation accepted; R2 A-D1 = Prelude-2; R3 preludes gate the 2B-S charter; R4 M2 route stays closed; R5 row-18 rider results recorded (clustered correction field, weak incumbent alignment). The 2B-S outline (adjudication §6) is the fixed interpretive target for both preludes; it is not yet a charter and nothing in this handoff authorizes training.

## 1. Prelude-1 — K4-recovery mechanism probe

**Question (autopsy Q4):** is the init-time K4 recovery (K1=162, K2=10, K3=2, K4=160 generative /461) reusable computation or brittle phase cancellation?

**Scope:** score-only; **init checkpoints only**, both seeds; ≤2 GPU-hours total; the 461-row generative slice from the autopsy K-sweep, identical row set.

**Pre-flight gate (mandatory, before any probe cell):** re-run the unperturbed K-sweep on both seeds and reproduce the autopsy init numbers **bit-exact** (162/10/2/160 and the seed-1 counterpart). Failure = stop-the-line, report, no probe cells run. This is the runtime-pinning law's unchanged-endpoint reproduction gate; name accelerator class, torch/CUDA versions, dtype (bf16), attention backend (SDPA) in the receipt.

**Probe (a) — perturbation robustness.** Inject isotropic Gaussian noise into the loop state entering K2, scaled relative to the per-row RMS of that state: ε ∈ {0.001, 0.003, 0.01, 0.03, 0.1} × RMS, one fixed noise seed per (row, ε) cell, both model seeds. Measure K4 generative score at each ε.
**Registered keys:** `SHATTERS` = ≥50% of the K4 recovery margin over K3 (i.e., of the ~158-row gap) lost at ε ≤ 0.003. `SMOOTH` = monotone-or-flat decline retaining ≥50% of the recovery through ε = 0.03. Intermediate patterns report as `MIXED` with the full curve.

**Probe (b) — loop-order sensitivity.** Zero the innovation writes during K2 and K3 only (the validated `inherited_flow_off` machinery, applied per-loop; carry and bridge execution preserved; per-loop zero-contribution activation receipts required as in the autopsy), then measure K4.
**Registered keys:** `SURVIVES` = K4 ≥ 80% of unperturbed K4. `DEPENDENT` = K4 ≤ 50%. Between: `MIXED`.

**Probe (c) — cross-question transplant.** On matched-battery pairs (same battery, both init-K4-correct), carry row i's K3 state into row j's K4 pass; ≥64 pairs per seed, pairing fixed by row-index parity before any measurement.
**Registered keys:** `GRACEFUL` = transplanted K4 accuracy ≥ 50% of native on the pair set. `CATASTROPHIC` = ≤ 15%. Between: `MIXED`.

**Decision mapping (pre-registered, from the adjudication):** SMOOTH + SURVIVES + GRACEFUL ⇒ **reusable computation** — 2B-S anchors on K4 rehearsal as outlined. SHATTERS + DEPENDENT + CATASTROPHIC ⇒ **phase cancellation** — 2B-S pivots to protecting shallow behavior and building depth rather than preserving it. Any MIXED, or any disagreement between seeds on a key, escalates to a strategy ruling before charter; do not resolve locally.

**Registered prediction (strategy, blind, for the scoreboard):** partial-computation outcome — probe (a) SMOOTH at small ε, probe (b) DEPENDENT (the recovery consumes the intermediate writes it follows), probe (c) MIXED. I expect the recovery to be real computation *about* the intermediate disruption, not independent of it.

## 2. Prelude-2 — A-D1 weight-delta desk audit

**Scope:** CPU-only; held init and step-1,000 EMA endpoints, both seeds; no forward passes on the substrate required.

**Measurements, per seed:** (i) per-matrix relative movement ‖ΔW‖_F/‖W₀‖_F for W_H, W_P, the AnchoredBridge trainables (B_L and g_L; report B₀ separately if trainable in the as-built config), and each loop-LoRA A/B factor pair; (ii) top-3 singular vectors of ΔW for W_H and W_P with singular-value fractions; (iii) alignment cosines of those top singular directions against the arm-6 cluster centroids and against the fitted common-mode direction from the autopsy state diagnostics (both available in the autopsy receipts); (iv) the F1 discriminator below.

**Registered key (F1 serialization test):** `STARVED` = relative-movement ratio (‖ΔW_P‖_F/‖W_P₀‖_F) / (‖ΔW_H‖_F/‖W_H₀‖_F) ≤ 0.25 on both seeds. `NOT_STARVED` = ratio ≥ 0.75 on either seed. Between: `PARTIAL`.

**Registered prediction (strategy, from the architecture audit, unchanged and still blind):** `STARVED` — the multiplicative zero-init gate starves W_P of gradient early (foot-gun F1). If confirmed, the E2 scalar zero-gate repair is promoted from recommended to required in the 2B-S charter; if refuted, F1 is downgraded to theoretical and E2 becomes optional.

## 3. Constraints (both preludes)

No optimizer construction anywhere; Prelude-1 is forward passes only, Prelude-2 is linear algebra on checkpoints. CONFIRM and EVAL-E remain sealed. DEV row identities beyond the specified slices are not enumerated in the handoff back. Same-session condition applies with the R1 understanding: infrastructure resumptions are tolerable if disclosed, semantics pinned, and the pre-flight gate re-passed after any resumption that touches Prelude-1. Every artifact in the result handoff carries SHA-256; retention list = the two result tables + probe curves + the pre-flight receipt, verified present at handoff time (retention-verification-at-look).

## 4. Deliverable contract

One result handoff document on Drive (folder `1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr`), byte-sized and SHA'd in the relay message, containing: pre-flight reproduction receipt; the three probe (a)–(c) result tables with registered keys resolved per seed; the A-D1 table with the F1 key resolved; runtime pin block; deviations section (empty or itemized); open questions. Wave rule: both preludes report together in one wave; strategy adjudicates and, if the mapping is clean, proceeds directly to the 2B-S charter draft for Mark's pre-signature review.

---

*Signature block*

**Strategy:** authorized 2026-08-21 under Mark's ratification of adjudication R1–R5.
**Coding agent:** acknowledge by relaying the pre-flight gate result before running probe cells.
