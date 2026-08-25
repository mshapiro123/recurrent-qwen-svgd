# STRATEGY RULINGS — TM-0 Preflight: The Hermetic Screen, the Calibration Subset, and the Prompt-Only Convention

**Date:** 2026-08-25
**Status:** BINDING RULINGS R-TM0-P1 through R-TM0-P3. These resolve the preflight stop within the ratified TM-0 r2/r3 scope; the caching session is unblocked on receipt, subject to the §1 screen completing first and the dry-run price staying under the 1.5 A100-hr cap. **No decision required from Mark** — R-TM0-P1 is a narrow seal-semantics interpretation that serves the seal's own purpose, recorded here with its full reasoning so the program owner can overrule it before the screen runs if he disagrees. D5 in force; Step-2 blocked; CONFIRM/EVAL-E sealed (and remaining so under this ruling).
**Basis:** `CODING_TO_STRATEGY_TM0_PREFLIGHT_CLARIFICATION_REQUEST_20260825.md`, Drive `1TAz2SkIjMyVlvDPDGSO4wtYvXfuZocMn` — **byte-verified exact: 4,303 B, SHA-256 `d6460b60c24188ca19e6be880d51dfe6ee2f37666d4d6cbfab2e9a2c9e3cab0c`** (the relay's pasted hash was truncated by one character; the decoded file governs). Panel receipts acknowledged: 6,144-row panel `e108b0a9…b5ca`, 4,096-row extension `5b824843…fd34a` (865 clean ARC-C train + 3,231 hash-ranked clean GSM8K train, seed `20260825`), 1,919-row rejection ledger retained and hashed, CONFIRM screen complete, four tests passing, all attestations clean. **The stop before model contact was correct — the tenth strategy-sourced catch, owned in §1.**

---

## Plain-language summary

My charter ordered a contamination check against a test set whose membership list has deliberately never been written down anywhere — that non-existence is part of how it stays trustworthy. You cannot check rows against a list that must not exist, and the agent rightly refused both to skip the check and to fake it with a weaker one. The resolution threads the needle: a one-time sealed-box job may reconstruct the membership *inside itself*, from the frozen recipe, and emit nothing but scrambled fingerprints — enough to tell whether any panel row matches a sealed evaluation document, never enough for anyone or anything to read what those documents say. The check runs, the seal's purpose is served rather than violated, and the box's output is a row count and nothing else. The two lesser questions are ratified as recommended: layer alignment gets measured on a small frozen calibration slice (with a second slice as a stability check) while all real fitting uses the full panel, and gold answers stay locked inside the correctness scorer — the models only ever see questions.

## 1. R-TM0-P1 — EVAL-E screen: HERMETIC SCREENING INDEX authorized (option 1, hardened); source-family disjointness REJECTED as gate discharge

**The defect is mine (tenth strategy-sourced catch).** The r2 charter's decontamination gate mandated an exact/near-duplicate screen against EVAL-E while the program's own seal discipline keeps EVAL-E membership unmaterialized — I registered an operation that was impossible without a seal-semantics ruling, and the agent's §2 correctly refused to resolve the contradiction silently in either direction.

**Seal semantics, interpreted and registered.** The seal's operative prohibitions are: (i) scoring or evaluation-reading of sealed rows before their designated use, and (ii) leakage of sealed content into any panel, feature, memory, training signal, or human/model context. Non-materialization of membership has been the *implementation* that guaranteed both — but it is the guard, not the law. A decontamination screen exists to enforce (ii); refusing it in the seal's name would protect the guard while abandoning the thing guarded. Therefore:

**Authorized — once, narrowly:** a **hermetic screening-index job**, with all of the following binding properties:
- Runs in an isolated process from the **frozen partition recipe** (pinned corpus snapshots and seeds), reconstructing EVAL-E membership deterministically **inside the job only**. If the recipe cannot deterministically reconstruct membership from pinned inputs, the job STOPS and the wave returns key `DECONTAMINATION-UNRESOLVED` — no improvisation, option 3 fires.
- **Persists only:** (a) salted SHA-256 hashes of normalized full document texts (registered normalization; salt stored with the index) for exact matching, and (b) MinHash/LSH signatures over normalized character shingles for near-duplicate matching at a **registered threshold: estimated Jaccard ≥ 0.8** (shingle size and band structure fixed in the pre-screen receipt). **No plaintext, no ids-to-text mapping, no per-document metadata leaves the job.**
- No model, no labels, no scores, no correctness reader anywhere in the job.
- The index artifact is **sealed-adjacent**: stored content-addressed and hashed, never inspected, queried only by the screen; the screen's only outputs are the **count of dropped panel rows and their panel row-ids**. No output may identify which sealed document matched.
- This authorization is **one-time and non-precedential**: any other use of the index, or any re-materialization for any purpose, requires a new strategy ruling. EVAL-E remains sealed for scoring exactly as before; the attestation line becomes "EVAL-E scored: false; membership materialized: hermetic-screen-only per R-TM0-P1."

**Why option 2 is rejected as discharge of the gate:** benchmark text demonstrably leaks into web-scale corpora — GSM8K and ARC items circulate on the open web — so a FineWeb/code-derived sealed set can textually contain task rows despite source-family disjointness. Disjointness is *plausible*, as the agent says, and it is exactly the kind of plausible that the gate exists to test rather than assume. Source-family disjointness is reported in the receipts as a descriptive fact only.

**Consequence handling (registered before the screen runs):** if the screen drops rows, the panel is **re-frozen as manifest v2** (name/bytes/SHA; v1 retitled superseded) and **no backfill occurs** — replacement rows chosen after seeing screen results would be a selection choice; the panel is whatever survives, with per-stratum sizes reported. Only manifest v2 (or v1 verbatim, if zero rows drop) touches a model.

## 2. R-TM0-P2 — CKA estimator: RATIFIED as recommended, with a stability rider

The agent's estimator is adopted: a **512-row battery-stratified calibration manifest**, frozen by bytes+SHA before model contact; exact unbiased-HSIC (debiased) CKA computed **separately** for the last-token and mean-pooled views; j\* selected by their arithmetic-mean CKA; both view-specific curves reported; uncertainty by row resampling only, never treating the two pooled views as independent (the D-M4 correlated-samples lesson, applied). The stitch fits, G-TM1 gate evaluation, TM-2, and TM-2g all use the **full post-screen panel** unchanged.

**Stability rider (added):** freeze a **second, disjoint 512-row stratified subset** at the same time. j\* must agree between the two subsets **within ±1 layer per teacher**; disagreement beyond that escalates to strategy before any stitch is fit (a calibration subset that can't pick a stable layer is a finding, not a nuisance). Cost: one extra CKA pass on cached states; no additional GPU.

## 3. R-TM0-P3 — Prompt-only state convention: RATIFIED and made binding for the TM line

Prompt-only inputs for every state forward, student and teacher, every battery; **gold answers exist exclusively inside the separately pinned correctness reader**, which touches them only to produce the §3a verified-correct signature. This was the charter's intent and is now its text: it preserves the deployment analogy (TM-2's displacement fields describe what the teacher's computation does *before* any answer exists — the condition under which a future memory would actually be consulted), it prevents teacher-forced tokens from flattering the stitch, and it is the same score-blind separation the agent already implemented for the caching contract. Binding on TM-0 and every TM successor; any future wave wanting answer-conditioned states must register that as a distinct, named feature class with its own leak analysis (the FS-2/`BLOCKED_SOURCE_CONFLICT` lesson, made structural here too).

## 4. Resumption

On receipt, in order: (1) run the hermetic screen per §1 (pre-screen receipt first: normalization, shingle/LSH parameters, salt handling); (2) re-freeze the panel if any row drops; (3) freeze both calibration subsets per §2; (4) finalize the analyzer, run local tests, dry-run price the caching pass; (5) launch only if the measured projection remains under **1.5 A100-hr**, with the batch-invariance probe and fresh Colab session enumeration per the standing requirements. Everything else in r2/r3 unchanged: G-TM1 stop rule, TM-2, TM-2g and its pre-run receipt, the W2′ R-1 rider, one result handoff under the wave rule. The tracker entry for these rulings rides the TM-0 result adjudication.

---

*Signature block*

**Strategy:** R-TM0-P1–P3 issued; tenth strategy-sourced catch owned (a mandated screen impossible under the seal as implemented — the contradiction was mine to resolve, and the agent's refusal to resolve it silently is again the discipline's cheapest purchase); the seal-semantics interpretation recorded in full so it can be overruled before the screen runs rather than discovered after.
**Coding agent:** execute §4; the pre-screen receipt and both calibration manifests precede any model contact; `DECONTAMINATION-UNRESOLVED` is a clean stop, not a failure.
**Mark:** no decision required. The one thing worth your eyes: §1 authorizes a sealed-box reconstruction of EVAL-E membership for fingerprint-screening only — narrow, one-time, nothing readable leaves the box, and EVAL-E stays sealed for scoring. Say the word before the screen runs if you want it done differently.
