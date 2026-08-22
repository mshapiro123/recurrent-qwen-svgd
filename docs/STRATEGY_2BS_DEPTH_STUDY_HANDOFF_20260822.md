# STRATEGY HANDOFF — Depth-Capability Existence Study: Executable Authorization

**Date:** 2026-08-22
**Author:** Strategy agent (session bf36cdbb)
**Status:** LOCKED FOR EXECUTION — score-only; no optimizer; no training
**Basis:** `STRATEGY_2BS_RECON_ADJ_DEPTH_STUDY_20260822.md` §5 (Drive `122c2W-ITzUlwLncl3DZRSonsYL3qg7Z6`, SHA-256 `d9200a484160142a36b5579ba346aed19b7b6b6e5ef0c4f143ffcd85b6b087b4`), **locked by Mark 2026-08-22** ("locked"). D1–D5 rulings binding.

---

## Plain-language summary

The design is signed off. This study asks one clean question on the graph that actually defines our training: does running the loop deeper ever help, and if it hurts, is the culprit the *schedule* of writing-and-re-entering on every loop rather than depth itself? We already have a strong hint — the old one-shot scorer, which defers its single write to the end and never re-enters, scored far higher than the real graph that writes every loop. So the study's main axis is write schedule. Everything is score-only forward passes, both seeds, and — the hard lesson from last wave — every number is tagged with the exact program that produced it, and no curve mixes programs.

## 0. Lock record

Mark locked the §5 design on 2026-08-22. Binding rulings from the reconciliation adjudication: D1 native Stage 2B K4=2/461 is operative truth; D2 the P3.5 provenance footnote is bound; D3 P3.5 results are descriptive for their one-shot graph only; D4 this study; D5 training stays closed until this study reports and the 2B-S charter is drafted against its verdict.

## 1. Question

On the authoritative `Stage2BTaskInferenceGraph`, can any score-only configuration make loop depth K2–K4 additive over K1, or is the depth/loop pathway subtractive as built? If additivity is recoverable, is it recovered by changing the **write schedule** (the reconciliation's write-schedule hypothesis)?

## 2. Regime and cardinal rule

Score-only; **no optimizer constructed; no training; CONFIRM and EVAL-E sealed.** Runtime-pinned (name accelerator / torch / backend / dtype in the receipt). **Both seeds.** Endpoints: **init** (primary — capability before training corrupts it) and **step-1,000 EMA** (secondary — did training change the additivity structure). Effect-floor: additivity threshold **≥ 20 rows over K1**. Battery: the matched 461-row generative slice + per-row DEV-2 margins; both comparators.

**Cardinal rule (no-hybrid-cells).** Every K-cell carries an explicit evaluator-provenance tag naming the exact schedule and graph that produced it. **A curve may only combine cells sharing one evaluator.** This is the standing provenance-tag habit, mandatory here — it is the specific error the reconciliation corrected, and it must not recur.

## 3. Pre-flight consistency check (before any variant cell)

Reproduce the **native-schedule** K-curve on the authoritative graph and confirm it matches the operative truth **162 / 10 / 2 / 2** (seed 0) and its seed-1 counterpart, on the matched 461-row slice. This anchors the additivity bar (native K1) and confirms the harness is the authoritative evaluator. A mismatch is stop-the-line — report before running variants; do not proceed on an unanchored harness.

## 4. Configurations (write-schedule axis primary)

All on the authoritative substrate/head with matched evaluation semantics; each cell provenance-tagged.

1. **Native (baseline).** Identity pass, then per-loop {sidecar update, bridge write, recurrent re-entry}. Reproduces 162/10/2/2; native K1 = the additivity bar.
2. **Deferred-terminal-write, no re-entry.** k sidecar updates with no intermediate bridge write and no recurrent re-entry, then a single terminal bridge write + head. The only change from native is the write/re-entry schedule. Sweep terminal-write-after-k, k ∈ {1,2,3,4}. **The direct test of the write-schedule hypothesis.**
3. **Per-loop write, no re-entry.** Per-loop bridge writes but recurrent re-entry removed between loops — isolates whether the harm is the *write* or the *re-entry*.
4. **Partial-interleave.** Write/re-enter every other loop — dose-response between (1) and (2).
5. **Amplitude cross.** γ ∈ {0, 0.02, 0.05} on configs 1–2; γ=0 is the no-write identity control — separates schedule effect from amplitude.

## 5. Registered discriminator keys

- `ADDITIVE`: some K>1 configuration exceeds native K1 by ≥ 20 rows on matched rows, both seeds.
- `SUBTRACTIVE`: no K>1 configuration exceeds native K1; native and all variants stay ≤ K1 within the floor.
- `SCHEDULE-DEPENDENT` (key outcome): the deferred-terminal-write schedule (config 2) recovers additivity (≥ 20 rows over K1) while native interleaved (config 1) does not — localizing the harm to the per-loop write-and-re-enter schedule.
- Any seed disagreement on a key, or a mixed pattern, escalates to strategy before interpretation (do not resolve locally).

## 6. Decision mapping (pre-registered → successor)

- **SUBTRACTIVE (all schedules):** the depth/loop pathway is non-additive as built; the successor drops or radically re-architects multi-loop depth, and the latent-multi-loop-reasoning thesis meets a boundary result — **escalate to Mark for a program-level decision before charter.**
- **SCHEDULE-DEPENDENT:** the successor gains a primary architectural axis — defer the bridge write / reduce per-loop re-entry — combined with correction-aligned supervision and cluster-conditional writes. Feeds directly into the 2B-S charter.
- **ADDITIVE on native:** unexpected given K4=2; re-examine the native-schedule reads for a measurement issue before proceeding.

## 7. Registered strategy prediction (blind, LOW confidence)

`SCHEDULE-DEPENDENT` — I expect config 2 (deferred-terminal-write) to recover a large fraction of K1 at higher k on the authoritative graph; i.e., the harm is the per-loop write+re-entry schedule, not depth itself. Registered at **low confidence by design**: my mechanism-location predictions have missed twice (F1 STARVED; divergence-at-bridge). The configuration sweep decides, not this guess.

## 8. Deliverable contract

One result handoff on Drive (folder `1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr`), byte-sized and SHA'd in the relay, containing: the pre-flight native-curve reproduction receipt; per-config, per-seed, per-endpoint **provenance-tagged** K-tables (K1–K4 correct-row counts) + per-row margin summaries; the three keys resolved; the runtime pin block; a deviations section (empty or itemized); open questions. Retention list = all of the above, verified present at handoff (retention-verification-at-look). Wave rule: report in one handoff; strategy adjudicates and drafts the 2B-S charter against the verdict.

## 9. Constraints recap

No optimizer anywhere. No training. CONFIRM/EVAL-E sealed. Runtime pinned. Same-session/semantic-pinning discipline; if infrastructure resumes mid-run, disclose and re-pass the pre-flight. Score-only forward passes over ≤461 rows × schedule×γ cells × 2 seeds × 2 endpoints — single-GPU, cheap.

---

*Signature block*

**Strategy:** locked and authorized 2026-08-22 under Mark's sign-off. Prediction registered blind, low confidence.
**Coding agent:** acknowledge by relaying the **pre-flight native-curve reproduction receipt** (§3) before running any variant cell.
**Mark:** informed; training stays closed (D5) until this study reports and the charter is drafted.
