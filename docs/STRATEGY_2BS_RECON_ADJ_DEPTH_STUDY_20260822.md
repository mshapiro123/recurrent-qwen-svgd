# STRATEGY MEMO — Reconciliation Adjudication + Depth-Capability Existence Study Design

**Date:** 2026-08-22
**Author:** Strategy agent (session bf36cdbb)
**Status:** ADJUDICATION (reconciliation ratified; operative truth banked; my prediction scored honestly) + DESIGN (the authorized depth-capability existence study, for pre-signature review and lock)
**Adjudicates:** `PAPER2_STAGE2BS_RECONCILIATION_RESULT_HANDOFF_20260822.md` (Drive `1LRXB5QpAMOjdtPhm5pKBYnD0MPClGzDM`)
**Handoff verification:** 12,613 bytes; SHA-256 `29ccff05bbffb6863155ce50cff8ca4424b9a7b03057d600fda92cf70c65ebc8` — **byte-verified by strategy**. Executed lock SHA `0396ae35…4121`. Optimizer steps: 0. Training: none. CONFIRM/EVAL-E sealed. Commit `78f519fb`, 5 tests. Colab released.

---

## Plain-language summary

The reconciliation answered the question cleanly and both seeds agree. Our two scoring programs were never two versions of the same computation — they run genuinely different procedures. The old amplitude scorer lets the sidecar refine itself four times in isolation and then makes a single correction at the end; the real Stage 2B evaluator writes into the model and re-runs the recurrent block on every loop. The agent proved each program reproduces its own live evaluator bit-for-bit, and proved by tracing the source that the Stage 2B program is the one that generated our training losses and floors. So the operative truth is settled: on the graph we actually train and serve, four loops score 2/461, not 160. The 160 belongs to the other program and stays there.

Two things matter going forward. First, I have to score my own prediction honestly: I said the two programs would first diverge at the bridge/amplitude step. They diverge earlier — at the very first sidecar update, before any bridge write. My structural calls (the amplitude scorer is one-shot; Stage 2B is authoritative) were right, but the specific location I predicted was wrong, so the registered prediction is recorded as unsupported. That is the third time this program that I've called the shape of a thing correctly and the exact mechanism-location incorrectly, and I'm treating that as a calibration signal, not a coincidence.

Second — and this is the real prize — the contrast between the two programs is itself a strong clue about *why* depth hurts. The only structural difference between the program that scores 160 and the program that scores 2 is the write schedule: defer the bridge write to the end and skip per-loop re-entry (160), versus write and re-enter on every loop (2). That suggests the harm may not be depth itself but the *per-loop write-and-re-enter schedule*. The study I'm designing below tests exactly that on the authoritative graph, with no hybrid cells this time — every number tagged with the exact program that produced it.

## 1. Adjudication — reconciliation ratified

Ratified as delivered. Both instrumented traces reproduced their native evaluators bit-exactly (max delta 0, both seeds); the disagreement is not a hook artifact. Static call-chain provenance passed all four registered checks: Stage 2B training invokes the Stage 2B recurrent wrapper, DEV floors invoke `Stage2BTaskInferenceGraph`, and the P3.5 scorer runs four flow steps + one terminal bridge write. **`Stage2BTaskInferenceGraph` is the success-defining graph; native K4 = 2/461 is the operative Stage 2B result.** The registered losses and floors were not trained under one graph and served under another — **no serving repair is indicated.** The strict prefix non-identity (max delta 0.0625, cosine ≈ 1.00007, downstream initializer states cosine 0.99999845) is an instrumentation-level numerical difference (monolithic frozen-base forward vs manual prelude/block/coda), correctly preserved in the receipt and correctly excluded as the cause of the 160-vs-2 outcome.

**Mechanistic localization (ratified):** the first *algorithmic* divergence is `loop_1_post_state` (cosine 0.65/0.68 across seeds). At the first sidecar update, P3.5 calls single-lane `base_flow.step`; Stage 2B performs its multi-lane flow + constitutive update under M2, then writes through the bridge and re-enters the recurrent block. The states separate at that first update and never reconverge; final logits cosine 0.64, top-1 disjoint (1249 vs 785), top-128 overlap ~50%. Structural, seed-replicated.

## 2. Honest prediction scoreboard

| Registered prediction | Result | Score |
|---|---|---|
| First divergence at the bridge/amplitude application | Divergence begins at the first sidecar update, before any bridge write | **UNSUPPORTED** |
| P3.5 is a one-shot corrective estimate, not full iterated recurrence | Confirmed by dynamic trace + source provenance | SUPPORTED |
| Stage 2B graph defines registered success | Confirmed | SUPPORTED |

Compound registered prediction: **UNSUPPORTED** — the primary discriminator (divergence location) was the load-bearing claim, and it failed. I do not claim a partial pass.

**Calibration note (on the record).** This is now a pattern across three waves: F1 `STARVED` (wrong — W_P moved 2×), divergence-at-bridge (wrong — it's the first sidecar update), each alongside a *correct* structural/directional call. My read of *what kind of thing is happening* has been reliable; my read of *exactly where the mechanism sits* has not. The operational consequence is already baked into the study below: it is deliberately a **diverse-configuration empirical sweep**, not a bet on one predicted mechanism. I register a blind prediction for it (§5) but explicitly at low confidence — the data decides.

## 3. The architectural lead — write schedule as the candidate mechanism

The reconciliation is not only a correction; it is a natural experiment. Two programs on the *same frozen substrate and sidecar* differ only in write schedule and score 160 vs 2 at "K4":

- **P3.5 one-shot schedule:** four sidecar-only refinements, **no per-loop re-entry**, one terminal bridge write → 160/461.
- **Stage 2B native schedule:** identity pass, then three cycles of **{update, bridge write, recurrent re-entry, coda}** → 2/461.

The agent's §10 caution is correct and I hold to it: this does **not** establish that the one-shot computation is scientifically preferable, only that its result cannot be transferred to Stage 2B. But it makes a sharp, testable hypothesis: **the harm is the per-loop write-and-re-enter schedule, not depth per se.** If deferring the write and dropping per-loop re-entry recovers task capability *on the authoritative graph*, the successor's architecture changes at the schedule level — a lever not previously in the 2B-S outline. That is precisely what the study tests.

## 4. Rulings on the five requested decisions

**D1 — Bank the scorer-artifact mapping + native Stage 2B K4 = 2/461 as operative truth. ACCEPTED.** Recorded in the tracker (row 17/19) and Paper Two.

**D2 — Provenance footnote on the P3.5 amplitude-surface result. ACCEPTED.** Exact wording bound: *"P3.5 amplitude-surface results were produced by the one-shot amplitude scorer — one frozen-base forward, four sidecar-only state refinements with no recurrent re-entry, a single terminal loop-4 bridge write, one head projection. They are valid for that graph only and are not evidence of fourth-loop computation in the Stage 2B recurrent graph."* Applied to tracker row 10 and to be carried into the Paper Two amplitude section.

**D3 — Preserve P3.5 results as descriptive for their one-shot graph; not Stage 2B depth evidence. ACCEPTED.**

**D4 — Design the depth-capability existence study on `Stage2BTaskInferenceGraph`, matched-graph K1–K4, no hybrid cells. DELIVERED below (§5).**

**D5 — Keep training closed until that design is reviewed and locked. ACCEPTED.** Nothing trains; the study is score-only; it goes to Mark's pre-signature review before the coding agent runs it.

## 5. Depth-Capability Existence Study — design (for review and lock)

**Question.** On the authoritative `Stage2BTaskInferenceGraph`, can any score-only configuration make loop depth K2–K4 additive over K1, or is the depth/loop pathway subtractive as built? And if additivity is recoverable, is it recoverable by changing the **write schedule** (the §3 hypothesis)?

**Regime.** Score-only; no optimizer; no training; CONFIRM/EVAL-E sealed. Runtime-pinned (name accelerator/torch/backend/dtype). **Both seeds.** Endpoints: **init** (primary — the architecture's capability before training corrupts it) and **step-1,000 EMA** (secondary — did training change the additivity structure). Effect-floor law: additivity threshold ≥ 20 rows over K1. Battery: the matched 461-row generative slice + per-row DEV-2 margins; both comparators.

**Cardinal rule (this wave's lesson).** **No hybrid cells.** Every K-cell carries an explicit evaluator-provenance tag naming the exact schedule and graph that produced it. A curve may only combine cells that share one evaluator. This is the standing provenance-tag habit, now mandatory for this study.

**Configurations — the write-schedule axis is primary.** All run on the authoritative substrate/head with matched evaluation semantics:

1. **Native (baseline).** Identity pass, then per-loop {update, bridge write, re-entry}. Reproduces the operative 162/10/2/2. Anchors the comparison; its K1 = the additivity bar.
2. **Deferred-terminal-write, no re-entry.** k sidecar updates with no intermediate bridge write and no recurrent re-entry, then a single terminal bridge write + head — the P3.5-*style* schedule brought onto the authoritative graph, so the only change from native is the write/re-entry schedule. Sweep terminal-write-after-k, k ∈ {1,2,3,4}. **This is the direct test of §3.**
3. **Per-loop write, no re-entry.** Isolate whether the harm is the *write* or the *re-entry* by keeping per-loop writes but removing recurrent re-entry between loops.
4. **Partial-interleave.** Write/re-enter every other loop — a dose-response on schedule frequency between (1) and (2).
5. **Amplitude cross.** γ ∈ {0, 0.02, 0.05} on configs 1–2 (cheap; γ=0 is the no-write identity control), to separate the schedule effect from amplitude.

**Registered discriminator keys.**
- `ADDITIVE`: some K>1 configuration exceeds native K1 by ≥ 20 rows on matched rows, both seeds.
- `SUBTRACTIVE`: no configuration with K>1 depth exceeds native K1; native and all variants stay ≤ K1 within the floor.
- `SCHEDULE-DEPENDENT` (the key outcome): the deferred-terminal-write schedule (config 2) recovers additivity (≥ 20 rows over K1) while native interleaved (config 1) does not — localizing the harm to the per-loop write-and-re-enter schedule.
- Any seed disagreement on a key, or a mixed pattern, escalates to strategy before interpretation.

**Decision mapping → successor (pre-registered).**
- **SUBTRACTIVE (all schedules):** the depth/loop pathway is non-additive as built; the successor drops or radically re-architects multi-loop depth, and the program's latent-multi-loop-reasoning thesis meets a boundary result — escalate to Mark for a program-level decision before charter.
- **SCHEDULE-DEPENDENT:** the successor architecture gains a primary new axis — defer the bridge write / reduce per-loop re-entry — combined with correction-aligned supervision (the surviving 2B-S lever) and cluster-conditional writes. Most actionable outcome.
- **ADDITIVE on native:** unexpected given K4=2; re-examine the native-schedule reads for a measurement issue before proceeding.

**Blind strategy prediction (registered, LOW confidence per §2 calibration): `SCHEDULE-DEPENDENT`.** I expect the deferred-terminal-write schedule (config 2) to recover a large fraction of K1 at higher k on the authoritative graph — i.e., the harm is the per-loop write+re-entry schedule, not depth itself. Registered explicitly as low-confidence: my mechanism-location calls have missed twice; this study exists to let the configuration sweep decide, not to confirm my guess.

**Cost.** Score-only forward passes over ≤461 rows × a handful of schedule×γ cells × 2 seeds × 2 endpoints — single-GPU, no training, cheap. Retention list: per-config provenance-tagged K-tables + per-row margins + runtime pin, verified present at handoff (retention-verification-at-look).

## 6. Sequencing

This memo → Mark's pre-signature review of the §5 design → lock → coding agent runs the study (score-only) → strategy adjudicates → **then** the 2B-S charter, whose architecture now depends on the SUBTRACTIVE / SCHEDULE-DEPENDENT verdict. Training stays closed until the charter is ratified. Paper Two carries the reconciliation as a methods-integrity result (two evaluators, one authoritative) and the corrected operative K-curve.

---

*Signature block*

**Strategy:** adjudicated, scored, and designed above, 2026-08-22. Prediction registered at low confidence by design.
**Coding agent:** reconciliation ratified; no further judgment needed there. The §5 study awaits Mark's lock before execution.
**Mark:** pre-signature review requested on the §5 depth-capability study design (reply "locked"/"ratified" or amend); D1–D5 rulings recorded.
