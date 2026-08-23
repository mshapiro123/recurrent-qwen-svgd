# STRATEGY AUTHORIZATION — Final Cell: `per_loop_write_no_reentry` (Ratification Record + Executable Relay)

**Date:** 2026-08-23
**Author:** Strategy agent (session bf36cdbb)
**Status:** RATIFIED BY MARK 2026-08-23 ("OK, ratified") — final GPU cell OPEN; study closes on its result absent a surprise
**Basis:** Cascade adjudication `STRATEGY_2BS_CASCADE_ADJUDICATION_20260823.md` (Drive `1E4YU29vmMYLoxuoDRo6RxyhnyWywby5k`, 13,322 B, SHA `67716c4e…a027`). D1–D5 binding.

---

## Plain-language summary

Mark signed off. The coding agent runs one last comparison — letting the writes pile up across loops while still never re-entering the transformer — with new instrumentation that reports both what the loop accumulated and what it actually deployed. If the curve stays flat, the study closes with its verdict (depth made harmless, not yet useful), no further tuning of this implementation, and charter drafting begins. If it improves or collapses, that surprise comes straight back for adjudication.

## 1. Ratification record

Mark ratified 2026-08-23 ("OK, ratified"), covering the cascade adjudication's rulings: (1) `SCHEDULE-NEUTRALIZED` granted provisionally on the deferred arm (K1+20-miss machine record preserved); (2) final cell `per_loop_write_no_reentry` authorized, with the pre-registered three-way map binding and `partial_interleave` closed absent a surprise+relay; (3) dual write-magnitude telemetry adopted as a standing instrument; (4) DEV-2 margin panel on the deciding cells before the final key banks; (5) D-M3 deferred (charter line item only); (6) D-M4 carried as sizing prior with the correlation-aware power calculation a charter prerequisite; (7) close-out interpretation bound — any non-recovery ending fires the operative effect (no more tuning of this implementation), with final key `SCHEDULE-NEUTRALIZED` rather than `SUBTRACTIVE`.

## 2. Relay to the coding agent (authorized to run)

> Cascade adjudication RATIFIED by Mark 2026-08-23. Execute the final cell:
> 1. `per_loop_write_no_reentry`, γ=0.05, K1–K4, both seeds, init endpoint, matched 461-row slice, provenance-tagged cells, runtime-pinned. (~2.4 A100-hr.)
> 2. **Standing instrument, effective now:** report per cell BOTH unnormalized accumulated write magnitude AND deployed post-aggregation magnitude.
> 3. **Pre-registered three-way map (binding):** accuracy flat while accumulated write magnitude grows → final key `SCHEDULE-NEUTRALIZED`, cascade ends, do NOT open `partial_interleave`; accuracy improves (terminal aggregation was suppressing real multi-step signal) → stop and relay; accuracy collapses (repeated writes harmful without re-entry) → stop and relay. Any seed split → stop and relay.
> 4. Absent a surprise: score the **DEV-2 margin panel on deferred-terminal-write K1+K4, both seeds** (plus the per-loop cells if they surprised) — the deciding schedule must be margin-backed before the final key banks.
> 5. D-M3 stays deferred — no GPU for it. No optimizer, no training, CONFIRM/EVAL-E sealed, effect-floor, no hybrid cells, durable receipts — all unchanged.
> **Acknowledge by relaying the final-cell result with the dual write-magnitude telemetry (and the margin panel, if reached) in one result handoff under the wave rule.**

## 3. What happens on the result

Flat → study closes as `SCHEDULE-NEUTRALIZED` (margin-backed); training stays closed (D5); strategy drafts the 2B-S charter against the four measured constraints (signal preservation through depth primary; accumulate-don't-dilute aggregation; supervision must create correction-alignment; cluster-conditional writes) with the D-M4 correlation-aware power calculation as a prerequisite. Improves/collapses/split → strategy adjudication first. Either way, the next artifact after the agent's handoff is the charter draft or a surprise adjudication — no further GPU beyond §2 without a new ratification.

---

*Signature block*

**Strategy:** authorization recorded and relayed 2026-08-23.
**Coding agent:** authorized for §2 immediately; acknowledge with the result handoff.
**Mark:** ratified 2026-08-23; next decision point is the charter draft (or a surprise adjudication).
