# STRATEGY RULING — Prelude Amendment PA-1: A-D1 F1-Key Repair (RATIFIED)

**Date:** 2026-08-21
**Amends:** `STRATEGY_2BS_PRELUDE_HANDOFF_20260821.md` (Drive `17nYHGA1dzY-G-aC614ynkSspt7_lrPbk`, SHA `0c738c28…be566`), §2 registered key only. All other content unchanged.
**Trigger:** coding-agent implementation finding at commit `aae08373` (`codex/stage2bs-preludes`, 65/65 tests): `W_P0` is exactly zero in the as-built config, so the registered relative-movement ratio (‖ΔW_P‖_F/‖W_P₀‖_F)/(‖ΔW_H‖_F/‖W_H₀‖_F) is undefined (division by zero). The agent correctly blocked the machine lock rather than resolving locally.

## Ruling

**RATIFIED with one addition.** The F1 key is replaced by the absolute-movement ratio:

    R_F1 = ‖ΔW_P‖_F / ‖ΔW_H‖_F

`STARVED` = R_F1 ≤ 0.25 on both seeds. `NOT_STARVED` = R_F1 ≥ 0.75 on either seed. Between: `PARTIAL`. Thresholds carry over unchanged; the matrices have identical shapes, so the absolute Frobenius ratio is dimensionally fair, and with W_P₀ = 0 it has a clean reading — ‖ΔW_P‖_F is simply the final norm ‖W_P‖_F, so the key doubles as a "did W_P ever leave zero" test, which is exactly the F1 serialization question.

**Added guard (strategy):** if ‖ΔW_H‖_F is numerically zero (below dtype-epsilon scale for the matrix size), the key resolves to `DEGENERATE` and escalates to strategy rather than reporting STARVED — a zero denominator would otherwise let noise in ΔW_P masquerade as a verdict. The result table must report the raw values ‖ΔW_P‖_F, ‖ΔW_H‖_F, ‖W_H₀‖_F, and both final norms alongside the ratio, so the ratio is never the only view.

## Blind-status note

No A-D1 measurement was unblinded before this amendment; the repair is definitional, discovered at implementation time — the same posture as the KP-1R repaired-estimand precedent. The registered prediction is unchanged and still blind: `STARVED`. The Prelude-1 specification is untouched.

## Effect

Machine lock unblocked. Proceed at `aae08373`; the pre-flight K-sweep reproduction gate relay remains the next required communication before Prelude-1 probe cells run.

---

**Strategy:** ratified 2026-08-21 (PA-1). Relayed to Mark in-session.
