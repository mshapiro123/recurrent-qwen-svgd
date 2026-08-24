# Bicameral W1 Injection-Position Clarification Request

Date: 2026-08-24. Coding-agent stop-the-line report to strategy.

## 1. Conflict

The executable W1 authorization defines the injection as the final-cell
deployed-write convention, exactly, and also says that the row vector is
"applied at the write interface at the terminal position." Those clauses do
not identify the same tensor operation without a clarification:

- The frozen final-cell implementation passes the full active-token mask to
  the bridge and therefore writes every active token position.
- The current W1 implementation matches that behavior by broadcasting the
  row vector across every active position.
- A literal terminal-token reading would write only the last active token and
  would define a different score-only intervention.

The authorization explicitly requires a stop if the formula differs
mechanically from the final-cell implementation. The agent therefore stopped
before seed 1 reached any margin cell and did not silently choose between the
two readings.

## 2. Work preserved

Seed 0 completed the all-active-position implementation over all 2,048 DEV-2
rows and all 11 Phase-A cells. It remains engineering evidence pending this
ruling and is not banked as the registered W1 result.

- Seed-0 archive: `artifacts/bicameral_w1_20260824/remote/seed_0_bundle.tar.gz`
- Bytes: 35,002,614
- SHA-256: `e465cc3252defde42a2a571b8d5e87352cbb6de4faa75231b5f91834cd63f7fc`
- Seed-0 summary SHA-256:
  `901829de34f80d745f223dc27bfa231f10e1e2498c18606bf1f7b2df6e1648b3`
- Optimizer steps: 0
- CONFIRM scored: false
- EVAL-E scored: false

Seed 1 was terminated during target extraction, before any margin cell. Its
partial target extraction was not checkpointed and carries no scientific
result. The paid A100 session was terminated; `colab sessions` returned no
active sessions.

## 3. Requested ruling

Choose one interpretation before execution resumes:

1. **Recommended: terminal interface, all active positions.** This preserves
   exact equivalence to the frozen final-cell bridge mask. Bank seed 0 and
   resume seed 1 under the existing code.
2. **Alternative: terminal token only.** Supersede seed 0, change the mask and
   RMS estimator to the last active token, add an exact mask test, and rerun
   both seeds from their original registered endpoints.

No other constants, targets, controls, or decision rules are implicated.

