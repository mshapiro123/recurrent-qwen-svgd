# Strategy — P3.3 Lock Erratum e1: Four Bindings Before First Step

Date: 2026-08-11. Amends the ratified P3.3 lock (Drive `1q2MxeBKtMhm87yPCHKaL_Ssnpix3Mjbn`, SHA `45e2221b…455c8d`). Status: issued as clarification-level bindings within the ratified lock's intent — items 1, 2, and 4 resolve ambiguities, item 3 adds one measurement artifact the lock's own estimators require. Mark may veto any item; absent veto these are effective now. A5 is acknowledged satisfied: the forecast receipt (`PAPER2_PHASE3_LINEAR_DECODABILITY_FORECAST_RECEIPT_20260811.md` + `summary.json`, commit `131e720e`) is banked, seed-replicated (0.0952 / 0.0874, both ridge 1e5), and reads exactly as the B5 framing requires — weak but real linear decodability, not a bound on the nonlinear bridge.

## e1.1 — γ = 0.02 versus exact migration: the ceiling is an operating clamp, not part of the identity test

The A1 migration-equivalence assertion runs **with the ceiling disabled**: raw migrated model against the Phase 2 reference, exact scalar-reference writeback, as built. Only after A1 passes is γ = 0.02 enabled as the training-time operating configuration, from step 0. If any migrated per-loop gate value exceeds 0.02, the ceiling binds immediately at init — that is acceptable, is the operating rule doing its job, and is **reported per loop in the run receipt** rather than treated as a failure. The identity test proves the migration; the clamp governs the run. They are different obligations and are asserted in that order.

## e1.2 — the c = 0.15 factor: an audit condition, not a code symbol; do not implement it

Per t1 B3 the code is canonical, and the lock erred in listing c = 0.15 among the run-config symbol bindings — no such factor exists in the implemented bridge, and none is to be added. Reclassification:

- **The training-time write bound is the implemented equation exactly**: delta normalized to the capped state RMS reference (p99 = 0.550893), multiplied by the gate, ceiling γ = 0.02. Reproduced verbatim from source in the run config with those two symbol bindings only.
- **c = 0.15 is the V-series measured safe radius — an analysis quantity.** It lives in the *audit condition*, where it always did: the forced-open π_dir measurement sets the write magnitude to the V-series audit radius (0.15 × the capped-RMS reference), with the trained bridge supplying direction, and the **oracle denominator computed at the identical magnitude** per B2's matched-conditions rule. Forced-open means gate replaced by this audit constant — not gate = 1, which would evaluate at a magnitude the perturbation studies never validated. π_dep remains the deployed accounting: realized gate, realized magnitude, matched denominator.

## e1.3 — gate precision requires negatives the audit slice does not contain: one new artifact

Correct: the 4,096-row audit slice is positives only, so it supports recall but neither precision nor χ. Binding: draw a **negative audit slice of 12,288 rows** (3:1 to the positive slice, mirroring the training ratio) by the same confidence-rank criterion, from confident-agreement rows **excluded from the 103,563 training negatives**, hashed before the first training step and added to pre-run assertion A3. Gate recall is measured on the positive slice; precision and false-positive rate against the combined slices; collateral χ on the negative slice, which is also its proper estimand — flips induced on held-out confident-agreement rows. The agreement pool is large enough that this costs nothing but the draw.

## e1.4 — optimizer and cadence: pre-registered, with the look count fixed by the calibration

- **Looks: exactly 20**, evenly spaced over the training budget. The Tier-S and Tier-W operating characteristics were calibrated on that schedule, and the certificate is void at any other cadence. If the session budget forces a different step count, spacing changes, the look count does not.
- **Optimizer: the banked Phase 2 A2-stage configuration carries over** (family, betas, weight decay, clipping) with the learning rate and total step count **declared in the run config before the first step** and unchanged thereafter — pre-registered, not tuned mid-run. Step count is set to fill one A100 session per seed with the 20-look schedule and the end-of-run audit battery inside the session. No adaptive early stop exists apart from the rule inventory. Audit-slice π measurement runs at end of training, with one optional midpoint reading at look 10, both from the same estimator spec.

## Effect

With e1.1–e1.4 bound and the negative audit slice hashed into A3, the pre-run assertion list is complete and **P3.3 training is cleared under the ratified lock**. Report-back obligations unchanged, plus the per-loop init-clamp report of e1.1. The plain-language version: the identity check runs before the safety clamp switches on, the 15% radius belongs to the measurement rather than the model, the gate now gets judged on examples it should ignore as well as ones it should act on, and the run's dials are written down before the run starts. Nothing about the experiment's question changed. It is cleared to run.