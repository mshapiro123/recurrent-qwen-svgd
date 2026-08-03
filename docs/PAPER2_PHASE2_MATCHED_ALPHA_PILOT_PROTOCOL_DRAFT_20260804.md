# Phase-2 Matched Alpha Pilot Protocol, Draft 1

Date: 2026-08-04

Status: draft before lock. Training is not authorized by this document. The protocol becomes lockable only after the CPU canonicalizer arbitration and exact floored-fraction receipt land.

## Question

Among the shared-basis canonical transforms with alpha in `{0.0, 0.5, 1.0}`, which scaling produces the best verified acceptance after the full student module is trained under a matched DEV-only budget, without violating identity, frozen-lineage, tube, preservation, or upper-model-quality constraints?

Experiment 0A/0B established numerical validity and geometry differences. They did not select alpha.

## Fixed arms and invariants

- Arms: alpha `0.0`, `0.5`, and `1.0`.
- Minimum identical seeds: `{0, 1}` for every arm.
- Canonicalizer method: filled from the CPU arbitration receipt before lock.
- Shared across arms: calibration rows, canonical mean, PCA orientation, retained rank, raw and effective eigenvalues, recurrent architecture, initialization stream, optimizer, batch order, budget, evaluation rows, and stopping rule.
- Sole alpha-arm difference: multiplication by `lambda_eff ** (-alpha / 2)`.
- Four future slots are populated. Four trace/span slots remain reserved, masked out of every loss, and excluded from effective-rank telemetry.
- Full fp32 on recurrent-gradient paths.
- Loop cap `K <= 4`.
- V1d writeback constants: `c = 0.15`, RMS cap `0.5508932316303252`.

## Proposed matched budget for strategy lock

- Optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`.
- Batch size: 128 anchor pairs.
- Steps: 1,000 per seed and alpha arm.
- Evaluation cadence: step 0, 250, 500, 750, and 1,000.
- No early stopping on quality or acceptance. Abort only on a hard assertion, non-finite state/loss, or frozen-hash mutation; aborted arms write receipts and do not silently restart.
- Per-module clipping: refiner 1.0, bridge 0.5, heads 1.0.

These values extend the 600-step 0B trainability screen enough to observe convergence while keeping this a DEV-only pilot. They are proposals, not locked constants, until strategy accepts this draft.

## Lexicographic selection rule

1. Exclude any arm with an identity, frozen-lineage, tube, or preservation assertion violation.
2. Require final-answer and upper-model quality non-inferiority to the zero-loop reference under the existing two-tier margin. Quality is a disqualifying gate, not a ranking axis.
3. Rank qualifying arms by verified acceptance on matched DEV rows.
4. Tie-break in order by flow convergence, gradient balance, then clipping burden.
5. Treat two arms as practically equivalent when the paired bootstrap 95% CI on the difference in mean accepted length is wholly inside a relative `+/-2%` band, or when between-seed variance exceeds the between-arm difference. Alpha `0.5` wins an equivalence outcome.

No additional seeds are added because two-seed variance is large. High seed variance is itself the equivalence reading.

## Refinement trigger

- If alpha `1.0` beats `0.5` outside the equivalence band, add exactly one alpha `0.75` arm with the same seeds and budget before selection locks.
- If alpha `0.0` beats `0.5` outside the band, add exactly one alpha `0.25` arm under the same conditions.
- Otherwise the three-arm screen is sufficient and no denser grid runs.

## Required measurements

- Verified acceptance and mean accepted length, with paired row-level intervals.
- Upper-model/final-answer quality versus zero-loop.
- Flow convergence and endpoint error by loop.
- Per-module and per-loss gradient norms, conflict cosines, gradient coefficient of variation, JVP gains, clip fractions, gate-open rates, and effective rank on populated slots only.
- Per-loop canonical update ratio and radial drift.
- Realized writeback ratio and tube-equivalent radius.
- Trained-module analogue of the 0B path audit: among accepted updates, fraction worsening probe KL and fraction worsening final-answer quality, stratified by gate state.
- Per-arm correlation table among probe KL, probe top-1 agreement, and verified acceptance.
- Alpha `0.0` failure prediction: worse gradient balance/clipping and decision alignment. This is a prediction, not a gate.

## Hard assertions

- Zero-loop bit identity under the checkpoint-integrated path.
- Teacher, canonicalizer, probe, and pretrained backbone are frozen, run under no-grad where applicable, and hash-identical before/after.
- Target construction and sampled tokens are gradient-isolated.
- Constants-file hash and all frozen artifact hashes appear in every receipt.
- Document isolation holds in every packed batch.
- No loss or rank telemetry reads a masked trace/span slot.
- Shared PCA basis and all non-alpha state are byte-identical across arms.

## Interpretation boundaries

- Pilot selection is DEV-only and does not establish confirmation performance.
- Probe KL and probe top-1 are diagnostic surfaces; verified acceptance is the primary currency.
- A winning alpha is conditional on the selected canonicalizer, budget, seeds, and this student architecture.
- The pilot does not authorize E1 confirmation until the resource note and full E1 lock are complete.

