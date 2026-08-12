# Strategy — P3.4 Charter: The Answer-Distillation Main Run (Awaiting Ratification)

Date: 2026-08-12. Authorized in direction by Mark's Option A ruling on the i1 re-read (Drive `1U8gN4j6TP1dD8vJrgu_rYWaIALjOAPBe`). Governing chain: charter r2 → t1 (ratified) → reasoning-scope addendum → policy resolutions r2 → P3.3 lock + e1/e2 + i1 record. This charter specifies the main campaign. The coding agent builds its prerequisites and binds its receipts, and **training begins only after Mark ratifies the executed lock** — this document plus the bound numbers.

## 1. Purpose and the question

P3.4 trains the sidecar's final distribution toward the teacher on the teachable stratum and measures, for the first time, whether token-level correction converts into task-level answer improvement. The currency is gap_closed per target battery with raw deltas beside every ratio, read on DEV halves as curves at every checkpoint window. Expectation set honestly from the banked physics: at 15 to 20 percent aim capture the oracle agreement ceilings scale to roughly 0.6 to 1.1 token-level points, and the task-level yield of that through generation is the unknown this campaign exists to measure. A positive, replicating gap_closed curve earns the P3.6 confirmation. A flat one, with the diagnostics below, tells us which lever to pull or which boundary to write.

## 2. Prerequisites (build order, all receipted before the lock executes)

- **2a. The task inference graph, v1 defaults set here, bound by the coding agent.** At generation, the sidecar operates per emitted token: the scratchpad initializes fresh from the current prefix's states, runs K = 4 flow loops, and the bridge writes at the current position under the controller's gate ceiling. No state persistence across emitted tokens in v1 — cross-token persistence is a named P3.5 lever, not a default. The draft head is inactive during battery scoring (acceptance remains separate telemetry). Battery decoding is greedy, position zero closed, reader per the pinned P3.1 manifest at canonical serving precision. The per-loop diagnostic coda runs on the sentinel panel only, audit-only, yielding accuracy-versus-K and the marginal-improvement curve. Any deviation from these defaults is a lock amendment, not an implementation choice.
- **2b. The A_r audit — the fork's pricing receipt, CPU on cached artifacts, first deliverable.** Compute the fraction of the cached oracle directions' energy lying in the subspace the bridge's readout actually spans, and alongside it the same fraction for the leading state-covariance subspace at matched rank. High A_r relative to realized capture means the state carries aim the projection cannot extract and the capacity arm is justified. Low A_r means the information itself is thin and the slot-supervision arm is the lever. The receipt returns to strategy with the fork recommendation before any arm spends GPU.
- **2c. Guardrail recalibration onto the task estimator** (per e2.3, now possible with 2a in hand): paired augmented-versus-base battery correctness on a 1,024-row DEV panel, empirical discordance and autocorrelation, same three-tier structure — Tier-S Δ_cat searched at ≥ 99 percent power and familywise false-stop ≤ 1e-4 on the campaign look schedule, Tier-W at the −3-point class with demotion consequence (the controller now has rungs to demote), Tier-E unchanged at the claim. The e2 token-retention panel continues in parallel as telemetry.
- **2d. χ_max calibrated** per t1 B2: from measured collateral of oracle writes at each rung's gate ceiling plus a stated margin, estimator in the same clause — the controller cannot advance without it.

## 3. Initialization, trainable set, and losses

**Initialization:** the i1 endpoint checkpoints, both seeds, hash-pinned — the aim-pretrained bridge with the verified selector. **Trainable:** bridge (all parameters including Q/K/V and output projection), gate head, control state. Flow and draft head frozen — flow unfreeze at 0.1× LR and draft-head work are P3.5 levers. The gate resumes training in the joint setting (its freeze was an i1 attribution device, not doctrine), with its P3.3/i1 statistics as the baseline it must not degrade below Tier-W scrutiny.

**Loss set, with per-loss share contracts per the standing law:** verifier-masked `L_KL` (top-K = 128 lattice, τ = 1, teacher-wrong rows masked on the verified stratum, unmasked-limitation stated on the agreement stratum) and `L_CE` (verified stratum only) as the system primaries; `L_aim` on write-stratum positives and `L_gate` on the tri-state labels as channel primaries; preservation KL on confident agreement at λ_p per the controller. **v1 share targets, calibration-derived, revisit-labeled, enforced as per-loss inequalities on the trailing-window training estimator with two-tier warn → stop: L_KL ≥ 35 percent, L_aim ≥ 15 percent, L_CE ≥ 10 percent, L_gate ≥ 3 percent, preservation ≤ 25 percent.** No loss starves inside a bucket again. Deep supervision at w_k = k/K with subsampled k permitted per the charter's cost note.

## 4. The annealing controller, live

The controller runs as chartered: per evaluation window it reads π_dep and χ on the audit slices and the battery floor check, advances at most one rung per window (requirements π_dep ≥ 0.10 / 0.25 / 0.40 with χ ≤ χ_max), sets the gate ceiling γ ∈ {0.02, 0.08, 0.20, 0.50} and λ_p ∈ {1.0, 0.5, 0.2, 0.05}, and demotes one rung on a Tier-W event. Rung 1 eligibility is already met on the banked numbers, so the campaign opens with a legitimate fourfold widening of the deployed channel — the first time the system operates beyond the pilot ceiling, which is itself a measurement. Controller state and transitions appear in every evidence record. The rung table remains v1-default and revisit-labeled.

## 5. The pre-registered fork: capacity or slots

One arm may be added to the campaign without a new charter, chosen by the A_r receipt and ratified by a one-line strategy confirmation:

- **Capacity arm (A_r high):** a bounded aim-path expansion retaining the validated selector and safety machinery — the frozen-selector second tower or a widened/two-layer output projection, parameter budget capped at 1.0M additional, zero-initialized so attachment recovers the current system exactly, trained under the same share contracts.
- **Slot-supervision arm (A_r low):** the registered LOTUS-style loss — project each future slot through a zero-initialized lift and the frozen tied head, cross-entropy against the lattice's teacher tokens at the corresponding horizon, deep-supervised — attacking the information side by teaching the scratchpad to name the teacher's future tokens. Slot decodes through the frozen head join the audit telemetry as the interpretability readout.

Either arm reports its own gap_closed delta against the no-arm configuration on paired windows.

## 6. Measurement and audits

Per checkpoint window: gap_closed and raw delta per target battery, per K, on DEV halves with document-bootstrap CIs; the floor battery trend; accuracy-versus-K and per-loop marginal improvement on the sentinel panel; π_dir, π_dep, gate recall/precision, and χ on the audit slices; retention trends on the e2 panel; EAL as telemetry; Tier-1 observatory scalars throughout. Tier-2 checkpoint audits per the observatory memo: 16 replicas on sentinel subsets, Gaussian W2 mean/covariance decomposition against the canonical teacher, covariance spectra with empirical nulls, A_r refresh, JVP gains, overlap distributions. The failure-signature table is the standing post-mortem key. Every run is a curve — the 20-point pattern, never a verdict.

## 7. Guardrails (rule inventory, bound at lock execution)

Absolute stops: non-finite loss/state, base or frozen-hash mutation, CONFIRM or EVAL-E contact. Calibrated: Tier-S on the task panel at the 2c-searched Δ_cat, two consecutive looks. Trajectory: gradient explosion at 10× trailing-100 median, three consecutive. Contracts: the per-loss share inequalities of §3, warn → stop. Demote-and-flag: Tier-W at the recalibrated −3-point class, two consecutive looks — controller demotes one rung, strategy review flagged, training continues. Everything else is telemetry. Look schedule matches the 2c calibration receipt or the certificate is void.

## 8. Budget, deliverables, do-not-claim

A100 sessions, both seeds, resumable, sessions released promptly, step count and LR pre-declared per e1.4 practice. Report-backs per window with the standard evidence-record discipline and one creativity slot. Do-not-claim, standing: no gap_closed claim from exploration numbers, no "better model" language before the pooled accounting at P3.6, no inference-time oracle quantities, sealed partitions untouched, and π remains an audit ratio. The P3.6 confirmation preregistration is drafted only when exploration shows a stable positive gap_closed across seeds, with bands set from measured effects, spending CONFIRM once.

## 9. Plain-language summary

This is the run where corrections are supposed to become better answers. The model trains on three signals at once — match the big teacher's judgment where it is teachable, aim writes at fixable mistakes, and keep everything it already does right — with the training pressure between them contractually balanced so nothing starves in silence again. The safety rails now widen on a published schedule the system has already earned the first step of, and every widening is a measurement. The one question the whole program turns on gets read at every checkpoint: how much of the distance to the 14B does the small model close on real math, code, and science questions. Before any expensive detour, a cheap desk audit decides whether the aiming bottleneck is a too-small funnel or too-little information, and the matching remedy is pre-approved. The final exams stay sealed. Mark signs the executed lock, and the main campaign runs.

---

**Ratification line:** the coding agent delivers 2a–2d with receipts, binds checkpoint hashes, calibration values, χ_max, share-weight calibrations, and the look schedule into the executed lock, and P3.4 training begins on Mark's approval of that document.