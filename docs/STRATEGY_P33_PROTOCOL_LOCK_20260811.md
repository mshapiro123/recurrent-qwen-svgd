# Strategy — P3.3 Protocol Lock: The Aimed-Writeback Falsifier (Awaiting Ratification)

Date: 2026-08-11. Drafted against the P3.1/P3.2/P3.3-prep report-back (commits `68b7847d`, `b189cb75` on `codex/phase3-opening-build`). Governing chain: charter r2 → amendment t1 (ratified) → reasoning-scope addendum → build response → policy resolutions r2 → P31/P33-prep handoff. **This lock authorizes P3.3 training upon Mark's ratification and not before.**

## 0. Preamble: the three open items, resolved

- **Panel choice: the 1,024-row panel is adopted.** Its operating characteristics are the ones a real guardrail needs — Tier-W null-warning rate 0.441%, 91.5% power against a true 5-point decline, and a catastrophic threshold of 8.5 points — and the eval cost is a paired scoring pass per checkpoint window, small against an A100 training session. The smaller panels' cheapness is not worth detection blindness on the one cliff that protects the claim.
- **Population minimum: strategy sets none above what is staged.** 34,521 training positives against 1.19M trainable parameters is ample for this pilot. The 32B extension pass stays unused, available to P3.5 as priced.
- **The missing decodability-forecast receipt blocks training, not drafting.** The forecast's numbers are in the session record (holdout cosine 0.070–0.076; ridge-extended loop-4 cosine 0.0952, CI 0.0842–0.1077 seed 0, and 0.0874, CI 0.0792–0.0993 seed 1, ridge 1e5), and the L4 forecast receipt was reported landed before shutdown. Pre-run assertion A5 below requires the receipt located and banked repo-side with hashes — or, if it cannot be located, regenerated from the cached probe artifacts on CPU. Session memory is not a receipt. Training does not start until the file exists in the ledger.

## 1. Purpose and readings

P3.3 is the phase's falsifier: does a trained bridge capture a usable fraction of oracle aim? Primary reading on **π_dir** (forced-open direction capture): **π_dir ≥ 0.25** — the ceilings are a roadmap, proceed at full speed to P3.4. **π_dir < 0.05 after the iteration budget** (one preregistered iteration on features/capacity) — the approach has found its boundary and strategy writes the boundary memo with the observatory's failure-signature differential attached. Between — run the single iteration, re-read. Calibrated prior on record: linear decodability reads at roughly three times chance, so a modest π_dir is expected and the readings above are unchanged by it. These are exploration readings, not gates.

## 2. Initialization and pre-run assertions

Initialize from the **E1-confirmation checkpoints, both seeds**, hash-pinned from the checkpoint ledger. Migration per t1 C2: trained scalar gate into per-loop bias, position-dependent weights and control projection zero-initialized. Pre-run assertions, all mandatory, all receipted:

- **A1** — migrated model reproduces the Phase 2 checkpoint's writeback on the standard audit batch (scalar-reference equivalence, as built).
- **A2** — zero-loop identity bit-exact.
- **A3** — label artifacts match this lock's counts and hashes: 34,521 positives (T ≥ 0.70, strict 14B/32B concurrence), 103,563 negatives (rank-based, realized confidence cut as frozen in the staging receipt), 4,096-row audit slice untouched and hash-matched, position zero in no class.
- **A4** — instrumentation non-perturbation check (RNG streams, precision, kernels unchanged with telemetry enabled).
- **A5** — the linear-decodability forecast receipt banked repo-side (locate or regenerate), per §0.

## 3. Trainable set, losses, and the write bound

Trainable: **bridge, per-position gate head, control state only** (1,185,973-parameter configuration; flow and draft head frozen). Loss set: `L_aim = 1 − cos(u_p, d*(p))` on positives (pre-gate direction, magnitude untouched); `L_gate` = BCE on positives/negatives with inverse-class weights at the realized 3:1 ratio, ignored class excluded; preservation KL on confident-agreement positions at fixed **λ_p = 1.0**. Gate ceiling fixed at **γ = 0.02** for the whole pilot — the annealing controller is P3.4 machinery and does not run here; π_dir is measured forced-open, so the closed ceiling cannot mask aim learnability. **The write bound is the bridge forward equation as implemented in source, reproduced in the run config verbatim with symbol bindings (c = 0.15, p99 = 0.550893, ρ = 0.95, γ = 0.02) per t1 B3.** Directional share contract, calibrated at step 0 and enforced as inequalities on the training estimator with a trailing window: primary losses (aim + gate) ≥ 50% of post-clip gradient share, preservation ≤ 25%, two-tier warn → stop, estimator named in the run config beside the thresholds.

## 4. Guardrails (rule inventory for this run)

| Rule | Class | Grounding |
|---|---|---|
| Non-finite loss/state | stop | absolute |
| Base/frozen-hash mutation; CONFIRM or EVAL-E contact | stop | absolute |
| Tier S: paired one-sided UCB(aug − base) < −8.5 pts, 1,024-row panel, 2 consecutive looks | stop | calibrated (receipt: ≥ 99% power at −8.5, familywise false-stop ≤ 1e-4) |
| Gradient explosion (10× trailing-100 median, 3 consecutive) | stop | trajectory |
| Directional share contract (≥ 50% / ≤ 25%) | warn → stop | matched estimator, trailing window |
| Tier W: paired 90% UCB < −3 pts, 2 consecutive | observe + flag for strategy review (no rungs exist in P3.3, so no demotion consequence) | calibrated (0.441% null-warning) |
| Everything else (π, χ, all observatory quantities) | telemetry | — |

Look schedule matches the calibration receipt's assumed cadence; deviations require recalibration before use.

## 5. Measurement

On the 4,096-row audit slice, both seeds, per t1 B1/B2 with both oracle accountings: **π_dir** (gate forced open, magnitude at the standard bound, oracle denominator under identical conditions), **π_dep** (realized gate and magnitude, matched denominator), gate recall and precision against the tri-state labels at γ = 0.02, all with prompt-level bootstrap CIs, reported per seed and pooled, and stratified by teachability decile. Collateral χ on confident-agreement audit rows. Tier-1 observatory telemetry throughout (r^B, gradient-dot-write, tortuosity, turning angle, fixed-point residual, effective rank, participation ratio, per loop). End-of-run A_state intervention battery on the audit slice: zero, norm-matched random, cross-example permutation, stale state, bridge bypass — paired from cached pre-intervention states, ratio unclipped, numerator and denominator reported separately. Every run reported as a curve (20-point pattern), never a verdict.

## 6. Budget, deliverables, and do-not-claim

One A100 session per seed, resumable, sessions released promptly per the banked ops practice. Report-back: π_dir, π_dep, gate statistics with CIs; decile stratification; χ; telemetry curves; A_state table; the failure-signature reading if π_dir is low; rule-inventory outcomes (looks taken, warnings, none-fired); one creativity slot. Do-not-claim, standing: no gap_closed claims from this run, π is an audit ratio and not a deployment property, no inference-time oracle use, no CONFIRM or EVAL-E contact, and no "better model" language anywhere in exploration reporting.

## 7. Plain-language summary

This is the run the whole phase was built to reach. The model's write channel gets trained, for the first time, to aim — using the cached perfect-aim directions as its teacher — while everything else stays frozen. We then measure, on 4,096 examples it never trained on, what fraction of perfect aim it learned: measured once with the gate held open to test pure aim, once as deployed. A quarter or better and the campaign is funded. Under five percent after one revision and we write up the boundary honestly, with the diagnostics to say why. The alarms are finally real: the disaster stop is certified to catch an 8.5-point collapse with near-certainty while crying wolf less than once in ten thousand campaigns, and the smaller-damage watch flags us without stopping anything. One paperwork item blocks the start: the probe forecast that told us what to expect must be filed as a proper receipt, not a memory. Mark signs this lock, and the experiment runs.

---

**Ratification line:** P3.3 training is authorized upon Mark's approval of this document. Strategy will bind the checkpoint hashes, realized confidence cut, calibration receipt IDs, and look schedule into the ratified copy from the coding agent's ledger at execution time.