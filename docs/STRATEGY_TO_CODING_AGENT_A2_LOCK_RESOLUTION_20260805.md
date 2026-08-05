# Strategy Handoff to Coding Agent — A2 Calibration Banked, Lock Questions Resolved, Four-Run Matrix Authorized

Date: 2026-08-05. Responds to: STRATEGY_HANDOFF_A2_CALIBRATION_20260805.md and a2_amendment_draft.json (repo commit `d5c9bbbc`; repo byte-lock governs). Governing: A1-bank/A2-contract handoff (Drive `1CCIZqKgIvaveFit8IEOzcXfEcf-4YYWZ`). Both lock questions are resolved below; the amended contract may lock and the four-run matrix may build.

## 1. Calibration banked

All receipts accepted: private 51-batch receipts reproducing public statistics and hashes, zero updates, zero mutation, no conflict or raw-spread pathology. Two findings are banked with emphasis. First, **the primary losses are structurally aligned** — cumulative-KL/local-CE conflict cosines of +0.7253 and +0.7061 with no batch below −0.5. The stop-and-report trigger did not fire because there is nothing to report: the two acceptance-facing objectives pull in nearly the same direction, which was never guaranteed and materially raises the prior that A2 is trainable. Second, **the A2 graph is better conditioned than A1's** — raw-norm spreads of 2,202 and 1,789 against A1's five orders of magnitude, with correspondingly moderate clip candidates (2.362, 2.649). The large scalar weights are accepted as the arithmetic of share-matching across three orders of magnitude, per the A1 precedent. The step-0 violation of the directional contract by the legacy 35/35/10/20 initialization is ratified as intentional: the contract governs training, hard auditing begins at step 200, and the step-0 shares are the calibration record, not a contract event.

## 2. Lock question 1 — miss behavior: the recommendation is adopted, with a severity tier added

Warning on one miss and stop on two consecutive audits is right for *marginal* misses — a step-200 wobble should not execute a healthy trajectory. But a contract whose only failure response is "wait 200 more steps" cannot distinguish a wobble from an inversion, so the lock carries two tiers:

- **Marginal miss** — primary share in [40, 50) percent, or a non-primary in (25, 35] percent: warning with per-batch distribution (not just the mean) in the receipt; **two consecutive** marginal misses at the same bound → stop with receipts.
- **Gross miss** — primary below 40 percent or any non-primary above 35 percent: **immediate stop.** A ten-point breach is not drift, it is the old failure mode arriving; letting it train 200 further steps to confirm would repeat the exact mistake the contract exists to prevent.

Both tiers evaluate on the matched 51 × 128 training estimator only, per the standing estimator rule.

## 3. Lock question 2 — control objective: the recommendation is adopted, and the asymmetry is noted as conservative

The draft-head-only controls train on cumulative KL + local CE with the seed-specific calibrated draft/control weights; final CE and preservation are evaluation metrics only. This is correct — those losses are bridge-specific and copying them into a no-writeback graph would be fitting ghosts. One property is recorded so the eventual comparison is read fairly: the control gets a *purer* acceptance objective than the full system, which spends part of its gradient budget on bridge-path auxiliaries — so the asymmetry biases the superiority test **against** the full system. If the full system beats the control anyway, the claim is conservative; if it loses, the loss is not attributable to auxiliary drag alone without checking the realized shares. Controls run under identical budget, steps, seeds, rows, scorer, and eval cadence; the endpoint-quality tripwire applies to all four runs (trivially satisfied by construction in the controls, whose executed path is unchanged — asserted, not assumed).

## 4. Authorization and sequencing

Lock the amended contract with sections 2–3 folded in (the commit is the lock) → build the four-run matrix (seed 0 full A2, seed 0 control, seed 1 full A2, seed 1 control) → A100 → evaluate against the unchanged gates: recomputed oracle headroom ≥ +2 percent, full-system superiority over the matched control, endpoint quality retained, plus the registered probe-KL/probe-top-1/acceptance correlation table and the trained-module monotonicity analogue. `budget-limited` vocabulary and the single extension allocation carry over. No further CPU or GPU work before the lock, as stated. **V1d: the receipt exists and remains undelivered — attach it to the A2 results handoff so the E1 blocker clears in the same round-trip.**

## 5. Plain-language summary

The free measurement pass came back with the best possible shape: the two training signals that matter most for speed are pulling in nearly the same direction, and the next phase's gradients are far better behaved than the last phase's were. The two open rule questions are settled the same way the whole recent stretch has taught us: a small drift gets a warning and a second chance, a large breach stops immediately, and the simple rival system trains on the purest version of the speed objective — which makes any win against it a win we can defend. The four training runs are cleared to build. When they land, we get the answer the program has been walking toward: whether the state the module just learned to build can actually be turned into faster, verified output.
