# Phase-2 A2 Calibration Reconciliation and Amendment Draft

Date: 2026-08-05

## Decision surface

- Calibration verification: `two_seed_public_private_receipts_reconciled`.
- Pathology verdict: `clear_for_amendment_review`.
- Optimizer updates: `0`.
- A2 training launched: `false`.
- Amendment status: `draft_unlocked_strategy_review_required`.

## Seed receipts

| Seed | Primary cosine mean | Fraction below -0.5 | Raw spread | A1 spread | Clip p99 x 10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7253 | 0.0000 | 2201.8 | 353253.4 | 2.3620 |
| 1 | 0.7061 | 0.0000 | 1789.4 | 252610.3 | 2.6490 |

Both seed-specific private 51-batch receipts reproduce the public raw norms,
initialized shares, and primary-loss cosine means. No mutation, conflict, or
raw-spread pathology is present.

## Amendment frame for strategy approval

- Keep the seed-specific calibrated static weights as initialization only.
- Audit the matched 51 x 128 training estimator at steps 200, 400, 600, 800, and 1,000.
- Require cumulative KL plus local CE to hold at least 50% of independent trainable-path share.
- Cap each non-primary loss, including final CE and preserve KL, at 25%.
- Keep preserve KL descriptive and enforce preservation through endpoint-quality tripwires.
- Use each seed's observed p99 x 10 value as a catastrophe-only clip tripwire, not a shaper.
- Keep the +2% oracle-headroom gate, matched draft-head-only superiority, and endpoint quality gate unchanged.

The legacy 35/35/10/20 point shares are intentionally incompatible with the
future directional contract at step zero. This is not a calibration pathology:
strategy demoted them to initialization targets only. The amendment must state
explicitly that hard directional audits begin at optimizer step 200.

## Next decision

Strategy must approve or revise this draft. Only the subsequent committed
amendment lock may authorize the two A2 arms and two matched controls.
