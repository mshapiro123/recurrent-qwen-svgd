# P3.3 Token-Retention Preflight Handoff

Date: 2026-08-11. Status: complete. Training authorization is now live under
the ratified P3.3 lock plus errata e1 and e2. No optimizer was constructed, no
training step ran, no task-level capability score was computed, and CONFIRM and
EVAL-E remain untouched.

## 1. Purpose and governing correction

Erratum e2 replaced the unavailable task-level guardrail estimator with the
estimator P3.3 actually possesses: the fraction of held-out positions where the
augmented model's top-1 token matches the frozen base model's top-1 token. The
old task-scale thresholds are void for P3.3. This run had three jobs before any
optimizer could exist: freeze the token-retention panel, measure both migrated
seeds at step zero under the operating clamp, and calibrate the two sequential
rules for exactly twenty looks.

## 2. Frozen panel and data integrity

- The panel contains `1,024` confident-agreement positions, exactly `256` at
  each horizon one through four.
- It is selected after the training-negative cohort and the negative audit in
  the same confidence ranking, and is disjoint from training, the `4,096`-row
  positive audit, and the `12,288`-row negative audit.
- The lowest retained confidence is `0.9904632568`.
- Canonical panel SHA-256: `fd5d5c0b6733230a710a3bd439f14866424e25b58434fdf7dfb23308a41bb5b0`.
- File SHA-256: `03167599552601caca753ae67233c569283757456065cade201c0312814a7418`.
- The broader staged population remains `34,521` positives and `103,563`
  negatives, with the realized negative confidence cut `0.9937900901`.

Every cohort-disjointness, hash, exclusion, optimizer-absence, and no-task-score
assertion passed.

## 3. Exact step-zero read

Both migrated seed lineages retained every panel token under the deployed
`gamma = 0.02` clamp:

| Seed | Retained | Total | Retention |
|---:|---:|---:|---:|
| 0 | 1,024 | 1,024 | 100.0% |
| 1 | 1,024 | 1,024 | 100.0% |

The frozen-base reader was reproduced exactly. The old and new population LM
heads were bit-exact copies of the same Qwen2.5-0.5B revision.

As required by erratum e1, the operating clamp binding was reported rather than
treated as a failure. It binds the first loop at initialization in both seeds:
the migrated first-loop gate is `0.08403` for seed 0 and `0.08849` for seed 1,
while loops two through four start at `0.01799` and remain below the ceiling.

## 4. Calibrated sequential rules

The calibration used `100,000` simulated campaigns, exactly twenty looks, the
banked adjacent-look autocorrelation `0.7387033`, and conservative discordance
`0.00640954`. Because exact step-zero nonretention was zero in both seeds, the
noise rule selected the larger prior token-trajectory discordance rather than
pretending the variance was zero.

### Tier S stop

- Trigger: retention is at least `0.006` below its step-zero value at two
  consecutive looks under the registered one-sided statistic.
- Estimated familywise false-stop probability: `0.00003`; conservative 95%
  upper probability `0.0000775`, below the `0.0001` target.
- Calibrated catastrophic drop: `0.0115`.
- Power at that drop: `0.99408`, above the `0.99` requirement.
- Descriptive power at sustained drops of 0.5, 1.0, and 2.0 points: `0.0801`,
  `0.9392`, and `1.0000`.

### Tier W warning

- Trigger: retention is at least `0.001` below initialization at two
  consecutive looks under its stricter one-sided statistic.
- Estimated null-warning probability: `0.00424`; conservative 95% upper
  probability `0.004594`, below the `0.005` ceiling.
- Calibrated warning-scale drop with at least 99% power: `0.009`.
- Descriptive power at sustained drops of 0.5, 1.0, and 2.0 points: `0.54143`,
  `0.99950`, and `1.0000`.

Tier W is a warning, not a stop. Its one-point-on-this-panel sensitivity is
intentional and should not be promoted into a hard training shaper.

## 5. Interpretation

The result is a clean infrastructure pass. The migrated sidecar is neutral on
the exact token-retention estimand at initialization, and the sequential rules
now have measured false-alarm and detection behavior on that estimator. This
does not establish aim capture, useful deployment, general capability
preservation, or task-level performance. It only removes the last blocker to
the registered P3.3 causal falsifier.

The first-loop clamp binding is a real watch item. It means P3.3 begins from a
deliberately smaller deployed write than the migrated Phase-2 endpoint on that
loop. This is already disclosed by e1 and is not grounds to alter the locked
recipe after results arrive.

## 6. Artifacts

Repo receipts:

- `outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/summary.json`
- `outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/p33_retention_step0_summary.json`
- `outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/p33_retention_guardrail_recalibration.json`
- `outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/status.json`

Drive-readable public receipts:

- Summary: `1d_lO-XvqPKhNKQa7Rh1KSY8SDUWClaGn`.
- Guardrail calibration: `1eYQN4VNPoxbj6nOKI4K7pPcfa5hgSLPz`.
- Step-zero summary: `12akpEq-cklhrZrttB-BNBrRmNstGIndq`.
- Status: `1KDnJ8XR3OaFJ2DVDqdsvF-69V8234Eef`.

The complete private bundle is split only to satisfy the connector's 100 MB
limit. Concatenate part 01 then part 02 to reconstruct the zip. Full-zip
SHA-256: `b3272a82c98c340a04f8f52a4039e0baf2fb5784340e78487035a5affa348138`.

- Part 01: Drive `1xH4tgTJOx5KAjGKyZAoZIv7cxnC8rdUt`, 78,643,200 bytes,
  SHA-256 `c783a473b0647d6d4902e08de6ca560488d9a3058c36f5b8b8ee978fcd3c4068`.
- Part 02: Drive `1FvM-l3bCHsVMTHD1uRv5LQ3ZWnElUA0o`, 37,090,846 bytes,
  SHA-256 `7382e1076542978a8160dc98ea7e2c44150f3c88999aa91299cfac53d6242991`.

## 7. Next step

Run the locked two-seed, 1,000-update P3.3 aimed-writeback falsifier with one
registered look every 50 updates. Report `pi_dir`, `pi_dep`, gate recall and
precision, collateral `chi`, the token-retention trajectory, all tripwire and
warning events, and the Tier-1 observatory interventions. P3.3 must continue to
make no task-level capability claim; that inference graph remains a P3.4 lock
prerequisite.
