# Paper Two Phase-2 V1d Capped-Radius Handoff

Date: 2026-08-04

## Purpose

V1d tested whether the p99 hidden-state RMS cap selected after the V1b tail audit preserves the useful reach of the larger `c = 0.15` writeback radius while restoring the preregistered preservation criterion. It was a DEV-C-only, read-only causal diagnostic. It did not train a controller and did not touch a frozen confirmation partition.

Canonical receipt: `outputs/stage5/stage5_paper2_phase2_prewindow_20260731/v1d/summary.json` (SHA-256 `b8ec5e81649d7a7917d98a0f988cd39c64be16ea51a34b150b02ef07df6d86ca`).

## Design

- Checkpoint SHA-256: `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`.
- Same sampled position keys as V1c, seed `20260731`.
- Cohorts: 2,000 oracle-help positions and 2,000 preserve controls.
- Oracle-help strata: 713 code, 1,287 general.
- Preserve strata: 1,084 code, 916 general.
- Radius: `gamma * c * min(RMS(h0), cap) * sqrt(d) / (1-rho)`.
- Constants: `gamma = 0.05`, `c = 0.15`, `rho = 0.8`, state-RMS cap `0.5508932316303252`.
- Intervention: exact local wrong-token-versus-teacher-token margin-gradient direction.
- Primary reach metrics: first-order pair crossing, realized pair crossing, and realized teacher-token top-1 flip.
- Safety metrics: preserve retention and collateral prediction changes at all other scored positions.

## Results

### Reach

| Metric | Count | Rate |
|---|---:|---:|
| First-order predicted pair crossings | 1,297 / 2,000 | 64.85% |
| Realized pair crossings | 1,290 / 2,000 | 64.50% |
| Realized teacher-token top-1 flips | 1,209 / 2,000 | 60.45% |

The realized pair-crossing rate was 0.35 percentage points below the first-order prediction. The teacher-token top-1 rate was 4.40 points below the first-order pair-crossing estimate, preserving the distinction between crossing one competing-token margin and becoming global top-1.

### Preservation and collateral

| Metric | Result |
|---|---:|
| Preserve controls retained | 1,997 / 2,000 (99.85%) |
| Wilson 95% lower bound | 99.56% |
| Registered diagnostic preservation reading | PASS |
| Oracle-help collateral hurts | 155 / 930,625 (0.0167%) |
| Oracle-help collateral helps | 207 / 930,625 (0.0222%) |
| Net collateral change | +52 helps |
| Rows with any collateral hurt | 146 / 2,000 (7.3%) |
| Causal-prefix prediction changes | 0 |

The preservation rule required a point estimate of at least 99.7% and a Wilson 95% lower bound of at least 99.0%. Both conditions passed.

### Cap behavior

- Twenty-three of 2,000 oracle-help positions had their raw state RMS capped.
- Effective state RMS maxed at exactly `0.5508932316303252`.
- Radius median was `0.45719`; p95 was `0.55378`; max was `0.61838`.
- Neutral-versus-registered numerical-kernel prediction changes were 14 of 930,625 positions and did not change any intervention target.

## Interpretation

The p99 RMS cap achieved its intended tradeoff on this diagnostic sample. Relative to uncapped V1c at the same `c = 0.15`, reach was nearly unchanged while the registered preservation criterion passed. This supports using the capped radius as the bounded writeback envelope for the first trained window. It does not show that a deployable gate can identify the useful directions or that a learned module will realize the oracle intervention.

The constants file `training/paper2_phase2_dc2_constants.json` records the confirmed cap and source receipt. Its current SHA is part of every later launcher receipt.

## Limits and do-not-claim boundaries

- V1d uses an oracle teacher-token gradient direction, not an inference-time controller.
- Pair crossing does not guarantee global teacher-token top-1.
- Local finite perturbations do not prove global reachability.
- DEV-C results are design evidence, not frozen-slice confirmation.
- The receipt bounds the bridge-writeback path only, not the direct residual-draft logit path.

## Decision consequence

V1d is banked as a passed pre-window safety diagnostic. The constants `c = 0.15` and `state_rms_cap = 0.5508932316303252` are no longer provisional. V1d itself no longer blocks E1; canonicalizer arbitration, the matched-pilot protocol, matched pilots, alpha selection, and the resource note remain.

