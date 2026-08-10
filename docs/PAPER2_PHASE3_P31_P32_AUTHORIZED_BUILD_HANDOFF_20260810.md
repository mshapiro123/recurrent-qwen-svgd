# Paper Two Phase 3.1/3.2 Authorized Build Handoff

Date: 2026-08-10. Branch: `codex/phase3-opening-build`. Status: build complete;
receipt run prepared; P3.3 training remains unauthorized.

## 0. Bottom line

The strategy response was fetched byte-exact from Drive, verified at 5,506
bytes with SHA-256
`925082fc56ffa5fab776aa4dbd44c42440b2e7aa7965f916106236640f8fd37d`,
and added to the governing contract.

Both strategy resolutions are implemented. Agreement-stratum aim targets and
positive gate labels require 14B/32B concurrence. Fourteen-billion-only rows
remain usable for KL distillation, confident-agreement negatives, and
preservation. The P3.2 manifest now reports coverage by label and loss class,
including uncovered flip candidates for the named targeted-32B extension.
There is no invented thinness threshold and no concurrence relaxation.

The sequential-stop simulator now distinguishes false-stop probability under
the no-drop null from detection power under real degradation. This corrects the
opening build's boundary-null approximation. The planning design at 512 rows
and one-sided alpha 0.00005 clears the false-stop target but has very low power
under the provisional noise model. The binding design must therefore come from
the empirical DEV rerun, as strategy required.

## 1. Statistical contract and planning result

The registered stop remains a one-sided paired Student-t upper bound below
-0.03 on two consecutive looks. Simulation uses correlated {-1, 0, +1} paired
outcomes across 20 looks.

At the strategy planning values, 100,000 simulated campaigns with provisional
discordance 0.20 and adjacent-look correlation 0.80 produced:

| Quantity | Result |
|---|---:|
| Familywise false stops under no true drop | 0 / 100,000 |
| Conservative 95% upper probability | 0.00002996 |
| Target | below 0.0001 |
| Detection power at a sustained -3 point difference | 0.00003 |
| Detection power at a sustained -5 point difference | 0.00242 |

The first row is favorable and the power rows are not. A true -3 point effect
lies exactly on the stopping boundary, so high power there is incompatible
with a valid one-sided confidence rule. The -5 point result is the practically
important warning: the planning alpha is strongly over-conservative for
detection under this provisional model.

No threshold or gate was added to power. The calibration CLI now optionally
accepts actual DEV paired-difference trajectories, estimates paired
discordance and adjacent-checkpoint autocorrelation, and labels a calibration
binding only when those empirical inputs are present. Without them it emits a
planning forecast. Battery size and alpha remain design variables; the
familywise 0.0001 target remains fixed.

## 2. Exact source and reader manifest

The source manifest pins:

| Battery | Dataset/config | Revision | Reader role |
|---|---|---|---|
| ARC-Easy | `allenai/ai2_arc`, `ARC-Easy` | `210d026f...` | floor, same MCQ reader |
| ARC-Challenge | `allenai/ai2_arc`, `ARC-Challenge` | `210d026f...` | secondary target, same MCQ reader |
| GSM8K | `openai/gsm8k`, `main` | `740312ad...` | primary, final number after `####` |
| MBPP | `google-research-datasets/mbpp`, `sanitized` | `4bb6404f...` | primary, sandboxed unit tests |
| MMLU | `cais/mmlu`, `all` | `c30699e8...` | floor, fixed seeded slice |
| Tier-1 | local Paper One 64-row canary | `d24f48b2...` | floor, Paper One same reader |

A local score-blind materialization dry run completed without loading a model:

| Battery | Verified train | DEV | CONFIRM |
|---|---:|---:|---:|
| ARC-Challenge | 1,119 | 140 | 159 |
| ARC-Easy | 0 | 270 | 300 |
| GSM8K | 7,473 | 684 | 635 |
| MBPP | 120 | 125 | 132 |
| MMLU fixed slice | 0 | 272 | 240 |
| Tier-1 | 0 | 28 | 36 |
| Total | 8,712 | 1,519 | 1,502 |

The complete ledger hash was
`d83150cd620ce7dbc037df91e47776f4e1e187d038669982526053bce9e4d2ea`.
All train/DEV/CONFIRM document-overlap checks were empty. These are build dry-run
numbers, not yet the durable P3.1 receipt. The Colab target repeats the process
to Drive and records the private membership file and public hashes.

## 3. P3.2 admission and coverage

The cache schema now applies the latest ruling by scope:

- Agreement positive and aim target: teacher/student disagreement, sufficient
  teachability, a present 32B prediction, and 14B/32B concurrence.
- Verified positive: programmatically verified teacher-right/student-wrong;
  it does not need a second teacher because correctness is directly known.
- Confident-agreement negative: 14B-only is admissible.
- KL distillation and preservation: 14B-only is admissible.
- Missing-32B flip candidate: ignored for write supervision and counted as a
  targeted extension candidate.
- Present but conflicting 14B/32B predictions: excluded and counted as a
  cross-scale conflict, not mislabeled as missing coverage.

The preflight confirms all of these paths, keeps agreement rows free of
unverified correctness claims, and preserves exact batched-versus-single oracle
gradient equivalence.

## 4. E1 checkpoint-integrated migration

The integration found and fixed an important lineage detail. E1 resume files
store only the active Option B parameter subset; their frozen flow comes from
the banked A1 endpoint. Migration therefore reconstructs each Phase 2 endpoint
from:

1. seeded Phase 2 initialization;
2. the hash-pinned seed-specific A1 flow state;
3. the hash-pinned seed-specific E1 full-system active state.

Only then is the 1,185,973-parameter Phase 3 state formed, with the scalar gate
retained as the loop bias and all three new position/scratch/control projections
zeroed. The exact banked state RMS cap (`0.5508932316303252`) is pinned in the
migration source contract rather than reconstructed from the rounded module
default. The runnable receipt asserts source file hashes, the E1 active-state
digest, distinct seed lineages, bit-exact Phase 2 versus Phase 3 outputs at
depths zero through four, a fresh destination hash, and absent optimizer state.

## 5. Runnable target

The bootstrap target is `paper2_phase3_p31_p32_receipts`. It is a no-training
Colab job that:

1. materializes and seals the pinned score-blind source ledger;
2. runs the planning false-stop and power simulation, or the empirical version
   when `STAGE5_PHASE3_EMPIRICAL_DIFFERENCES_JSON` is supplied;
3. migrates both E1 full-system seed endpoints from their complete two-source
   lineage;
4. runs the P3.2 schema and batched-gradient preflight;
5. writes public receipts and private rows/checkpoints to Drive.

It does not score CONFIRM, construct an optimizer, or authorize P3.3.

## 6. Verification

- New and focused Phase 3 tests: 24 passed.
- Phase 2 student, matched-alpha, Option B training, and Option B lock
  regressions plus the expanded Phase 3 suite: 70 passed.
- Exact pinned-source dry run: complete, score blind, no model loaded.
- P3.2 preflight: all assertions green.
- Source compilation and `git diff --check`: green.

## 7. Remaining work before P3.3 can lock

1. Run the prepared Colab receipt target so both checkpoint migrations and the
   source ledger become durable Drive receipts.
2. Supply or generate the DEV paired-difference trajectory and rerun empirical
   calibration. The planning power result makes this substantive, not clerical.
3. Generate the real P3.2 cache and coverage arithmetic.
4. Fit the document-disjoint linear-decodability forecast.
5. If the concurrent write stratum is too thin under the future locked
   criterion, authorize a targeted 32B extension over uncovered flip candidates.
6. Draft and lock P3.3 only after these receipts. No Phase 3 training is
   authorized now.
