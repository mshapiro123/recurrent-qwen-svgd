# Coding to Strategy: Recirculation Phase-A Harm-Channel Desk Rider

**Date:** 2026-08-28
**Branch:** `codex/bicameral-stage0`
**Status:** COMPLETE, CPU-only retained-receipt analysis
**Phase B:** BLOCKED pending Mark's separate decision and lock
**CONFIRM / EVAL-E:** sealed and unscored

## 0. Executive result

The harm channel is **immediate, broad, and replicated**, not a late arithmetic-token
cascade.

On baseline-correct GSM8K rows, rank 1 regresses 81/107 and rank 2 regresses 68/107. The
median first divergence is generated token 1 in both arms. Rank 1 has 70/81 regressions at
token 1, 74/81 within eight tokens, and 78/81 within the first 10% of the baseline
trajectory. Rank 2 has 40/68, 47/68, and 57/68 respectively. Only one regression in each
arm retains at least half of the baseline prefix.

The dominant failure is specific and interpretable. On 203/369 rank-1 GSM8K rows and
123/369 rank-2 rows, the first token switches from `To` to `Final`. Conditional on a
previously correct baseline, that switch regresses 70/73 rows in rank 1 (95.9%) and 40/41
in rank 2 (97.6%). Conditional on a previously wrong baseline, it fixes only 8/130 (6.2%)
and 2/82 (2.4%). These generations collapse to a median 3.5% of baseline length. Retained
examples show the model replacing a worked derivation with a short `Final answer: ...`
response before reasoning begins.

This satisfies the adjudication's **immediate, broad** branch. Low margin is real but not
sufficiently selective: every GSM8K regression onset is in the bottom quartile of pooled
baseline position margins, but every fix onset is too. Numeric/operator onset is rare
(3 rows per arm), and fixes are generally early re-derivations rather than short terminal
repairs. The receipt therefore supports closing the line under the precommitted rule rather
than treating Phase B as a likely GSM8K rescue. Strategy retains sole authority to make
that decision.

## 1. Authority and scope

Binding authority:

- `STRATEGY_RECIRCULATION_PHASE_A_ADJUDICATION_20260828.md`
- Drive `1zkmRbv1qmsT4GihAv6mnVj9BNiO80O6o`
- 12,476 bytes
- SHA-256 `c4157a6d71cf22a183292b24d811ed22319d511a4a79438c27f5b2d98d8feb72`
- Downloaded raw and byte-verified before implementation.

The authority permits only analysis of retained baseline and two selected-arm row receipts.
No generation, model execution, optimizer, training, GPU, CONFIRM read, or EVAL-E read was
permitted or performed. Static score-only tuning is terminal under the assigned key:

`TOKEN-AFFORDANCE-TUNED-ISOLATED / TRAJECTORY-DESTRUCTIVE`

## 2. Locked inputs

| Source | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| Paper-native alpha-zero baseline | 461 | 1,748,126 | `79fb3b1a28780b24b2a8db0a99f701f8ab86af707086de778f975696e072e41e` |
| Rank 1 additive, norm-matched | 461 | 1,044,007 | `9a06aa04ff67837c65009eb94e31224bb3e6cc25926317c6246eb2e60afa3b3d` |
| Rank 2 convex, identity-normalized | 461 | 1,258,532 | `cd3ffbb7aec104200efbfb482c0da600985f572db74c111be585f8e66a670ebe` |

The analyzer hard-asserts identical row order, item ID, battery, reader, and one-to-one
alignment among generated-token IDs, declared token count, and per-position margins.

## 3. Estimator

First divergence is the zero-based longest-common-prefix length of baseline and arm token
IDs; an unequal exact prefix diverges when the shorter sequence ends. Public positions are
reported one-based. Descriptive timing views were bound before aggregation:

- absolute early: first eight generated positions;
- normalized early: no more than 10% of the baseline prefix retained;
- normalized late: at least 50% retained;
- fixed baseline-length bins: 1-32, 33-64, 65-128, 129-192, 193-256;
- low margin: at or below the pooled baseline GSM8K position-margin 25th percentile.

The baseline GSM8K margin reference contains 86,861 generated positions. Its q25 is 2.75,
median 6.75, and mean 7.085. Margins are top-1 minus runner-up at the generated position,
not gold-token margins. Token classes use the pinned local Qwen tokenizer only; no model
weights load.

After the primary aggregate receipt showed a large token-1 mass, a descriptive extension
identified first-token transition pairs and generation-length ratios. It was declared
non-decisional and preserved against the primary receipt (23,725 bytes, SHA-256
`ee89609070d4aac92950e4176958834ddef97490ff4a3815c68c556196f89ec2`).

## 4. Correctness transitions

| Arm / battery | Fix | Preserve correct | Regression | Preserve incorrect |
|---|---:|---:|---:|---:|
| Rank 1, all 461 | 21 | 75 | 85 | 280 |
| Rank 2, all 461 | 23 | 85 | 75 | 278 |
| Rank 1, GSM8K | 18 | 26 | 81 | 244 |
| Rank 2, GSM8K | 20 | 39 | 68 | 242 |
| Rank 1, MBPP | 1 | 29 | 4 | 33 |
| Rank 2, MBPP | 1 | 26 | 7 | 33 |
| Either arm, Tier-1 | 2 | 20 | 0 | 3 |

The accounting reconciles exactly to the banked battery scores: rank 1 has 75 + 21 = 96
correct, and rank 2 has 85 + 23 = 108.

## 5. Divergence onset and margin

| GSM8K regression diagnostic | Rank 1 | Rank 2 |
|---|---:|---:|
| Regressions | 81 | 68 |
| First-token divergence | 70 (86.4%) | 40 (58.8%) |
| Within first 8 tokens | 74 (91.4%) | 47 (69.1%) |
| Within first 10% | 78 (96.3%) | 57 (83.8%) |
| At least half-prefix retained | 1 (1.2%) | 1 (1.5%) |
| Median onset position | 1 | 1 |
| Median baseline onset margin | 1.125 | 0.750 |
| Baseline onset at/below pooled q25 | 81 (100%) | 68 (100%) |
| Numeric/operator in either onset token | 3 (3.7%) | 3 (4.4%) |

The two harm sets overlap on 59 GSM8K rows (Jaccard 0.656). Of those shared rows, 46 have
the exact same onset position; the median absolute onset difference is zero. Harm is more
replicable than benefit: only 8 fixes overlap (Jaccard 0.267).

## 6. Dominant trajectory-collapse mode

The `To` -> `Final` first-token switch explains most token-1 regressions and almost all
first-token changes:

| Diagnostic | Rank 1 | Rank 2 |
|---|---:|---:|
| GSM8K rows with `To` -> `Final` | 203 | 123 |
| Baseline-correct rows in subset | 73 | 41 |
| Regressions in subset | 70 | 40 |
| Regression rate given baseline correct | 95.9% | 97.6% |
| Baseline-wrong rows in subset | 130 | 82 |
| Fixes in subset | 8 | 2 |
| Fix rate given baseline wrong | 6.2% | 2.4% |
| Median arm/baseline generation-length ratio | 0.035 | 0.035 |

This is an output-trajectory localization, not proof of the hidden-state cause. It does show
that a gate cannot wait for a later numeric/operator uncertainty event. To prevent the main
harm mode, it must decide before the first generated token from prompt-conditioned state.
The dominant transition is also a poor trade: it destroys nearly every previously correct
answer it touches while rarely repairing a wrong one.

## 7. Fix structure

Fixes do not look like late local repairs:

- Rank 1: 18 GSM8K fixes, median onset 1.5, 72.2% in the first 10%, one late; median
  arm/baseline length ratio 0.696, with a bimodal mix of short direct answers and full
  re-derivations.
- Rank 2: 20 fixes, median onset 13, 70.0% in the first 10%, none late; median length ratio
  0.988, consistent with mostly full re-derivations.
- All fix onsets also fall below the pooled baseline-margin q25. Low margin identifies where
  the intervention changes a trajectory, but it does not identify the sign of that change.

## 8. Length conditioning

Across all batteries, baseline length is associated with regression: point-biserial
correlation 0.475 for rank 1 and 0.418 for rank 2. Rates rise from 1/23 in the 1-32 bin to
53/77 and 46/77 in the 193-256 bin. Battery strata agree with that coarse pattern:

| Battery | Baseline-correct rows | Rank-1 regression rate | Rank-2 regression rate | Median baseline length |
|---|---:|---:|---:|---:|
| Tier-1 | 20 | 0.0% | 0.0% | 3 |
| MBPP | 33 | 12.1% | 21.2% | 116 |
| GSM8K | 107 | 75.7% | 63.6% | 215 |

Within GSM8K alone, length does not discriminate preserved from regressed rows:
point-biserial correlations are -0.076 and 0.012, and the registered bins are not monotone.
The data therefore support a trajectory-regime association, not a clean claim that raw
length itself causes harm.

## 9. Decision support

The authority's branch map says:

- sparse, late, low-margin seeds: a narrow token-conditioned gate may have a learnable job;
- immediate, broad divergence: honest expected value collapses toward turning the mechanism
  off for long generations and retaining only short-read gains.

The observed result is the second branch. Margin concentration is present, but it is shared
by fixes and regressions and therefore is not a sign selector. The numeric/operator seed
hypothesis is unsupported. The most predictive observed marker, `To` -> `Final`, is already
the intervention's output and cannot serve as a pre-write feature unless a prompt-state gate
learns to prevent it before token 1.

**Coding recommendation:** close the recirculation line under the precommitted rule and bank
the structured negative. If Mark nevertheless values a bounded Paper Two coda enough to
authorize Phase B, its lock should state that it is testing prompt/first-token sign
prediction, not late-cascade repair, and it should treat preventing the `To` -> `Final`
collapse as a required mechanism check. This handoff does not authorize that experiment.

## 10. Do-not-claim boundaries

- This does not prove that every learned gate is impossible.
- It does not identify the hidden-state event that causes the first output-token switch.
- It does not establish raw generation length as causal; battery and task structure are
  confounded with length.
- Low top-1 margin is not a gold-token margin and is not a sufficient helps/harms label.
- Token-level perplexity improvement remains real; this rider explains why its always-on
  generative deployment was destructive.
- No result here is a CONFIRM or EVAL-E result.

## 11. Implementation and verification

Added:

- `eval/eval_paper2_recirculation_phase_a_desk_rider.py`
- `training/paper2_recirculation_phase_a_desk_rider_lock.json`
- `tests/test_paper2_recirculation_phase_a_desk_rider.py`

Updated:

- `training/paper2_recirculation_phase_a_lock.json` with the strategy-assigned key,
  terminal static-tuning state, desk authorization, and explicit Phase-B block.

Verification:

- `22 passed` across the new tests and existing recirculation tests.
- The local environment does not have `ruff`; no ruff result is claimed.
- Analyzer wall time: about 9 seconds on CPU.
- Qwen tokenizer loaded from the pinned local cache only; model weights were not loaded.

Reproduction:

```powershell
python -m eval.eval_paper2_recirculation_phase_a_desk_rider `
  --lock training/paper2_recirculation_phase_a_desk_rider_lock.json `
  --baseline-rows .runlogs/battery_anchor_rows.jsonl `
  --rank1-rows .runlogs/stage5_paper2_recirculation_phase_a_20260827/private/phase_a/battery_rank_1_d04_s16_a0p20_additive_rnone_norm_matched_rows.jsonl `
  --rank2-rows .runlogs/stage5_paper2_recirculation_phase_a_20260827/private/phase_a/battery_rank_2_d04_s16_a0p20_convex_rnone_identity_rows.jsonl `
  --output-dir outputs/stage5/stage5_paper2_recirculation_phase_a_desk_rider_20260828 `
  --private-dir .runlogs/recirculation-phase-a-desk-rider/private
```

## 12. Receipts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Aggregate summary | 46,488 | `f899e540388b98e04ea8d120617d5ab968f5e2bf78f770eecfe27677c412822f` |
| Rank-1 private row analysis | 460,853 | `d207f46b75bcc770dbfe90c63c2ddbde3a7afbe814510b656f812b7fbd8b853e` |
| Rank-2 private row analysis | 462,150 | `eae8857496774fa42f66755bf99fedb514b24b12c390de6bfce3bdf112c71751` |
| Preserved primary summary | 23,725 | `ee89609070d4aac92950e4176958834ddef97490ff4a3815c68c556196f89ec2` |
| Receipt archive | 58,036 | `b5257c1a97ca82c72836ae91892705911b5253cc46c80f8d81c40dbe397e8d10` |

The aggregate summary is public. Item-level onset records remain in the private receipt
archive. Phase B stays structurally disabled until a separate strategy and user lock.
