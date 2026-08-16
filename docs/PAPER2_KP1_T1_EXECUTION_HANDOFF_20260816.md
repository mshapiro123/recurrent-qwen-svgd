# Paper Two KP-1 and Amended T1 Execution Handoff

**Date:** 2026-08-16  
**Status:** Score-only DEV wave complete; KP-1 is indeterminate because its locked target degenerates on generative batteries; T1 state extraction is valid and teacher-fingerprint retrieval remains pending  
**Authority:** `STRATEGY_DIAGNOSTIC_WAVE_ANALYSIS_20260816.md`, Drive `1l8yDmL97eI3a4iTybM0m9yq989xvMBrz`, 12,906 bytes, SHA-256 `29464906d40dab5af4805d2d941370ede382e1da068b0939639fcf88adaa4b62`  
**Receipt root:** `recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_kp1_t1_20260816/`  
**Effective source commit:** `27638fab` on `codex/phase3-opening-build`

## 0. Bottom line

The authorized score-only wave completed without touching CONFIRM or EVAL-E and without constructing an optimizer. The amended T1 extraction produced a clean, interpretable state-geometry result. Recurrent state changes systematically with forced depth inside each checkpoint, but the absolute coordinate system is seed-specific. P3.4 and P3.5 states are essentially identical within a seed, while the mean cross-seed cosine of the 44-cell core fingerprint is only `0.0715`.

KP-1 does **not** yet answer whether the smaller model possesses but fails to read out the missing knowledge. The locked first-token target collapsed all 122 GSM8K rows to one formatting token, all three Tier-1 rows to the same token, and 22 of 25 MBPP rows to one of only two tokens. The best pooled probe scored 62/100, but a simple train-derived battery-aware label-frequency control scores 56/100. Much of the apparent 24-point gain over the registered pooled intercept (38/100) is therefore task-family/format recognition, not answer-knowledge recovery. The MCQ-only slice is mildly suggestive (best 17/54 versus 11/54 for the train-derived battery-majority control), but it was selected from 20 probe surfaces and row-level probe predictions were not retained, so paired support and multiplicity-corrected inference are unavailable.

**Program consequence:** do not use this KP-1 pooled result to choose the Stage 2A memory lead. Repair the estimand first. T1 can be banked as a valid state-extraction receipt, but the fingerprint spine is not complete until teacher fingerprints are added.

## 1. Question and rationale

### KP-1

The intended question was whether rows that a pinned 14B teacher answers correctly and the 0.5B student answers incorrectly contain recoverable answer information inside the student. A positive result would support “present but unread”; a negative linear probe could not prove absence.

The population was the 329-row DEV knowledge gap:

| Battery | Rows |
|---|---:|
| ARC-Challenge | 31 |
| ARC-Easy | 52 |
| GSM8K | 122 |
| MBPP | 25 |
| MMLU | 96 |
| Tier-1 | 3 |
| **Total** | **329** |

The locked split was battery-stratified and row-disjoint: 229 probe-training rows and 100 probe-evaluation rows, seed `20260816`. The target was the first token of the canonical gold-answer suffix. Probes were affine ridge maps from frozen states into the frozen output-embedding target, evaluated against the full vocabulary. No model parameter was optimized.

### Amended T1

The intended question was whether the recurrent sidecar states contain stable fingerprints across depth, ceiling, checkpoint, and seed. The four endpoints were:

| Endpoint | Registered SHA-256 |
|---|---|
| P3.4 seed 0 step 4,000 | `381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7` |
| P3.4 seed 1 step 4,000 | `97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec` |
| P3.5 seed 0 EMA step 4,400 | `a047e2e7b35320376a736492c79d913b8690937da785efa2af002c8f54d26ca6` |
| P3.5 seed 1 EMA step 4,400 | `e36cddb76407c8f853ccb43824c77cf01d15f144780726dd9aec23215467fccb` |

The fixed core schema contains eight prelude cells, eight recurrent cells for each of four loops, four layer cells, and 128 dimensions per cell: 44 × 128 values per row. A separate post-write deployment cell carries the ceiling comparison because the core is ceiling-invariant by construction.

## 2. Locked execution and integrity

The pre-model manifest passed before the base model was loaded:

- DEV panel: 1,024 rows.
- Merged reference table: 10,231 rows, including all panel rows and 9,207 rows outside the panel.
- Knowledge-gap population: exactly 329 rows with the registered per-battery counts.
- Base-reader identity mismatches: zero.
- Prompt-plus-answer tokenization boundary stable: 329/329 rows.
- CONFIRM scored: false.
- EVAL-E scored: false.
- Optimizer constructed: false.
- Optimizer steps: zero.
- Frozen base and sidecar fingerprints unchanged after scoring: true.

The exact model was `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, BF16 with SDPA.

## 3. KP-1 results

### 3.1 Native readouts

The base model is wrong on all 329 rows by construction. The native final-layer logit lens placed the locked target first on 36/329 rows (10.94%), in the top 10 on 179/329 (54.41%), with median rank 4. Earlier layers were substantially weaker. Native recurrent draft readouts ranged from 7.29% to 10.64% top-1 across checkpoints and loops, with the best value at seed 1, loop 4.

These native numbers inherit the target problem below and should not be read as answer recovery on GSM8K or MBPP.

### 3.2 Linear probes

On the 100-row held-out probe split:

- Best pooled linear-probe top-1: 62/100.
- Registered pooled intercept control: 38/100.
- Best substrate-layer probe: 61/100 at layer 6.
- Best recurrent probes: 62/100 at seed 0, loops 2 and 3; the P3.4 and P3.5 values are identical within seed.
- Typical top-10 probe accuracy: 88% to 93%.

Taken alone, 62 versus 38 looks large. It is not a valid knowledge-presence comparison because the labels reveal battery format.

### 3.3 Target-degeneracy audit

| Battery | Rows | Unique locked target tokens | Dominant-token share |
|---|---:|---:|---:|
| ARC-Challenge | 31 | 4 | 35.5% |
| ARC-Easy | 52 | 4 | 28.8% |
| MMLU | 96 | 4 | 29.2% |
| GSM8K | 122 | 1 | 100.0% |
| MBPP | 25 | 2 | 88.0% |
| Tier-1 | 3 | 1 | 100.0% |

For GSM8K, the “gold token” is the common separator/whitespace token `220`, not a digit from the answer. The analogous collapse affects Tier-1, and MBPP is mostly token `707`. A probe can therefore score well by decoding which benchmark generated the row.

A post-hoc but necessary control was computed from the locked split: choose the most common target token within each battery using **probe-training rows only**, then apply it to the probe-evaluation rows, with the pooled train majority as fallback for the one Tier-1 evaluation row whose battery has no training row. This control scores 56/100. The apparent probe advantage is therefore six rows, not 24.

The 54-row MCQ-only evaluation subset removes the formatting collapse:

- Best recurrent probe: 17/54 (31.5%).
- Train-derived battery-majority control: 11/54 (20.4%).

This is suggestive but not confirmatory. The best result was selected from 20 probe surfaces, predictions were not retained row by row, and no multiplicity rule was registered for this post-hoc slice.

### 3.4 KP-1 verdict

**`INDETERMINATE_TARGET_DEGENERATE`.** The run demonstrates that frozen states support task-family and label-format decoding. It does not yet establish that missing answer content is present but unread. It also does not establish absence.

## 4. T1 results

### 4.1 Ordered depth progression inside each seed

Mean cosine between the active-depth core and that checkpoint’s K=4 core:

| Seed | K=1 | K=2 | K=3 | K=4 |
|---|---:|---:|---:|---:|
| 0 | 0.8914 | 0.9418 | 0.9838 | 1.0000 |
| 1 | 0.7275 | 0.8777 | 0.9740 | 1.0000 |

The state trajectory is ordered by recurrent depth in both seeds, with substantially greater early-loop movement in seed 1. This supports the existence of a latent depth trajectory within a trained instance. It does not imply a common coordinate basis across independent runs.

### 4.2 P3.4-to-P3.5 stability within seed

Within either seed, P3.4 and P3.5 core fingerprints have mean cosine `1.0000`; post-write deployment-cell means range from `0.9999972` to `1.0000`. This is consistent with the P3.5 landing leaving the state-construction map effectively unchanged while changing the trained output/write machinery. It should not be described as total parameter identity because the sidecar parameter fingerprints differ.

### 4.3 Cross-seed incompatibility

Across seeds:

- Core 44-cell mean cosine: `0.07150`.
- Core fifth-percentile cosine: `0.04487`.
- Core minimum cosine: `0.02197`.
- Post-write deployment mean cosine: approximately `-0.031`.
- Most negative deployment-row cosine: `-0.1671`.

The two successful seeds use markedly different state coordinates. A raw cross-seed nearest-neighbor or cosine fingerprint is therefore the wrong retrieval instrument. Future teacher/student or seed/seed comparisons need a frozen alignment or a relational, basis-invariant metric.

### 4.4 Ceiling stability

Against the registered 0.05 reference, post-write deployment-cell cosine remains at least `0.9999936` in mean across tested ceilings 0.02, 0.05, 0.08, and 0.11. This shows very small angular movement of this cell over the safe amplitude range. Because cosine ignores norm and the writes are deliberately small, this is not evidence that ceiling has no functional effect.

### 4.5 T1 verdict

**`DEPTH_PROGRESSION_WITH_SEED_SPECIFIC_COORDINATES`.** State extraction is complete. Teacher-fingerprint retrieval is still pending and must be completed before the fingerprint spine becomes a manuscript result or a Stage 2A selector.

## 5. Figure

`docs/figures/paper2_kp1_t1_handoff_20260816.svg` and `.png` contain:

1. The locked target-token dominance audit.
2. The within-checkpoint core-depth trajectories.
3. The four-checkpoint core-cosine matrix.

The figure deliberately does not plot the 62% pooled KP-1 result as a positive result because its estimand is contaminated.

## 6. Operational archaeology

Four implementation issues were caught and repaired without optimizer steps or sealed-partition access:

1. The initial foreground Colab CLI execution timed out during staging. The scientific runner was moved to a PID- and log-backed background process; no contract changed.
2. Drive contained duplicate P3.4 artifact folders. The FUSE mount selected the incomplete duplicate, hiding seed 1. The complete folder was identified by Drive folder ID, the seed-1 endpoint was copied directly, and its registered SHA was verified. Commit `6334c58e` makes the runner reuse a pre-staged P3.4 endpoint only after exact SHA validation.
3. The first evaluator version passed a Stage 0A feature cache to Hugging Face as a model directory. Commit `b3c85200` instead loads the exact locked model ID and revision into a dedicated cache.
4. A descriptive draft-head readout crossed a `torch.inference_mode()` boundary. Commit `27638fab` keeps the entire score-only read inside inference mode. The full targeted suite then passed: 26 tests.

The FUSE write cache retained the final public summary and status after computation. They were downloaded from the completed VM, SHA-verified, uploaded directly through the Drive API, and re-read from Drive. This was receipt transport only; the scientific output was not recomputed or edited.

## 7. Limitations and do-not-claim list

- Do not claim that KP-1 shows knowledge is present but unread.
- Do not claim that KP-1 shows knowledge is absent.
- Do not cite 62% versus 38% without the target-degeneracy and battery-aware-control disclosure.
- Do not promote the post-hoc MCQ-only comparison to a registered result.
- Do not claim a universal or seed-stable latent coordinate system.
- Do not claim that cosine-level ceiling stability means functional invariance.
- Do not claim T1 retrieval is complete; teacher fingerprints are missing.
- Do not use CONFIRM or EVAL-E numbers; neither partition was scored.

## 8. Decisions requested from strategy

### D1. Authorize KP-1R

Recommended. Keep the same 329-row population and split, but repair the target and controls before any Stage 2A architecture choice:

- MCQ: retain the answer-choice token.
- GSM8K and Tier-1: use the first **answer-bearing** token after removing formatting-only tokens, or preferably teacher-forced metrics over the normalized final-answer token sequence.
- MBPP: do not treat a common code prefix as knowledge. Use a content-bearing token/sequence target or a task-specific semantic target.
- Add a train-derived battery-aware label-frequency control.
- Add a label-permutation control.
- Save row-level predictions for every probe surface.
- Predeclare a primary state/loop or a multiplicity correction; do not select the maximum of 20 surfaces after the fact.
- Report pooled and battery-macro results with bootstrap intervals.

A cheap first repair can reuse the 62.97 MB state cache and score the first answer-bearing token on CPU. A stronger sequence-level repair requires teacher-forced model reads and should be separately authorized.

### D2. Complete the teacher-fingerprint side of T1

Recommended. Use the same 1,024 DEV rows and frozen schema. Because raw seed coordinates are incompatible, preregister one of:

- a frozen orthogonal Procrustes/CCA alignment fitted only on an alignment subset and evaluated on disjoint rows; or
- basis-invariant relational fingerprints such as row-to-row distance kernels or centered Gram matrices.

The retrieval target and split must be fixed before teacher states are inspected.

### D3. Stage 2A consequence

Do not let KP-1 choose between knowledge-memory and procedure-memory yet. T1 supports ordered internal computation but says its coordinates are run-specific. The defensible interim framing is:

> The recurrent mechanism develops an ordered depth trajectory, but the current probe does not yet determine whether task failures reflect absent knowledge or failed readout, and independently trained instances do not share a raw latent basis.

## 9. Receipt map

| Artifact | SHA-256 |
|---|---|
| `receipts/summary.json` | `63c7000456ee82193e08d34b78bc44c6eeb11b110b27928792816c39770adbd1` |
| `receipts/status.json` | `a055c862aba704e239eaf0d8dfe8e52333a859792d35115365c242da3396556b` |
| `receipts/pre_model_manifest.json` | `d2ea4dda8527807c75cc0ab8a889c57c4fd25dec7805de87c4bccbb52e63cbe4` |
| `private/chain_manifest.json` | `20bf44f55e5d4351bfc3d4efbcbc054d62105ffbecf624d273785c1d4f2a8403` |
| `private/kp1_gap_rows.jsonl` | `0bd9d64a44cc1192419efcb0428e40e0a7fffcc4e2cac9f21295caa2402e2ccd` |
| `private/t1_state_cache.pt` | `e4f0dfc1334b76ca219daa5cae7bb63acae1515ee547577510b7ca0f39a9cb6b` |
| Local `analysis.json` | `a939549f22d357f900408b76e0c82b3b9bea53c912a4a01160b0dde42715b548` |

## 10. Plain-language summary

We found a real internal depth signal, but not a universal internal language. Within either trained model, each additional recurrent step moves the state smoothly toward the four-step state. The two independent seeds, however, represent that trajectory in almost unrelated directions. That means depth is real, while raw coordinates are not portable.

The knowledge test needs to be rerun with a better target. It accidentally asked the probe to predict a space or code-prefix token on most generative problems. The probe learned that easy formatting distinction, which inflated its pooled score. The honest result is not “knowledge present” or “knowledge absent”; it is “the current measurement cannot decide.” Catching that now prevents the wrong memory architecture from being chosen on a misleading statistic.

The L4 session was terminated after receipt verification. `colab sessions` reported no active sessions.
