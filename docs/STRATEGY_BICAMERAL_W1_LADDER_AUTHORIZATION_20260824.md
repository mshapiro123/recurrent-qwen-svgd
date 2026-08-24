# STRATEGY AUTHORIZATION — W1: The Eleven-Rung Correction-Causality Ladder (X-1), Executable

**Date:** 2026-08-24
**Status:** EXECUTABLE AUTHORIZATION to the coding agent, under the ratified A2′ and the banked W0. Forward-only, score-only. **No training, no optimizer, no sealed reads. Step-2 remains blocked pending X-1 + D-M5.** The W3 desk items (X-6, D-M5) are authorized to start immediately in parallel (CPU-only).
**Authority chain:** Charter `1xeAJYfq2lIvIm76nszshH4xmaeAkiDl2` (SHA `51083fa2…72ac`) → Stage-0 adjudication `1sDoD6HXM63saL2KrWfL9AmD06S0gMOCH` (SHA `056337f3…4d7f`, §9 ratified by Mark) → **W0 result handoff `1M4GHkaro9ioNg6LGS1MF7XA9ha3FrpYi`, 8,838 B, SHA `b6d57d44fb984b1869f2c76afc0054f094a309a2687232b5b32196bdedb964d6` — byte-verified exact; W0 BANKED `PASS`.**

---

## Plain-language summary

W0 is banked exactly as requested: the repaired evaluator reproduces the base model to the last bit, the gates create two genuinely different hemispheres at their design values, and the fifth provenance-class catch of the program (line-ending conversion silently changing artifact hashes in transit) was caught by the agent's own hash assertion — the receipts are trustworthy because the machinery distrusts itself correctly. This document is the execution order for the experiment everything now waits on: inject eleven kinds of candidate "correction" directly into the model's state, one row at a time, through the schedule already proven harmless, and measure which — if any — actually move answers. Phase A races five definitions of a correction against their own scrambled controls; Phase B takes the winner up the granularity ladder (cluster mean, global mean, wrong-cluster mean) and injects the mysterious shared residual directions that triggered our feasibility escalation, settling whether they are signal or noise. Every branch of every outcome is mapped below before any data exists. The dry run prices everything first; the eight-hour cap binds; any surprise returns to strategy rather than being resolved locally.

## 1. W0 adjudication (brief)

**BANKED: `PASS`.** T1 exact identity (9,723,904 logits, 0 differences); T2 gradient contract full pass; operating-point divergence receipt clean (branch correlation 0.7446/0.7446, RMS difference 3.003, stable across seeds — conditioning differentiates, no mechanical collapse); 256-row manifest frozen (SHA `06b2ab04…c7bea`; 216 GSM8K / 32 ARC-C / 5 MBPP / 3 other); sequential evaluator 4.783 ms/row (40.5% slower than the invalid schedule — scientifically required, operationally immaterial); caching 0.00735 A100-hr two-seed. The LF→CRLF incident: superseding the first preflight was the correct call, and the summary-vs-file hash assertion it produced becomes **standing practice for all future receipt archives**. The escalation and Step-2 blocks are unchanged, as the handoff itself correctly states.

## 2. W1 scope and evaluator identity

- **Graph:** the deferred-terminal-write / no-re-entry serving graph (depth-study config-2 lineage), with the sidecar-computed terminal write **replaced by an externally supplied per-row injection vector**. No bicameral split is involved in W1; no re-entry anywhere.
- **Evaluator-provenance tag `EV-LADDER-1`** := this graph + `sequential_shared_middle_v1`-class declared execution schedule + DEV-2 margin estimator (2,048-row panel) + pinned runtime (A100-SXM4-40GB / torch 2.11.0+cu128 / CUDA 12.8 / BF16-SDPA). No hybrid cells; a curve combines only `EV-LADDER-1` cells with identical batch composition.
- **Injection convention (the final-cell deployed-write convention, exactly):** v_inj = γ · (RMS(h_row)/RMS(v)) · v with γ = 0.05, applied at the write interface at the terminal position; the ρ = 0.550893 RMS cap applies (non-binding at γ=0.05). **No amplitude exploration.** If this formula deviates mechanically from the final-cell implementation in any way, stop and surface it — do not adapt silently.
- Both seeds; init endpoint; content-addressed resumable execution; one result handoff (Drive + bytes + SHA).

## 3. Phase A — target-family bake-off (own-row injections)

Rungs, per row, per seed, margin panel on the 2,048-row DEV-2 population (targets computed fresh per row; the 256-row artifacts are not required for A-rungs computed from the model):

| Arm | Target v for row i | Computation |
|---|---|---|
| L0a | loss-gradient at the write interface | one backward per row, frozen model, init |
| L0b | 14B Procrustes late-layer state delta | teacher forward per row + row-13 split-fit map (see fallback below) |
| L0c | margin-gradient (∂(logit_gold − max wrong)/∂ write output) | one backward per row |
| L0d | teacher-forced state delta (gold-forced minus free-run state at the interface) | two student forwards per row |
| L0g | neighbor-consistency (centroid of k=8 nearest solved rows' interface states − own state; fingerprint-spine metric) | one caching forward pass over the panel + kNN |
| L5-x | for EACH family x above: within-family row-shuffled targets | free (permutation; registered seed) |
| L4 | random direction, norm-matched to L0a | free |

**L0b fallback (pre-registered):** if the dry run prices 2,048 14B forwards over budget, L0b runs on the 256-row manifest instead, WITH its own L5-b and L4 controls on that same 256-row population, provenance-tagged as a separate sub-population — never compared across populations.

**Winner rule (fixed now):** the family with the largest pooled DEV-2 margin delta whose bootstrap CI clears zero, in both seeds; tie broken toward L0d. **No family clears its own shuffle and the nulls, both seeds → key `TARGETS-NOT-ANSWER-GRADE`**; Phase B reduces to L6 only; results return to strategy.

## 4. Phase B — granularity and the escalation probe (winner family, plus L6)

| Arm | Target | Question |
|---|---|---|
| L1 | own-cluster mean (k=2 assignment, winner family) | cluster-level causality |
| L2 | global mean (winner family) | global steering |
| L3 | other-cluster mean | specificity control |
| L6 | shared residual directions from the R-S0-A receipts: ±u₁ first; ±u₂, u₃ if budget permits (registered order) | H-signal vs H-noise on the escalating residual |

Keys per the registered branch map: `CLUSTER-CAUSAL` (L1 > L2 > nulls, both seeds) / `GLOBAL-STEER` (L1 ≈ L2 > nulls) / `ROW-LEVEL-ONLY` (L0 positive, L1 ≈ L2 ≈ 0) / `TARGETS-NOT-ANSWER-GRADE` (nothing clears). **L6 positive → H-signal** (deflation re-scoped to protect the subspace; D-M5 sizes the subspace-map estimand); **L6 null → H-noise** (deflation stands; map estimand carries the program). Seed split on any key: escalate, no branch.

**Generative staging:** 461-slice generative cells run only for arms whose pooled margin CI clears zero (both seeds), after the margin look, within the cap.

## 5. Budget, dry run, and stop conditions

Cap **8 A100-hr total** (margin cells + target computation + staged generative cells). **Dry run first**: measured per-arm margin-cell time under the sequential evaluator, plus per-family target-computation cost (L0b teacher forwards explicitly priced), reported before any full arm runs; if the full registered matrix exceeds the cap, the pre-registered trim order is: L6 u₂/u₃ → L0b (to fallback population) → generative staging tightened. Any cost-cap breach, runtime mismatch, seed disagreement, or schedule change returns to strategy. Compute torn down after each session; receipts per the wave rule with the summary-vs-file hash assertion.

## 6. Receipts required

Per arm × seed: pooled DEV-2 margin delta + bootstrap CI; per-battery breakdown; per-row margin tables (preserved into the registered fragility corpus); injection norm statistics; flip tables for any generative cells. Machine key per §3/§4. One byte-sized, SHA'd result handoff.

## 7. Registered predictions on the record (scored at the look)

P-X1 L0d largest positive margin movement, both seeds. P-X2 L1 indistinguishable from L2. P-X3 nulls (L4, all L5-x) at zero, separated from any positive family. P-X4 L6 sits with the nulls. P-X5 L0b < L0d, both seeds. Decisions ride the branch map, not these.

---

*Signature block*

**Strategy:** W0 banked `PASS`; W1 authorized as above; W3 desk items (X-6 residual-structure audit; D-M5 map-estimand power analysis) authorized to start now, CPU-only; tracker r23 cut with this wave.
**Coding agent:** execute §2–§6; dry-run costing gates all cells; surface any mechanical ambiguity in the injection convention before running.
**Mark:** no decision required this wave — the ratified map determines everything; the next decision arrives with the W1 adjudication.
