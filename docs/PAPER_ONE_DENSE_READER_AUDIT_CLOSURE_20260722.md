# Paper One Dense-Reader Audit Closure

**Date:** 2026-07-22  
**Status:** Complete, evaluation-only correction  
**Canonical receipt:** `outputs/stage5/stage5_phase_a_dense_reader_audit_20260722/summary.json`

## 1. Correction boundary

The archived dense continuations were rescored at their first completed response. Direct completions use the leading valid symbol; serialized scratchpads use the first valid `answer:` marker. The prior reader preferred the last `answer:` marker anywhere in the continuation. Because the dense rows did not train an EOS boundary, later untrained continuation could overwrite an already completed answer.

This is a reader correction over hash-locked outputs. No checkpoint, frozen row, prompt, generation, or recurrent score changed.

## 2. Corrected step-4000 depth counts

Each cell is correct of 128.

| Depth | B: direct 0.5B | D: direct 1.5B |
|---:|---:|---:|
| 1 | 128 | 125 |
| 2 | 95 | 107 |
| 3 | 46 | 95 |
| 4 | 29 | 75 |
| 5 | 29 | 48 |
| 6 | 26 | 39 |
| 7 | 21 | 35 |
| 8 | 18 | 38 |
| 9 | 25 | 31 |
| 10 | 19 | 5 |
| 11 | 17 | 16 |
| 12 | 16 | 15 |
| 13 | 15 | 14 |
| 14 | 12 | 13 |
| **Total** | **496/1,792 (27.68%)** | **656/1,792 (36.61%)** |
| **Depths 11-14** | **60/512 (11.72%)** | **58/512 (11.33%)** |

The scratchpad C total is `1,292/1,792 = 72.10%`; its depth-11-to-14 tail is `13/512 = 2.54%`.

## 3. Corrected preregistered A-versus-B result

Paired over all 1,792 frozen rows:

- A-only correct: `1,048`;
- B-only correct: `38`;
- ties: `706`;
- net A: `+1,010`;
- exact paired two-sided `p = 5.72e-257`.

The preregistered count-based gate still passes. Its rule was A over B at at least three consecutive depths, with a one-sided Fisher exact `p < 0.05` at each depth. Corrected passing depths are **2 through 14**, thirteen consecutive depths. Depth 1 is a `128/128` tie and does not pass; the prior wording that A cleared all fourteen depths must not be retained.

| Depth | A | B | One-sided Fisher p | Pass at depth |
|---:|---:|---:|---:|:---:|
| 1 | 128 | 128 | 1.000 | No |
| 2 | 127 | 95 | 2.17e-10 | Yes |
| 3 | 126 | 46 | 9.97e-31 | Yes |
| 4 | 125 | 29 | 5.56e-40 | Yes |
| 5 | 127 | 29 | 4.87e-43 | Yes |
| 6 | 126 | 26 | 1.28e-43 | Yes |
| 7 | 124 | 21 | 1.09e-44 | Yes |
| 8 | 122 | 18 | 1.10e-44 | Yes |
| 9 | 116 | 25 | 3.95e-33 | Yes |
| 10 | 113 | 19 | 5.83e-35 | Yes |
| 11 | 97 | 17 | 2.52e-25 | Yes |
| 12 | 87 | 16 | 1.27e-20 | Yes |
| 13 | 57 | 15 | 2.79e-9 | Yes |
| 14 | 31 | 12 | 0.00117 | Yes |

## 4. Corrected checkpoint-extension analysis

The table compares step 4,000 against step 2,000 on identical rows. `4k only` and `2k only` are paired discordances.

| Arm | Step 2,000 | Step 4,000 | Net | 4k only | 2k only | Paired two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| B, direct 0.5B | 495 | 496 | +1 | 134 | 133 | 1.000 |
| C, scratchpad 0.5B | 1,291 | 1,292 | +1 | 2 | 1 | 1.000 |
| D, direct 1.5B | 643 | 656 | +13 | 174 | 161 | 0.512 |

The old extension values (`B +6`, `C +22`, `D -28`) are superseded. None of the corrected 2,000-to-4,000 changes is statistically resolved.

## 5. Revised Arm D interpretation

Corrected D exceeds corrected B by `160` rows overall (`36.61%` versus `27.68%`). The scale-control direction therefore no longer contradicts the undertraining account. However, undertraining is an interpretation rather than a demonstrated causal result: D gained only 13 net rows from step 2,000 to 4,000 (`p = 0.512`) and remained far below the scratchpad and recurrent systems. The defensible wording is that the larger direct model performed better than the smaller direct model under this recipe, while neither the recipe nor the checkpoint extension established convergence.

## 6. Marker closure

The seven requested audit placeholders are covered by the canonical JSON receipt and this document:

1. B per-depth counts: Section 2.
2. D per-depth counts: Section 2.
3. B depth-11-to-14 tail: `60/512`.
4. D depth-11-to-14 tail: `58/512`.
5. A-versus-B corrected paired result: Section 3.
6. Preregistered per-depth gate verdict: pass over depths 2-14; depth 1 tie.
7. Checkpoint-extension deltas and revised D interpretation: Sections 4-5.

