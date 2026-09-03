# STRATEGY — Engram Gate Ratification: The Literal 64-dim Form Stands, With a Trainable Key-Side Gain

**Date:** 2026-09-03 · **Status:** RATIFICATION RECORD + CORRECTION. Closes the engram-gate question opened by the coding agent's PF-3 follow-up (the live gate was a well-scaled 1024-dim learned query/key form; the agent replaced it with the catch-#37 literal form). Mark ratified on 2026-09-03 after the pros/cons and math below.
**Ledger:** **catch #39 (strategy's, process):** the catch-#37 binding was written unconditionally ("bound: this form") while its rationale was conditional ("if the code has the defect, fix it"); the agent took the unconditional reading, correctly under precedence, and replaced a working gate. **Standing drafting rule D-COND:** a binding that exists to fix a defect is written *"if and only if the defect is present,"* and names the defect's test. **Also corrected, against myself:** my chat claim that the row-as-key form "couples key and value roles" and loses gate precision was wrong — see §2.

---

## 1. The three forms, priced

| form | gate | params at the gate | MACs/token | rank of the bilinear form `hᵀ M e` |
|---|---|---|---|---|
| **A — literal 64-dim (as built now)** | `σ(⟨RMSNorm(W_Q h), RMSNorm(e)⟩ / √64)`, `W_Q: d → 64` | 65.5 K | 65.5 K (~0.26 % of a block) | **64** |
| **C — decoupled 64-dim** | A plus a learned key map `W_K: 64 → 64` | 69.6 K | 69.6 K | **64** |
| **B — live 1024-dim (before the change)** | `σ(⟨RMSNorm(W_q h), RMSNorm(W_k e)⟩ / √1024)`, both into `R^d` | 1.11 M | 1.11 M (~4.4 % of a block) | **64** |

All three are well-scaled at init (logit std 0.996 / 0.998 / 0.999, verified at 20,000 draws).

## 2. The math that decides it

**Expressivity.** Every one of the three gates is a bilinear form `hᵀ M e` with `M = W_qᵀ W_k` (or `W_Qᵀ` for A) of shape `1024 × 64`, followed by normalizations. Because the engram row `e` is 64-dimensional, **`rank(M) ≤ 64` for all three** — verified numerically at exactly 64 for A, B and C. The 1024-dim form buys no expressivity; it only over-parameterizes the same rank-64 form 16× more expensively.

**The "coupling" claim, retracted.** I argued in chat that with the row itself as the key, value-carrying directions of `e` contaminate the gate. They do not: the gate's *query* `W_Q h` already selects which directions of `e` it reads — directions the query does not touch contribute nothing to the inner product. Verified: with half the row's energy in "value-only" dimensions and a query confined to the other half, the gate logit correlates with the intended signal at **0.996** under form A and 1.000 under form C. The only difference is the **normalization denominator**: RMSNorm(`e`) normalizes the whole row, so the effective temperature of the gate scales with the fraction of row energy the query reads. That is a scalar effect, and a **trainable RMSNorm gain** on the key side absorbs it exactly.

**Therefore:** A ≡ C ≡ B in expressivity; A is cheapest; A with a trainable key-side gain has no temperature disadvantage. There was never a capacity reason to prefer B, and the reason I gave for C does not survive the arithmetic.

## 3. Ratified binding

> **EG-1.** The engram gate is **form A as built**: `g_t = σ( ⟨ RMSNorm_γq(W_Q h¹_t), RMSNorm_γk(e_t) ⟩ / √d_m )`, `d_m = 64`, `W_Q : d → d_m` (hidden-class μP, as the agent classified), **with trainable RMSNorm gains `γ_q, γ_k` on both sides** (vector class, LR `η_base`, no decay, init 1). Value path unchanged: `h¹_t ← h¹_t + γ_m · cap(W_V (g_t e_t))`. **T2 liveness re-runs on this form** (nonzero gradient at step 1 on `W_Q`, `γ_q`, `γ_k`, the tables, `W_V`, `γ_m`). Forms B and C are **recorded as alternatives, not registered arms** — the engram sweep's gate-selectivity diagnostic is the only thing that would reopen them. Handoff §5.11's formula is amended to EG-1 in place.

**Process:** the agent's change was admissible and transparently reported; nothing is charged to it. The lesson is mine (D-COND).

---

**Strategy:** the right answer was the cheapest one, and the argument I made for spending 4 K more parameters was wrong in a way one numerical check exposed. Recorded as such. **Coding agent:** add the two RMSNorm gains, re-run T2 on the engram, receipt the gate form as `EG-1`. **Mark:** recorded; nothing further.
