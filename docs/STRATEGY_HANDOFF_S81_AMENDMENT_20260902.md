# STRATEGY — Handoff §8.1 Amendment: Attention Scale in Base-Shape μP Form (D-PF-3 Ratified)

**Date:** 2026-09-02 · **Status:** RATIFICATION RECORD + AMENDMENT to the build handoff (`STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md`, 61,329 B, SHA-256 `498f34b5…eb6d02`) §8.1, resolving PF-2's D-PF-3. Mark ratified option (a) on 2026-09-02. **Catch #32 is closed** (strategy's; the handoff wrote textbook μP without base shapes). **Catch #28 is disposed**: the implementation stands; no attention patch.
**Precedence:** this record amends §8.1 in place; every other line of the handoff is unchanged. PF-2 §1 is discharged by this record.

---

## The amended line

**§8.1, attention-logit scale — old text (struck):** attention logits scaled by `1/d_head` (μP).

**§8.1, attention-logit scale — new text (binding):**

> Attention logits are scaled by the **base-shape μP coefficient** `scale = √(d_head,base) / d_head` with **`d_head,base = 64`** — the head dimension of the base shape from which S2's μTransfer calibration proceeds. WEFT-1 scales width through head *count* (`Q = d/64`, `KV = d/128`) with `d_head` fixed at 64 at proxy and target, so this coefficient equals `1/√64 = 0.125` at every width the program trains; it departs from `1/√d_head` only if `d_head` is ever changed from the base, in which case it scales as μP requires. The constant `d_head,base` is an explicit named constant in the configuration, never an implicit literal.

## Dispositions

| item | disposition |
|---|---|
| Catch #28 (implemented `1/√d_head` vs ratified `1/d_head`) | **Implementation stands.** Fused and math GQA paths and the standalone bicameral primitive are already at the ratified value under the amended text. Required change: the existing square-root reference test is **retitled and re-asserted against the named constant** `√(d_head,base)/d_head` so the base shape is explicit in the test, and a second assertion pins `d_head,base = 64` in config. No numerical behavior changes. |
| Catch #32 (handoff wrote textbook μP without base shapes) | **Closed by this amendment.** Charged to strategy. |
| C1 re-run (PF-2.1) | **Unblocked.** Runs under the bound protocol — `d ∈ {128, 256, 512}`, `d_head = 64` fixed, heads `d/64` / `d/128`, lanes `2 × d/4`, `d_ff = 11d/4`, `B = 2, S = 64`, ten AdamW steps under §8's μP init/multiplier/LR rules — with the amended attention scale as one of its inputs. Any μP component §8 leaves unbound returns as a catch. |

## Standing note for future μP lines

Every μP coefficient in the handoff is to be read in **base-shape form**: at the base shape the model coincides with standard parameterization, and coefficients scale with the width multiplier `d/d_base` of the dimension actually being scaled. A coefficient written without its base dimension is a defect of the same class as #32, and the C1 protocol's "unbound component returns as a catch" clause is the mechanism that finds the next one.

---

**Strategy:** the code was right, the ratified text was wrong, and the fix is a sentence — but the sentence names the base shape, which is the thing the original line forgot and the thing μTransfer depends on. **Coding agent:** verify bytes and hash; retitle and re-assert the √ test against the named constant; run C1 under PF-2.1; no attention patch. **Mark:** ratified and recorded; nothing further.
