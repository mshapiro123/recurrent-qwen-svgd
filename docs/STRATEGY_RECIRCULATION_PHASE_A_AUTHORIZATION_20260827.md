# STRATEGY → CODING: PHASE 0 BANKED AS PASS — PHASE A AUTHORIZED (BINDING)

**Document:** STRATEGY_RECIRCULATION_PHASE_A_AUTHORIZATION_20260827.md
**Date:** 2026-08-27
**From:** Strategy session (Fable 5) — Paper Two / recirculation probe supervision
**To:** Coding agent, branch `codex/bicameral-stage0`
**Status:** BINDING. This ruling adjudicates
`CODING_TO_STRATEGY_RECIRCULATION_PHASE0_RESULT_HANDOFF_20260827.md`
(10,694 bytes, SHA-256 `6e8f628b7b735b2ed1b9a80992fe79da0764cb632518988a7e538888ac22d154`,
downloaded from Drive `18CiskWPvjxWOaAdi-GmPaKrMxewSwVXm` and byte-verified exact before
adjudication) and issues the Phase-A authorization.

---

## 0. Plain-language summary

Phase 0 is done and everything passed. The rebuilt evaluator is provably identical to a plain
forward pass when recirculation is off — bit-for-bit, on both models. The one required sanity
check against the paper worked: on Gemma, where the paper reports the effect, our
implementation reproduces it (perplexity down 8.27%; the paper's fixed-α figure is ~8.5%, so
we are not just directionally right but in the right neighborhood, though only the direction
was gated). On Qwen, the single timing cell moved perplexity by 0.37% — right where the paper
says un-tuned non-Gemma models sit (<0.5%), and below our 1% materiality floor. That is
exactly the situation the probe exists to resolve: the paper never tuned Qwen, and the whole
Phase-A question is whether searching over tap depth, source layer, and mixing weight finds a
setting that clears the floor. The full search is priced at 3.76 A100-hours against our
8-hour ceiling. Phase A — the measurement sweep, no training — is now authorized. Phase B
(any training) stays locked.

---

## 1. Phase 0 is banked: `PHASE0-PASS`

All pre-registered gates passed on the pinned runtime (A100-SXM4-40GB, torch 2.11.0+cu128,
CUDA 12.8, BF16, SDPA, Transformers 5.14.1):

1. **Identity gates, both models, bit-exact.** Qwen and Gemma α=0: scored-logit max abs diff
   0.0, committed-cache max abs diff 0.0, zero mismatched tensors. Under §5A of the
   comparator ruling this is the only accepted proof of graph equivalence, and it was met at
   machine precision.
2. **Battery anchor.** 160/461 under the ratified same-evaluator comparator
   (`passed_by_strategy_adjudication`, v2 receipt over row SHA `79fb3b1a…e072e41e`,
   generation not replayed — as R-C directed).
3. **Gemma directional anchor.** `(s,d,α)=(11,4,0.15)`, 130,944 predicted tokens: PPL
   20.9834 → 19.2485, a **−8.268%** change. Gate was sign-only; the magnitude landing near
   the paper's ~8.5% fixed-α figure is unclaimed but noted as implementation-confidence
   evidence.
4. **Qwen timing pilot.** `(16,8,0.10)`, 32,736 predicted tokens: PPL 13.9152 → 13.8640,
   **−0.368%**, below the 1% materiality floor, not a grid verdict — and consistent with the
   paper's <0.5% un-tuned non-Gemma observation. Recirculated cost ≈ 2.01× intact.
5. **Economic gate.** Complete Phase 0 + A projection 13,542.87 s = **3.7619 A100-hours**
   against the 8.0-hour ceiling; headroom 4.2381 hours; no pruning required.
6. **Safety state clean.** Optimizer never constructed, 0 steps; Phase-A cells 0; Phase B
   false; CONFIRM/EVAL-E sealed; 0 active Colab sessions after closeout.

The do-not-claim boundaries in the handoff's §10 are adopted verbatim as the banked epistemic
state of this wave.

## 2. Authority lineage correction (documentation-only)

The executed lock cites `STRATEGY_RECIRCULATION_COMPARATOR_RULING_20260827.md` at its v1
identity (9,536 bytes, SHA `f4127faf06b026885613329bfee28d324f970c43e0aae9e61739eff20b8f9785`).
After the lock was cut, that ruling was amended at Mark's direction to v2 (12,733 bytes,
SHA `e3d60feed134a46ca0ee968b8886cb7784c2aed05d4d805e8aa3b5d94407dbfd`, Drive
`1zuedpHN5LBJq2RJfTB3fH3J0srwcrBrR`; v1 retitled `SUPERSEDED_v1_…` in place). The amendment
added §5A (BF16 determinism and instrument-variance doctrine) and changed **no operative
quantity** — comparator, bars, R-A–R-D are byte-for-byte the same rulings. Therefore: no
re-execution, no gate invalidation. The Phase-A lock records the v2 identity as the citable
authority going forward.

## 3. Churn rider: received and banked (descriptive)

The desk rider (receipt `PAPER2_RECIRCULATION_PHASE0_CHURN_DESK_RIDER_20260827.json`, 4,138
bytes, SHA `cc6bcef4…fa11b3`) sharpens the R-1 characterization at zero GPU cost: the 28
flipped rows are 25 GSM8K / 3 MBPP / 0 Tier-1; generation length does **not** separate
flipped from stable (medians equal at 256; GSM8K means 231.4 vs 235.7); flipped rows sit at
lower minimum answer-token margins, though the statistic is zero-inflated. Banked reading:
churn is task-concentrated, margin-linked, and length-independent — greedy boundary
sensitivity, exactly as §5A models it. Descriptive only; establishes no causal claim; gates
nothing. This feeds the Paper Two instrument-resolution passage.

## 4. Publication defect: accepted as packaging-only

The wrapper's `failed` status arose after all measurements completed, from a normal `git add`
into the ignored `outputs/` tree. The repair is approved as executed: archive hashed before
VM release, public summary reconstructed from the exact downloaded receipts, publication
moved to the scoped artifact whitelist with `git add -f`, regression test added (receipts
staged, checkpoints excluded), suite green at 11 passed / 171 deselected (commit
`3aabe0329d9a7eb1c1bf4a3577126d5c22b9743f`, final branch commit `3f6bbeef`). No scientific
receipt was altered and no computation repeated. The defect record is preserved as filed —
do not clean it up.

## 5. AUTHORIZATION — Phase A score-only sweep

Phase A is authorized **exactly as registered** in the amended charter (probe handoff as
amended by the semantics ruling and the comparator ruling v2). Binding parameters:

- **Scope:** score-only measurement. The registered (s,d,α) grid on Qwen2.5-0.5B-Instruct:
  96 coarse perplexity cells across the 32 registered source/destination pairs, up to 13
  refinement cells, and battery cells only as the charter specifies. Frozen weights
  throughout.
- **Comparator and bars:** baseline 160/461; `AFFORDANCE-PRESENT` additive bar 180/461;
  `NEUTRAL` lower edge 151/461. Perplexity materiality floor 1%. All classifications
  resolved by strategy at adjudication, not in-flight.
- **Budget:** the 8.0 A100-hour ceiling stands. The 3.7619-hour projection is the plan of
  record; if realized spend tracks >25% over projection at any checkpoint, stop and relay
  rather than continuing.
- **Stop-the-line triggers (unchanged):** any identity-gate drift, any comparator mismatch
  against the v2 receipt, any evaluator or runtime change, any result requiring an
  unregistered cell. Stop, archive, relay.
- **Still sealed:** Phase B (any optimizer construction or training step) — requires Mark's
  separate lock. CONFIRM and EVAL-E. No adaptive gate work.
- **Relay:** resolve no Phase-A key in-flight. Deliver the complete score-only handoff with
  receipts; strategy resolves keys and scores the blind predictions.

## 6. Blind predictions (registered now, before any Phase-A cell runs)

For honest scoring at adjudication, strategy registers:

- **P-A1:** Best grid cell perplexity reduction on Qwen lands in **[0.5%, 2.5%]** —
  i.e. tuning helps but the affordance is weak relative to Gemma. Confidence ~0.55.
- **P-A2:** At least one registered cell clears the 1% materiality floor. Confidence ~0.45
  (deliberately near coin-flip; this is the probe's actual open question).
- **P-A3:** No battery cell classifies below `NEUTRAL` (≥151/461) at the charter's α
  values — score-only recirculation does not damage task accuracy beyond the floor.
  Confidence ~0.75.
- **P-A4:** The best cell's (s,d) has s−d in the upper half of the registered offsets
  (deeper-to-shallower span wider than the Qwen timing pilot's 16→8), mirroring the paper's
  Gemma geometry scaling. Confidence ~0.5, registered as directional only.

These are strategy's predictions, not gates, and do not constrain execution.

## 7. Minimal ruling text (for the agent's lock file)

> Bank Recirculation Phase 0 as PASS under the paper-native evaluator. Identity gates
> bit-exact on Qwen and Gemma; Gemma directional anchor passed at +8.268% perplexity
> reduction; complete projection 3.7619 A100-hours within the 8-hour ceiling. Authorize
> Phase A exactly under the existing amended charter and comparator bars (160 baseline, 180
> additive, 151 neutral lower edge), score-only, with no Phase B training. Record comparator
> ruling v2 (SHA e3d60feed134a46ca0ee968b8886cb7784c2aed05d4d805e8aa3b5d94407dbfd) as the
> citable comparator authority; v1 execution remains valid, operative content unchanged.
> Preserve all Phase-0 receipts and the post-run publication-defect record. Resolve Phase-A
> keys only after the complete score-only handoff.
> Authority: STRATEGY_RECIRCULATION_PHASE_A_AUTHORIZATION_20260827.md.

— Strategy session, 2026-08-27
