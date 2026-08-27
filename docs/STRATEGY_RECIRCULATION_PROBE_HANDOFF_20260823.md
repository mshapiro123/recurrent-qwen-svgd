# STRATEGY HANDOFF — Recirculation Affordance Probe on Intact Qwen2.5-0.5B: Executable Specification

**Date:** 2026-08-23
**Author:** Strategy agent (session bf36cdbb), at Mark's request
**Status:** EXECUTABLE SPEC — Phase 0 and Phase A authorized on Mark's relay; Phase B specified but NOT authorized (requires a separate lock after Phase A adjudicates)
**Basis:** Mozer, Siddiqui, Sawyer, Sanyal, Liu, "Recirculation" (arXiv 2608.17981); Mark's summary of 2026-08-23 (verified against the paper by strategy); the program's standing laws (measurement floors, runtime pinning, provenance tags, sealed partitions). This probe runs on the **intact** base model — it does not touch the closed loop implementation and does not violate the close-out clause.
**Routing:** results relay to Mark as one handoff under the wave rule; adjudication lands with the strategy session hosting the successor program, which also logs this probe in the tracker (suggested: row 24).

---

## Plain-language summary

A new DeepMind paper shows that feeding a transformer's own deep-layer conclusions back into its shallow layers — gently, with a small mixing weight, one extra pass per token — improves the model with no training at all, because the content being written back is the model's own already-resolved computation. This is a near-miss diagnosis of our closed depth study on two counts: their recurrence axis is the one we never tested (deep state made available to the *next step's* shallow computation, rather than re-processing the *same* token harder), and their write content is pre-aimed by pretraining (ours was trained from scratch and came out aimless). The probe below asks one question cheaply: does our frozen Qwen2.5-0.5B have the same affordance, and where? The answer is a heatmap over (source layer, destination layer, mixing weight), read in both perplexity and our own answer batteries. If a robust region exists, it tells the successor program exactly where the model's natural computational seams are — measured, not guessed — and opens a two-timescale architecture. If nothing survives honest tuning, that is substrate-family evidence worth having before further retrofit investment. No training, no optimizer, sealed partitions untouched, and a hard identity gate: with the mechanism switched off, the model must be bit-exact to the intact base.

---

## 0. Why this probe exists (reasoning the implementer should hold)

The closed depth study established, both seeds, margin-backed: (i) within-token re-entry of the recurrent block destroys task signal at a measured per-loop attenuation near r ≈ 0.44 (undamped re-processing — effectively a full-strength substitution of state, α = 1 in the language below); (ii) removing re-entry makes depth harmless and inert; (iii) our trained write content is broad and weakly aimed — delivered at 18.65 percent of hidden RMS it changes ~45 percent of predictions with zero net gain. The failure was localized to *compounding* and to *unaimed content*, not to the write channel or the substrate's tolerance of writes.

Recirculation differs on exactly those two axes. **Damping:** its update is a convex, norm-matched mix, d′ = (1−α)d + α·s̄ with α ≈ 0.04–0.16 — a contraction-controlled carry (see §2.4) rather than full re-processing. **Content:** the injected state is the frozen model's own deep representation, pre-aimed by pretraining — no objective has to create alignment. It works training-free on Gemma3 (up to ~8.5 percent mean perplexity reduction fixed, ~23 percent with a learned gate, exceeding full fine-tuning's 21.6 percent in that comparison). The known caution: with shared, un-tuned hyperparameters, non-Gemma families including Qwen3 improved by less than ~0.5 percent — the authors explicitly did not tune normalization or α for them, so those numbers are un-tuned lower bounds, not ceilings. The probe must therefore tune before concluding absence.

Decision value: a robust Qwen2.5 region ⇒ the successor gains a measured [destination, source] interval — empirical evidence for where prelude, recurrent core, and coda should sit, and a candidate outer-recurrence mechanism to pair with the parallel inner architecture. A flat result after tuning ⇒ a real substrate-family finding that reweights the retrofit-versus-redesign decision. Both branches pay.

## 1. Regime and laws (binding)

Score-only. **No optimizer constructed, no training, no gradient computation in Phases 0 and A.** CONFIRM and EVAL-E sealed — untouched. Runtime pinned (name accelerator / torch / CUDA / dtype / attention backend in the receipt; the pinned reference runtime is A100-SXM4-40GB, torch 2.11.0+cu128, bf16, SDPA). Every cell carries an evaluator-provenance tag; no curve mixes evaluators; no hybrid cells. Battery reads use the matched 461-row DEV slice (369 GSM8K / 67 MBPP / 25 Tier-1) with all row-level predictions retained per cell. The intact-model battery baseline is the banked 162/461 (the loop graph's K1 is base-identical) — do not re-spend it, but DO re-verify it once on the probe's own harness as a pre-flight consistency check (stop-the-line on mismatch). Effect floors: ≥ 20 rows over 162 for any battery-additivity claim; the nine-row band is the neutral zone. Perplexity materiality floor: ≥ 1.0 percent mean reduction on the registered corpus slice (evaluation is deterministic, so the floor is materiality, not noise). **Budget from measured cell times:** Phase A1 is a mandatory timing pilot, and the agent reports the projected total before running the grid. Pre-authorized ceiling for Phases 0 + A: **8 A100-hours**. A projection exceeding the ceiling stops for relay before any grid cell runs.

## 2. Mechanism specification

### 2.1 Notation and the core update

Let the intact model have layers 1..L (L = 24 for Qwen2.5-0.5B), and let z_{t,ℓ} denote the residual-stream input to layer ℓ at token position t. Fix a **source** layer s and **destination** layer d with s > d. Recirculation, one iteration, per the paper:

Pass A (iteration 0) at position t: run the full stack normally; capture the source activation σ_t = z_{t,s} (the residual input to layer s; record this convention in the receipt and keep it fixed — input-to-layer, post-residual-sum, pre-block).

Pass B (iteration 1) at position t: recompute layers d..L with the destination input replaced by the mix

    z′_{t,d} = β · z_{t,d} + α · f(σ_t)

with the **norm-matched** normalization (paper default)

    f(σ) = ( ‖z_{t,d}‖₂ / ‖σ‖₂ ) · σ

computed per position (per token vector, not per batch). Variants: **convex** β = 1 − α (primary, used for their 1B results) and **additive** β = 1 (secondary arm; the paper used it for 4B/12B). The KV entries that enter the cache for position t, at layers d..L, are those of Pass B. Layers 1..d−1 are computed once (Pass A) and their KV stands.

### 2.2 Operational semantics — read carefully, this is where implementations go wrong

The mixing is applied within position t (Pass B re-derives the deep stack of the same token), and the **temporal carry is through attention**: position t+1's layers attend to position t's *recirculated* KV at layers d and above. That is the mechanism by which a conclusion reached deep at position t becomes available to shallow computation at position t+1 — and it is why prefill cannot be parallelized: position t+1's forward requires position t's Pass B to have completed. Prefill must run token-sequentially with the same two-pass procedure at every position (recirculation active during prompt processing and generation alike — one consistent evaluator). Cost per position ≈ 1 + (L − d + 1)/L stack-equivalents, so shallower destinations are more expensive. Batch across sequences (parallel in batch, sequential in position) to recover throughput.

**Implementation-verification duty:** before coding, read the paper's Section 2 and Appendix A (arXiv 2608.17981) and confirm this operational reading — the two-pass-per-token structure, the KV convention, and the source/destination indexing — against their description and Figure 3/4. If the paper's construction differs in any detail from §2.1–2.2 (in particular the exact tap point of σ and whether Pass A KV above d is ever visible to later tokens), **the paper wins**, and the deviation from this spec is disclosed in the receipt. This handoff's grid and keys do not depend on the fine convention, but cells are only comparable if one convention is fixed and tagged.

### 2.3 Positional ramp (registered variant, Phase A3 only)

The paper finds recirculation harms the earliest positions (little state to carry) and fixes it by ramping α: α_t = α · min(1, t / T_ramp) with T_ramp = 10. Implement as a flag, off in the coarse grid, on as one refinement arm.

### 2.4 Why this is expected to behave unlike our re-entry (context, not implementation)

Writing s̄ = f(σ), the convex update is d′ = d + α(s̄ − d): damped Picard iteration on the fixed-point problem d* = s̄(d*), i.e., forward-Euler on dd/dτ = s̄(d) − d with step α. Near a fixed point the update's Jacobian is (1−α)I + αJ, so small α keeps the spectral radius near 1 — a contraction-controlled carry. Our measured failure was the α → 1, iterate-K-times regime on trained content: geometric signal attenuation (r ≈ 0.44) compounding over loops. Recirculation runs one damped iteration on pretrained content. The paper's own observation that larger α enlarges both the benefit and the harmful (s,d) region is the same direction-relative amplitude physics our program banked. This is why the probe sweeps α as a first-class axis and never exceeds 0.25.

## 3. Pseudocode

```python
# ---- Recirculating evaluator (score-only; no grad anywhere) ----
# model: frozen Qwen2.5-0.5B-Instruct, hooks at layer inputs.
# All tensors bf16 on the pinned runtime; batch B sequences, positions processed sequentially.

def recirculating_forward(model, tokens, s, d, alpha, beta_mode, ramp=None):
    # tokens: [B, T]; returns per-position logits [B, T, V]
    assert s > d
    kv = KVCache(layers=model.L)            # one cache; layers d..L hold Pass-B entries
    logits_out = []
    for t in range(T):                       # sequential in position — no parallel prefill
        x_t = tokens[:, t]
        # ---- Pass A: full stack, normal attention over kv ----
        h = embed(x_t)
        actsA = {}
        for l in range(1, model.L + 1):
            actsA[l] = h                     # residual input to layer l
            h = layer_forward(model, l, h, kv, write_kv=(l < d))   # layers < d commit KV now
        sigma = actsA[s]                     # source activation, [B, H]
        # ---- mixing at destination ----
        z_d  = actsA[d]
        a_t  = alpha if ramp is None else alpha * min(1.0, t / ramp)
        beta = (1.0 - a_t) if beta_mode == "convex" else 1.0
        s_bar = (z_d.norm(dim=-1, keepdim=True) / sigma.norm(dim=-1, keepdim=True)) * sigma
        z_prime = beta * z_d + a_t * s_bar
        # ---- Pass B: layers d..L recomputed; their KV is what the cache keeps ----
        h = z_prime
        for l in range(d, model.L + 1):
            h = layer_forward(model, l, h, kv, write_kv=True)      # overwrite/commit d..L
        logits_out.append(lm_head(h))
    return stack(logits_out)

# ---- Identity gate (hard): alpha == 0 must be BIT-EXACT to the intact model ----
# Two checks, both required:
#  (g1) short-circuit path: alpha==0 -> skip Pass B entirely, commit Pass-A KV for all
#       layers; assert logits identical to the unmodified model's logits (bitwise).
#  (g2) live path at alpha==0: run the full two-pass machinery with a_t = 0
#       (z_prime == z_d exactly in convex mode); assert bitwise identity to (g1).
#       If any kernel nondeterminism breaks (g2) bitwise, report max |delta_logit|
#       and the responsible op; proceed only if max |delta| == 0 under deterministic
#       kernel settings, else stop-the-line and relay.

# ---- Perplexity cell ----
# corpus slice -> pack to [B=32, T=1024]; run recirculating_forward; report mean NLL
# and percent change vs the intact baseline computed ONCE with the same packing.

# ---- Battery cell (Phase A3) ----
# Greedy generation, recirculation active during (sequential) prompt processing and
# generation; identical prompts/stopping/scoring as the banked 461-row harness;
# retain all row-level predictions; provenance tag names (s, d, alpha, beta_mode,
# ramp, normalization) in every cell record.
```

## 4. Phase 0 — implementation gates (before any grid cell)

1. **Identity gate** (g1) and (g2) above — bit-exact at α = 0. Stop-the-line on failure.
2. **Battery-harness anchor:** re-verify the intact model scores 162/461 on the probe's own battery path (this is the banked baseline; a mismatch means the harness, not the model). Stop-the-line on mismatch.
3. **Published-result anchor (required if Gemma3-1B weights are accessible; otherwise disclose and skip):** implement the identical evaluator on Gemma3-1B, cell (s, d, α) = (11, 4, 0.15), convex norm-matched, ~128K corpus tokens. The gate is directional, not numeric: perplexity must *decrease* vs the intact Gemma3-1B baseline on the same slice. This validates the implementation against the paper's central published result before any Qwen conclusion is drawn. Report the measured reduction; do not tune.
4. **Timing pilot (Phase A1):** one Qwen cell at (s, d, α) = (16, 8, 0.10), 32K tokens, batched. Report measured seconds per cell and the projected total for the full Phase A plan. **If the projection exceeds the 8 A100-hour ceiling, stop and relay before proceeding.**

## 5. Phase A — the sweep (authorized on Mark's relay)

**A2 — coarse heatmap (perplexity, convex norm-matched, no ramp).**
Grid: d ∈ {2, 4, 6, 8, 10, 12, 14}; s ∈ {d+4, d+6, d+8, d+10, d+12} intersected with s ≤ 22; α ∈ {0.05, 0.10, 0.16}. (≈ 90–100 cells.) Corpus: a fixed, registered ~32K-token slice per cell drawn once from a public LM corpus mix (e.g., C4-style web text + PG19-style book text, 50/50), identical across cells, packed identically, hash-recorded. Decontamination hygiene: screen the slice for 13-gram overlap against the 461-row battery texts and drop overlapping documents (the sealed batteries are untouched by construction — this screen protects the battery reads of A3 from indirect leakage into cell selection). Baseline: intact-model NLL on the same slice, computed once. Deliverable: the (s, d) heatmap per α (percent NLL change, diverging scale), plus the per-cell table.

**A3 — refinement and battery reads (top region only).**
From A2, take the best contiguous region (not isolated cells). Then: (i) α fine-sweep {0.02, 0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25} at the best (s, d); (ii) the additive variant (β = 1) at the top three (s, d) pairs at their best α; (iii) the α-ramp variant (T_ramp = 10) at the single best cell; (iv) one alternative normalization (identity, no renorm) at the best cell — the paper found norm-matching buys robustness, and one cell prices that on Qwen. Then **battery reads:** the full 461-row generative battery at the best two configurations. Compare against the banked 162/461 baseline under the effect floor (≥ 182 for additive) and the nine-row neutral band. Optional, only if a battery read is additive or near-floor: the 2,048-row margin panel at that single configuration, to margin-back the read.

**Both-seeds note:** the intact model has no training seeds — determinism replaces replication here, and the identity gate plus fixed packing carry that burden. The battery harness is greedy and deterministic. State this in the receipt rather than fabricating a seed axis.

## 6. Registered keys and interpretation map (fixed before looking)

- `AFFORDANCE-PRESENT`: a contiguous (s, d) region (≥ 3 adjacent cells) improves perplexity by ≥ 1.0 percent at some α ≤ 0.25, AND the best-cell battery read is within the neutral band or better (≥ 153). The probe succeeds: Qwen2.5 has the affordance; the region's [d, s] interval is the measured seam estimate.
- `AFFORDANCE-PRESENT-BATTERY-ADDITIVE`: as above AND a battery read ≥ 182/461. (Not predicted; would be a headline.)
- `PERPLEXITY-ONLY-WITH-BATTERY-HARM`: a qualifying perplexity region exists but every battery read at qualifying cells falls below 153/461. Relay before interpretation — do not resolve locally.
- `ABSENT`: after the A3 tuning (α fine-sweep, both mixing variants, ramp, alternative normalization) no cell reaches the 1.0 percent perplexity floor. Banked as a substrate-family boundary datum.
- Any ambiguous or mixed pattern (e.g., isolated non-contiguous winners, additive-variant-only effects): stop and relay. No improvised keys.

## 7. Registered strategy predictions (blind; labeled with the standing calibration caveat)

`AFFORDANCE-PRESENT`, with a contiguous middle-layer region (destination roughly 4–10, source roughly 12–18) and best-cell perplexity reduction in the 0.5–3 percent range after tuning — a Gemma-fraction effect, consistent with the un-tuned Qwen3 prior — and battery reads inside the neutral band (no additive claim expected training-free). Confidence LOW: my mechanism point-predictions have a documented miss record; the keys, not this guess, decide. If Phase 0's Gemma anchor fails directionally, the implementation — not the substrate — is the first suspect.

## 8. Phase B — adaptive recirculation (SPECIFIED, NOT AUTHORIZED)

Recorded so the design is on file; requires Phase A adjudication and a separate lock from Mark before any part runs, because it is training. Design: freeze everything in the base; add a gate network G taking (z_{t,d}, f(σ_t)) and producing per-dimension vectors (α_t, β_t) via sigmoid outputs; update z′ = β_t ⊙ z_d + α_t ⊙ f(σ). Identity at initialization is mandatory per the program's doctrine: initialize the final-layer biases so sigmoid(α-logits) ≈ 0 and sigmoid(β-logits) ≈ 1 at step zero, giving bit-exact base behavior before training (no zero-matrix multiplicative traps — bias-shift, not zero weights). Train by truncated BPTT through the two-pass computation on next-token loss over the registered corpus, both seeds, under the guardrail doctrine (calibrated retention floor armed, per-loss gradient-share floors, dual write-magnitude telemetry adapted to (‖α_t ⊙ f(σ)‖ accumulated vs deployed), battery-based endpoint bars, effect floor ≥ 20 rows). The paper's ablation ordering (scalar < vector, static < token-conditioned) is the arm structure. Costing from A1's measured times before the lock.

## 9. Deliverable contract

One result handoff on Drive (folder `1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr`), byte-sized and SHA-256'd in the relay, containing: Phase 0 gate results (identity bitwise report, harness anchor, Gemma anchor with measured reduction or disclosed inaccessibility); the A1 timing report with projected-versus-actual cost against the 8 A100-hour ceiling; the A2 heatmaps (SVG + PNG) and full per-cell table with provenance tags; the A3 refinement table and battery rows (all 461 row-level predictions per battery cell); keys resolved per §6; the runtime pin block; a deviations section (empty or itemized — including any §2.2 convention correction from the paper); retention list verified at handoff. Colab closeout: no active paid sessions. Acknowledge by relaying the Phase 0 gate results before running the A2 grid.

---

*Signature block*

**Strategy (bf36cdbb):** spec authored 2026-08-23 at Mark's request; predictions registered blind §7; Phase B explicitly gated.
**Coding agent:** verify §2.2 against the paper before implementation; acknowledge with Phase 0 gates; stop-the-line rules as marked; the cost report is mandatory before the grid.
**Mark:** relay to the coding agent to authorize Phases 0 + A; Phase B requires your separate lock after adjudication. Adjudication and tracker logging route to the successor-program strategy session.
