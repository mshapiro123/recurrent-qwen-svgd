# STRATEGY BINDING RESPONSE — Recirculation Probe Semantics: R1–R5 RATIFIED, All Seven Fields Fixed

**Date:** 2026-08-27
**Author:** Strategy agent (session bf36cdbb) — author of the original probe spec, issuing the correction
**Status:** BINDING — amends `STRATEGY_RECIRCULATION_PROBE_HANDOFF_20260823.md` §§2–4 in place; Phase 0 and Phase A resume under this ruling on Mark's relay; Phase B remains unauthorized
**Basis:** Coding-agent clarification request `CODING_TO_STRATEGY_RECIRCULATION_P0_SEMANTICS_CLARIFICATION_REQUEST_20260826.md` (Drive `1H2XB99ds_6IZEM2QydKhW4tWDm9rdLT4`, 7,679 B, SHA-256 `719bb115533b372a92b404e765b11b8513c96a3813b9319eaf06244a53f938d8` — **byte-verified**, exact local decode; commit `c87d2234`). Paper citations **independently re-verified by strategy against arXiv 2608.17981** before ruling: Equation (1) index structure, the merged-stack sentence, the first-iteration-readout clause, the post-block residual-output definition of z, and the warm-up sentence all read as the agent reported.

---

## Plain-language summary

The coding agent read the paper more carefully than my spec did, stopped before writing a line of model code, and was right on both counts. My pseudocode scored the wrong pass — it read the answer off the *recirculated* computation, which would have quietly turned the probe back into the within-token reprocessing family we already closed — and it tapped the residual stream one position too early. The paper's construction reads the answer from the ordinary first pass and uses the recirculated computation only to change what *later* tokens see, which is precisely the new axis we want to test. This ruling ratifies the agent's recommended contract in full, fixes every open implementation field so the identity gate and the published Gemma anchor are well-defined, and corrects my spec on the record. Zero GPU-hours were spent finding this. The stop-the-line clause I wrote into the spec did its job against my own error — the fourth time the implementation layer has caught a specification or provenance fault before it could contaminate a result.

## 0. Verification and adjudication of the discrepancy

The clarification request is byte-verified (SHA above) and its safety state is clean: optimizer steps 0, GPU-hours 0, CONFIRM/EVAL-E false, no paid session opened. Strategy re-fetched the paper and confirmed the three decisive citations verbatim. Adjudication: **the agent is correct; my handoff §§2.1–2.2 and §3 misspecified the registered evaluator.** Specifically: (a) my pseudocode returned `lm_head(Pass-B)` — the paper reads out "following the first iteration of a stack"; (b) my z-convention was pre-block *input*, the paper's is post-block *output* ("the residual stream output after incorporating the computation of layer l"), an off-by-one that shifts every (s, d) cell, the Gemma anchor, the KV boundary, and the cost model; (c) my §2.2 carry mechanism — recirculated states reaching later positions through attention over the modified cache — was correct, and survives unchanged. The agent's scientific framing in its §3 is adopted verbatim into the probe's record: a recirculated-pass readout asks whether reprocessing the current token helps its own prediction, which is adjacent to the closed within-token family; a first-iteration readout with recirculated cache asks whether deep state from an earlier step improves later shallow computation, which is the paper's state-tracking claim and the near-miss axis this probe exists to test. My original graph would have run the former under the latter's name.

## 1. Ruling: R1–R5 RATIFIED as written

**R1 (registered readout) — RATIFIED.** Scored logits come from the first iteration of each input step's stack, at every position, in every cell type (perplexity and battery). The recirculated computation is state construction for later positions and is never the current token's scored readout.

**R2 (tap convention) — RATIFIED.** All layer indices refer to post-block residual-stream outputs, matching the paper. The destination mix combines two post-block-d states and the next computed block is d+1. The paper-layer-to-implementation mapping table (§2 below) is frozen and receipted before the Gemma anchor runs.

**R3 (reference before optimization) — RATIFIED.** The token-sequential reference evaluator realizing the paper's dependency graph is the registered evaluator definition. Any batched-diagonal or pipelined implementation is admissible only after passing the equivalence gate: identical scored logits on a fixed short sequence, identical committed per-layer K/V tensors, identical first-position and final-position behavior, tested at α = 0 and at one nonzero anchor cell. Throughput optimization never defines the reference. Clarification so the reference is not needlessly slow: "serial" means sequential in *position*; batching across independent sequences within the reference is permitted and does not alter the dependency graph.

**R4 (revised identity gate) — RATIFIED.** At α = 0: complete-sequence first-iteration logits and all committed future-visible K/V tensors must match the intact model bitwise. The same-token Pass-B logit comparison in the original gate (g2) is removed as a scored quantity — the recirculated pass is not the readout — but the committed-K/V comparison retains its function: at α = 0 the recirculated recompute must reproduce the scored states exactly, so any nondeterminism in the recompute path surfaces in the committed-tensor diff. Stop-the-line on any nonzero difference, with the responsible op reported.

**R5 (graph receipt) — RATIFIED.** Before the Gemma anchor, emit the machine-readable graph receipt: for each (position, layer), the architecture-copy index, input-step index, tensor tap, K/V owner, and the tensor's status — scored, provisional, committed, or discarded. The receipt is the frozen definition against which the equivalence gate and both anchors are judged.

## 2. The seven requested fields, fixed

**(1) Scored iteration.** First iteration, per R1.

**(2) Tap convention.** Post-block residual outputs, per R2. Formally, with z_{j,l} the post-block-l residual output at input position j from the scored stack: the recirculated state is z′_{j,d} = β · z_{j,d} + α · f(σ_j) with σ_j = z_{j,s} and f the norm-match of the original spec (‖z_{j,d}‖₂/‖σ_j‖₂ scaling, per position). The recirculated computation evaluates blocks d+1 through L on z′_{j,d} at position j.

**(3) Paper-layer to implementation mapping.** Paper layer l (1-based, "output after incorporating layer l") maps to the Hugging Face `hidden_states[l]` tensor under `output_hidden_states` semantics, where `hidden_states[0]` is the embedding output and `hidden_states[l]` is the output of decoder block `model.layers[l−1]` (0-indexed). Therefore: σ_j = `hidden_states[s]`; the destination state is `hidden_states[d]`; the recirculated recompute runs `model.layers[d]` through `model.layers[L−1]` (0-indexed), which are paper blocks d+1..L. The final pre-head norm is applied only where the intact model applies it. The identical convention is used for the Gemma3-1B anchor — Gemma3's within-block normalization placement does not change the identity of the inter-block residual tensors `hidden_states[l]`. This table is emitted in the R5 receipt and frozen before any anchor or grid cell.

**(4) K/V ownership.** One committed cache entry per (position, block). At position j: blocks 1..d commit the scored stack's K/V. Blocks d+1..L commit the **recirculated** computation's K/V, replacing the scored stack's entries for those blocks, which are provisional — used transiently inside the scored stack's own forward at position j and then discarded. This is the cross-step carry: later positions' scored stacks attend to recirculated deep state and unmodified shallow state. The paper specifies the activation recurrence and not a cache algorithm, so this ownership rule is a **strategy-fixed convention, disclosed as such in the receipt**; if any paper detail or released reference implementation contradicts it, the paper wins and the deviation is disclosed before results are compared to published numbers.

**(5) Warm-up and flush.** The recirculated computation exists for every position including the first — the paper's warm-up sentence ("two input stacks are run in parallel at each recurrence step, except for the very first step, which serves as a warm up") is read as a *scheduling* statement about the parallel slot, not an α-suppression at position one; position one's recirculated deep state is exactly what position two's scored stack should see. Position-ids and the causal mask are those of the intact model — the recirculated computation runs at the same position with the same mask, and no extra sequence dimension exists. At the final position of a sequence, the recirculated computation alters nothing observable and **may be skipped**; the skip is uniform across all cells and disclosed in the receipt. The α-ramp remains a separate registered A3 variant and is not conflated with warm-up.

**(6) Serial reference admissibility.** Yes — the correctness-first serial reference is the registered evaluator, per R3, even though the paper's deployment schedule is pipelined. Equivalence-gated optimized schedules are implementation detail, never definition.

**(7) Cost-ceiling application.** The 8 A100-hour ceiling covers the entire probe (Phases 0 + A) under whichever evaluator actually runs, including any equivalence-gate compute. Sequence: run the timing pilot on the serial reference first. If the projection fits the ceiling, run serial and build nothing further. If it exceeds, build the batched-diagonal evaluator, pass the R3 equivalence gate, re-pilot, and proceed only if the new projection fits; otherwise stop and relay. No grid cell runs on an evaluator that has not either *been* the reference or *passed the gate against it*.

## 3. Amended pseudocode (replaces §3 of the handoff)

```python
# ---- Paper-native recirculating evaluator (score-only; no grad) ----
# Registered readout: FIRST-iteration logits. Recirculated pass: cache construction only.
# hs[j][l] denotes the post-block-l residual output at position j (hs[j][0] = embedding out).

def recirculating_forward(model, tokens, s, d, alpha, beta_mode, ramp=None, L=24):
    assert s > d
    kv = KVCache(layers=L)                    # committed entries only (one per position, block)
    scored_logits = []
    for j in range(T):                        # sequential in position; batch across sequences OK
        # ---- scored stack (first iteration): full forward, standard cache attention ----
        h = embed(tokens[:, j]); hs = {0: h}
        prov_kv = {}                          # provisional own-position K/V for blocks d+1..L
        for l in range(1, L + 1):
            h, kv_l = block_forward(model, l, h, kv, pos=j)
            hs[l] = h
            if l <= d: kv.commit(j, l, kv_l)  # shallow K/V committed from scored stack
            else:      prov_kv[l] = kv_l      # deep K/V provisional (used in-pass, then replaced)
        scored_logits.append(lm_head(final_norm(hs[L])))   # <-- READOUT: first iteration
        # ---- recirculated computation: state construction for later positions ----
        if j < T - 1:                         # final-position skip (uniform, receipted)
            a_j  = alpha if ramp is None else alpha * min(1.0, j / ramp)
            beta = (1.0 - a_j) if beta_mode == "convex" else 1.0
            sigma = hs[s]
            s_bar = (hs[d].norm(dim=-1, keepdim=True) / sigma.norm(dim=-1, keepdim=True)) * sigma
            h = beta * hs[d] + a_j * s_bar    # mix of two post-block-d states
            for l in range(d + 1, L + 1):     # recompute blocks d+1..L at the SAME position j
                h, kv_l = block_forward(model, l, h, kv, pos=j)
                kv.commit(j, l, kv_l)         # deep K/V committed from the RECIRCULATED pass
        else:
            for l in range(d + 1, L + 1): kv.commit(j, l, prov_kv[l])
    return stack(scored_logits)

# Identity gate (R4), at alpha == 0:
#   scored_logits  == intact-model logits, bitwise, full sequence; AND
#   every committed K/V tensor == intact-model K/V, bitwise
#   (convex mode at alpha=0 gives h = hs[d] exactly; the committed-tensor diff catches
#    any recompute nondeterminism). Stop-the-line on any nonzero difference.
# Equivalence gate (R3), serial reference vs any optimized schedule:
#   identical scored logits + identical committed K/V on a fixed short sequence,
#   at alpha=0 and one nonzero anchor cell, before any grid cell runs.
# Graph receipt (R5): per (position, layer): copy index, step index, tap, K/V owner,
#   status in {scored, provisional, committed, discarded}; frozen before the Gemma anchor.
```

Cost model, corrected for the d+1 start: per position ≈ 1 + (L − d)/L stack-equivalents (the recirculated pass runs L − d blocks). The pilot-cell estimate and ceiling are unchanged in substance.

## 4. What does not change

Everything scientific is untouched, exactly as the agent proposed: the sweep grid, α values, corpus slice and decontamination screen, the 461-row battery protocol and its banked 162/461 anchor, the effect floors and neutral band, the registered keys and interpretation map, the Gemma (11,4) anchor cell labels (now well-defined under the frozen mapping), the Phase-A structure, the ceiling, and Phase B's non-authorization. My §7 blind prediction stands as registered — it predates any data and the semantics correction does not touch it.

## 5. Scoreboard entry (on the record)

The original handoff's §2.1 and §3 misspecified the registered readout (recirculated pass instead of first iteration) and the tap convention (pre-block input instead of post-block output, with the recompute starting one block early). The error was caught by the spec's own verification-duty clause — paper wins, stop-the-line — before implementation, at zero GPU cost. This is the **fourth implementation-layer catch** in the program record (after the F1 zero-denominator repair, the depth-4 mixed-evaluator artifact, and the seed-1 lock transcription), and the first whose subject was a strategy-authored specification of an external method. The §2.2 carry mechanism and all scientific framing survive. Standing lesson, added to the working practice: when specifying an external paper's computation, the spec must quote the paper's own equation with its index definitions rather than paraphrase the graph — the paraphrase is where the readout and the off-by-one entered.

---

*Signature block*

**Strategy:** R1–R5 ratified; seven fields fixed; spec error owned on the record 2026-08-27. Phases 0 + A resume under this ruling on Mark's relay; the graph receipt and both anchors now have a frozen definition.
**Coding agent:** a textbook stop — correct on the readout, correct on the tap, correct to freeze the cache fields before anchoring, and zero compute spent. Acknowledge by relaying the R5 graph receipt and Phase 0 gate results before the A2 grid.
**Mark:** no scientific decision required — this is a specification correction ratifying the agent's paper-native contract. Relay to resume Phase 0.
