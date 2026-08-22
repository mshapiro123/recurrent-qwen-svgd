# STRATEGY HANDOFF — Serving-Graph Reconciliation Desk Probe (2B-S Pre-Charter Step i)

**Date:** 2026-08-22
**Author:** Strategy agent (session bf36cdbb)
**Status:** AUTHORIZED FOR EXECUTION — desk/single-GPU, no training, no optimizer
**Basis:** `STRATEGY_2BS_PRELUDE_ADJUDICATION_20260821.md` (Drive `1UOhVd6wXCz8yBmQ0gCc3qwpvjH69TUMS`), **ratified by Mark 2026-08-22** ("Agreed on all points") — R1–R5 and the revised sequencing are binding.

---

## Plain-language summary

Mark has signed off on the whole prelude adjudication. The next step is cheap and diagnostic: figure out *why* two of our own scoring programs disagree so completely about the same model. On the identical initialization checkpoint, at loop depth 4, the P3.5 amplitude scorer says 160 questions are correct and the Stage 2B graph says 2 — and they disagree on every single one of the 461 rows, not a few. That is not noise; the two programs are computing different things. Before we design anything else, we trace both forward paths side by side, find the exact first point where they diverge, and decide which graph represents the behavior our training actually optimizes toward. If the amplitude scorer is doing something the training/serving graph should be doing but isn't, that inconsistency could itself be part of why 2B-D failed. No training runs, no optimizer, sealed sets untouched.

## 0. Ratification record

Mark ratified prelude-adjudication R1–R5 and the revised sequencing on 2026-08-22. Effective now: R1 artifact correction banked; R2 failed gate not waived; R3 reject route (a), adopt (c) native K2–K4 collapse as the boundary result **plus this reconciliation probe**; R4 no more prompt-path dose; R5 phrase audit as a pre-publication gate. Sequencing: (i) **this probe** → (ii) depth-capability existence question (designed after this result) → (iii) strategy adjudication → (iv) 2B-S charter.

## 1. State the identity contract FIRST (mandatory, before any trace)

Per the agent's own condition on route (b), the probe begins by writing down — before looking at any divergence data — the **identity contract**: the precise claim about whether the P3.5 amplitude scorer and the Stage 2B task-inference graph are *supposed* to implement the same forward computation at a given (depth K, amplitude γ) on the same checkpoint, and if so, at which tensor the two paths must agree. Register that statement in the handoff-back before the trace tables. This prevents the trace from being read into a post-hoc rationalization.

Strategy's framing to test (not to assume): at K=4, γ=0.05, on the same init weights, the two graphs *should* produce identical next-token logits if they implement the same recurrent forward pass. They produce 0/461 agreement. Therefore they implement different functions; the probe localizes where and classifies which one is the graph that defines training success.

## 2. The probe

Small matched row set (the shared 461-row slice, or a ≤64-row diagnostic subset if that resolves it — log which). CPU or single-GPU; runtime-pinned (accelerator/torch/backend named); no optimizer constructed; no CONFIRM/EVAL-E read.

**(a) Forward-path trace.** Run both evaluators on the identical checkpoint and rows, capturing intermediate tensors at each stage of the K=4 forward pass: prefix output, each loop's pre/post state, the AnchoredBridge write (B₀ + g_L·B_L, RMS-cap), the amplitude application (γ), the suffix/head input, and the final logits. Compute per-stage agreement (cosine / max |Δ| / rank agreement) between the two graphs.

**(b) First-divergence localization.** Identify the earliest stage at which the two paths diverge materially. Classify the divergence: does the P3.5 scorer (i) apply the bridge/amplitude write at a different point or with different semantics, (ii) run a different effective K or a single corrected pass rather than the full iterated loop, (iii) use a different generation/scoring mode (teacher-forced vs generative, different pooling, different answer extraction), or (iv) differ in prompt/answer formatting? One of these should account for the 160-vs-2 gap.

**(c) Success-defining graph.** Determine which graph the **registered Stage 2B training objective actually optimized** — i.e., which forward computation produced the CE/KL/mono losses and the DEV floors during 2B-D. That graph defines operative task success. State whether the historical P3.5 amplitude-surface results were measured on that same graph or on the divergent one (this bears on whether the amplitude doctrine, row 10, needs a provenance footnote).

## 3. Registered strategy prediction (blind, for the scoreboard)

First divergence at the **bridge/amplitude application step**: I expect the P3.5 amplitude scorer applies a single strong corrective write (amplitude-surface semantics — a one-shot "fix" estimate) while the Stage 2B graph runs the full per-loop innovation, so its "160" reflects a one-shot corrected estimate, not iterated fourth-loop depth. Prediction: the **Stage 2B graph is the success-defining graph** (K4=2 is the operative truth), and the P3.5 scorer over-credits at K4 by construction. If refuted — if the P3.5 path is the one training optimized — that is a more serious finding (we would have trained against a different graph than we scored on) and escalates immediately.

## 4. Decision mapping (pre-registered)

- **Divergence is a scorer artifact** (P3.5 applies non-training semantics; Stage 2B is the success-defining graph): bank K4=2 as operative truth; add a provenance footnote to the amplitude-surface results (row 10); proceed to step (ii) the depth-capability existence question. No serving-graph repair needed.
- **Divergence is a serving inconsistency** (the Stage 2B serving graph diverges from the graph that defined success, i.e., the model is served differently than trained/scored): this is a latent bug implicated in 2B-D; strategy ruling before any further design, and a serving-graph repair becomes a pre-charter item.
- **Ambiguous / both graphs internally defensible:** escalate to strategy with the trace; do not resolve locally.

## 5. Constraints

No optimizer; no training; CONFIRM and EVAL-E sealed. Runtime pinned. Every artifact SHA-256'd; retention list = identity-contract statement + per-stage trace tables + first-divergence classification + success-graph determination, all verified present at handoff (retention-verification-at-look). Wave rule: report in one handoff; strategy adjudicates and, if the mapping is clean, designs step (ii) against the result.

## 6. What comes after (bound, not specced)

Step (ii) — the **depth-capability existence question** — is designed *after* this probe lands, because its form depends on which graph is authoritative. Its target: on the success-defining graph, can any configuration make K2–K4 additive over K1, or is the depth/loop pathway subtractive as-built? That question, not "why did training break depth," is now the program's pivot. It is not authorized by this handoff.

---

*Signature block*

**Strategy:** authorized 2026-08-22 under Mark's ratification. Registered prediction blind above.
**Coding agent:** acknowledge by relaying the **identity-contract statement** (step §1) before running the trace.
**Mark:** informed; no further ratification needed to run this desk probe.
