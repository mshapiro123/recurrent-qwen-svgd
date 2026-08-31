# STRATEGY — G-TOK Confirmation Semantics: The Two Rulings That Unlock the A100 Screen

**Date:** 2026-08-31 · **Status:** RULING, resolving the two blockers named in `CODING_TO_STRATEGY_WEFT1_GTOK_CONFIRMATION_CLARIFICATION_REQUEST_20260830.md`. **The fail-closed A100 base screen unlocks the moment the coding agent verifies this document's bytes and SHA-256** — no further strategy input is on the path.
**Precedence:** handoff → A1 → A2 → A3 → this ruling. Nothing upstream is amended; this document *defines* two procedures the upstream chain named but did not fully specify, which is exactly what the agent's fail-closed stop said it needed. The stop was correct — the sixth consecutive correct fail-closed call.
**Scope:** §1 accepts the scaffold checkpoint at `f8176fec`. §2–§4 are the rulings (SEM-1, SEM-2, SEM-3). §5 is a non-blocking P-A throughput flag. Nothing here requires a decision from Mark.

---

## 0. Plain-language summary

The agent asked two questions that the protocol chain left underdetermined: exactly how "equal FLOPs" is enforced when training proceeds in whole optimizer steps, and exactly how the compute-matched confirmation pair is built and how a reversal there is recognized.

The first has an easy answer because the arithmetic is kind: at batch 256×2048, a G-TOK run is roughly 1,900–2,300 steps, so a single step is about a twentieth of a percent of a run. Rounding to whole steps therefore cannot distort anything the decision reads. The rule is a floor — never overshoot a budget, always stop at the last complete step under it — with the final partial batch dropped under one identical rule across arms, and a ±1 % validity band that whole-step quantization satisfies with two orders of magnitude to spare.

The second has a structural answer that saves compute. Byte-matched FLOPs *rise* with vocabulary size — the embedding and softmax work grows faster than fertility shrinks token counts — so of the two arms entering confirmation, the smaller-FLOP arm's byte-matched runs already sit exactly at the confirmation budget if we define that budget as the smaller arm's measured total. So we do: F\* is the smaller arm's realized measured FLOPs, its byte-matched runs are reused as its confirmation results, and only the larger-FLOP arm trains fresh — two seeds, same byte stream, tokenizer reused never refit, stopped at the whole-step count that lands within ±1 % of F\*. This halves the confirmation's fresh compute and introduces no asymmetry that matters, because the reused runs and the fresh runs are scored by the identical pooled-BPB rule at the identical budget.

Reversal is then defined the way the byte-matched decision was, with the same asymmetry: the byte-matched runner-up overturns the winner only if it wins at compute-match by more than two pooled standard errors with both seeds agreeing in sign — and if the would-be reversal installs the *larger* vocabulary, it must clear the same three-standard-error band D-C-2 imposes everywhere else, so the small-vocabulary default is never abandoned on noise. Inside the band, nothing happens: the byte-matched decision stands, because closeness was already handled by the tie rule at the decision stage. A seed split escalates, as it does everywhere in this protocol.

One operational note rides along: the P-A materialization pass is healthy but, at the observed throughput, roughly a week from completion. No intervention — just a request that the next receipt carry the agent's own ETA projection so the number is owned by the party who can see both workers.

---

# 1. Checkpoint accepted — the scaffold stands

| item | status |
|---|---|
| Governed G-TOK scaffold, commits `83d72753` + `f8176fec`, HEAD `f8176fec92f2647fc483adc231063e21f13f57da`, worktree clean 0/0 | **accepted** |
| Independent audit: no remaining P0/P1 findings | **accepted** — this is the audit posture the program wants before any GPU spend |
| Focused/adjacent verification 188 passed; repository gate PASS; 3,829 passed with the one governed quarantined failure | **accepted**; quarantine review date 2026-09-04 stands |
| P-A: both workers alive, 713,183,545 durable bytes at 2 h 30 m, `_INCOMPLETE` present, no P-A receipt claimed | **accepted as in-progress** — correctly unclaimed; see §5 |
| A100 base screen held fail-closed pending these semantics | **correct call** — the two questions are real underdetermination, not caution theater |

# 2. Ruling SEM-1 — whole-step equal-FLOP matching

**The quantization context, so the tolerance is seen to be safe:** at batch 256 × 2048 = 524,288 tokens/step, the byte-matched runs land at roughly 2,312 steps (16K arm) down to roughly 1,884 steps (48K arm). One step is 0.043–0.053 % of a run. Any whole-step rule is therefore invisible at the decision's resolution; what matters is only that the rule is *identical across arms and replayable*.

> **SEM-1.** Four clauses.
>
> **(a) Step count from the stream, floor rule.** For every run, the number of optimizer steps is `n = floor(N_tokens / (256 × 2048))`, where `N_tokens` is the token count of the arm's tokenized byte-matched stream (A1-R3/A2 byte rule, document-aligned). **The final partial batch is dropped — never padded, never wrapped** — under this one rule for every arm and every seed. The receipt reports, per run, the dropped-token count and its fraction of the stream (expected < 0.06 %).
>
> **(b) Measured FLOPs are the authority.** "FLOPs" everywhere in G-TOK means the R-G4g complete-measured-FLOPs counter — embeddings, softmax, all of it — never the analytic 6ND shortcut. Analytic figures (including the pre-registered per-arm projections `2.756e17 / 2.807e17 / 2.924e17 / 3.241e17`) are sanity references only; a measured value diverging from its projection by more than 10 % is a stop-and-report, not a silent overwrite.
>
> **(c) Budget stops are floors.** Wherever a run must stop at a FLOP budget `F` (the confirmation runs of SEM-2; the 0.25×/0.5× BPB curve points are *token*-count checkpoints and unaffected), the stop is at `n = floor(F / f_step)` whole steps, `f_step` being the run's own measured per-step FLOPs from its calibration burst (the A2-R6 burst already exists; it now also serves this purpose). Never round up: undershoot is bounded by one step, overshoot is unbounded discipline-wise.
>
> **(d) Validity band ±1 %.** A budgeted run is valid iff its realized measured FLOPs land within ±1 % of `F`. Whole-step quantization consumes at most ~0.05 % of this band; the band exists to catch counter drift or a mis-calibrated `f_step`, and a violation means re-run with the corrected step count — it is a validity condition, not a tolerance to engineer toward.

# 3. Ruling SEM-2 — construction of the compute-matched confirmation pair

**The structural fact the construction exploits:** byte-matched FLOPs are monotone *increasing* in `V` — parameter cost beats fertility at these scales (measured projections rise from 2.756e17 at 16K to 3.241e17 at 48K). So of the top-2 arms, the smaller-`V` arm is the smaller-FLOP arm, and its byte-matched runs already sit at the natural confirmation budget.

> **SEM-2.** Six clauses.
>
> **(a) The pair.** The confirmation pair is the top-2 arms of the byte-matched decision, taken *after* the tie rule and the D-C-2 asymmetric 3ŝ band have been applied — i.e., the declared winner `W` and the runner-up `U`.
>
> **(b) The budget.** `F* = min over {W, U} of (mean over both seeds of realized measured total FLOPs of the arm's byte-matched runs)`. The min is realized measurement, not projection.
>
> **(c) Reuse of the min-FLOP arm.** The min-FLOP arm's two byte-matched runs **are** its confirmation results — identical budget by construction, identical data, identical scoring. No re-run, no refit. This is legitimate precisely because the confirmation deliverable is curves and receipts, not retained checkpoints: nothing about the reused runs is inferior for the comparison being made, and re-running them would purchase only noise.
>
> **(d) Fresh runs for the max-FLOP arm.** The larger-FLOP arm trains fresh, **both seeds**, under registry seeds `gtok.confirm.{arm}.{seed}` (new streams — never the byte-matched seeds, so the confirmation is not correlated with the measurement it checks). Identical byte stream as its byte-matched runs; **tokenizer reused, never refit** (the tokenizer identity is the arm — refitting would silently change the arm under A2-R2); optimizer, schedule, and all hyperparameters identical to the byte-matched recipe (the cosine schedule's horizon is set to the run's own step count from SEM-1(c), which is the standard interpretation of "same recipe at a different budget"). Stop per SEM-1(c) at `floor(F*/f_step)`; validity per SEM-1(d).
>
> **(e) Scoring.** Pooled and per-stratum BPB exactly as in the byte-matched decision, three-point curve included (0.25×/0.5×/1.0× of the *confirmation* token count for the fresh runs; the reused runs' existing curve stands).
>
> **(f) Cost and tripwire.** Fresh compute ≈ one arm × two seeds ≈ the budgeted ~1.3 A100-hr, inside the 12 A100-hr tripwire with A2-R6's projection-halt and cumulative meter enforcing it — the calibration burst of each fresh run projects total cost *before* the run proceeds, per the existing rule.

# 4. Ruling SEM-3 — reversal semantics

> **SEM-3.** Five clauses.
>
> **(a) The statistic.** Let `ΔBPB_c = BPB_c(W) − BPB_c(U)` on pooled full-budget confirmation BPB, and let `ŝ_c` be constructed exactly as `ŝ` was at the byte-matched decision (D-C-2's estimator), computed on the confirmation results — the reused pair for the min-FLOP arm, the fresh pair for the other.
>
> **(b) Reversal condition.** `U` overturns `W` iff **both** hold: `ΔBPB_c > 2ŝ_c` (U strictly better at compute-match by more than two pooled standard errors) **and** both seeds agree in sign on the per-seed comparison. **Asymmetry inherited:** if `U` is the *larger*-vocabulary arm, the threshold is `3ŝ_c`, not `2ŝ_c` — D-C-2's small-vocabulary default applies at every decision point, not only the first.
>
> **(c) Inside the band: nothing happens.** If the reversal condition fails, the byte-matched decision **stands as made** — the confirmation's job was to catch a budget-artifact winner, not to re-litigate closeness, which the tie rule already owned at the decision stage. "Confirmation was close" appears in the receipt as a sentence, never as a decision input.
>
> **(d) Seed split escalates.** If the two seeds disagree in sign at confirmation, the existing seed-split rule extends unchanged: escalate to strategy with both datasets; no unilateral resolution.
>
> **(e) Reversal escalates too.** A satisfied reversal condition does not auto-install `U`: per R-G4h it **returns to strategy with both full datasets** (byte-matched and compute-matched, all curves, all strata). A reversal is information about *what the decision axis should have been*, and that reading is not delegated.

# 5. P-A throughput — a flag, not an intervention

Observed: 713,183,545 bytes durable in 2 h 30 m ≈ **285 MB/h**. Against the 47,632,339,814-byte source cache, a naive linear projection is ≈ 167 h ≈ **7 days** for the full pass. Both workers alive; nothing is wrong; and the projection is naive — it ignores per-source variance, dedup-stage throughput differences, and whatever ramp the first hours contain.

**Request, not a ruling:** the next P-A receipt should carry the agent's **own ETA projection with per-worker throughput**, so the number is owned by the party who can see the pipeline. If the agent's projection materially exceeds ~7 days, say so plainly and include the bottleneck — that is a "surface it" threshold, not a stop. G-TOK's GPU work is not gated on P-A, so the screen this document unlocks proceeds regardless.

# 6. What unlocks, and the order of things

**On the agent's verification of this document (bytes + SHA-256): the A100 base screen is unlocked.** The execution order is unchanged from the handoff: base screen → byte-matched 4-arm × 2-seed G-TOK → decision per D-C-2 → confirmation per SEM-2 → reversal check per SEM-3 → `V` freeze (or escalation). P-A continues in parallel on CPU; P-B waits on P-A + attribution text as ruled. Tripwire meter runs from the first GPU step per the standing rule.

Nothing here is for Mark: both rulings sit under standing authority (the handoff's own text delegated "exact confirmation semantics" to a strategy ruling), and both are reversible on a word before the confirmation stage runs.

---

*Signature block*

**Strategy:** the agent's two questions were the right two — "equal FLOPs" and "confirmation pair" were each doing load-bearing work in the handoff on one sentence of specification. The satisfying part of SEM-2 is that the monotone FLOP-in-V fact turns half the confirmation into a reuse: the protocol gets cheaper by being understood better, which is the good direction for a protocol to move. SEM-3(b)'s inherited asymmetry closes a gap nobody had asked about yet: without it, a larger vocabulary could lose at the decision under a 3ŝ handicap and then win at confirmation under a 2ŝ one, which would make the confirmation a second, easier door into the outcome D-C-2 deliberately weighted against.
**Coding agent:** verify bytes and SHA-256 against the delivery line, then the base screen is yours to run. SEM-1(a)'s dropped-token line and SEM-1(b)'s 10 % projection-divergence stop are the two new receipt obligations; `gtok.confirm.*` seeds enter the registry now but are consumed only if the byte-matched decision produces a pair needing fresh runs. Include the P-A ETA projection with per-worker throughput in the next receipt.
**Mark:** nothing to decide. The next thing you see should be either a G-TOK result or a gate firing — and the pipeline between here and there is now fully specified.
