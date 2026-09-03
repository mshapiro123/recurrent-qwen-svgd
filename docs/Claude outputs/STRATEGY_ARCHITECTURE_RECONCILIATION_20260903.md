# STRATEGY — Architecture Reconciliation: Two Diagrams, One Vocabulary, a Re-cut Build Queue, and an Honest Assessment of the Reasoner

**Date:** 2026-09-03 · **Status:** RECONCILIATION + RULINGS. Sources read as files (PS-1): the coding agent's figure `weft1_architecture_and_build_state_20260902.svg` (26,120 B, SHA-256 `480852a8…acb16` — hash verified byte-exact after staging from the repo), strategy's r2 diagram (`8816ea41…8f86c8`), the build handoff §2/§5/§9, and the build-status matrix.
**Verdict up front:** the two diagrams do **not** disagree about the design. They are drawings of two different things — r2 is *the ratified target*; the agent's is *the target overlaid with what is executable today* — and once that is said, they agree on every ratified surface. Where they look different is where the executable graph contains **four inherited modules the ratified forward pass never names**, plus one placement that drifted. Those are the real findings, and they are ruled below. Two of the agent's "open" items are stale (already resolved by S5/S6 + A1) and one is resolved *by* this document.
**Decisions for Mark:** none blocking. Two rulings carry veto notes (R-1, R-2). The assessment in §5 is opinion, labelled as such.

---

## 0. Plain-language summary

The first thing to settle is what each picture is. Mine draws the model we ratified. The coding agent's draws that same model with a colour code for what has actually been built — green for running, hatched for built-but-not-wired, dashed for absent — and a strip at the bottom showing what `AblationLM.forward` executes today. Read that way they are complementary, and both are adopted: mine as the design of record, the agent's as the build state of record.

Reading the agent's figure against the handoff surfaced something more important than the drawings: the executable graph carries four modules that the ratified forward pass never mentions. A Hadamard expert module sits at the *front* of the model, before the prelude, when the design puts Hadamard experts *inside the core* as the FFN alternative with a parameter-free router; a two-lane mixer sits between the lanes, which once the real callosum lands would be a second channel between hemispheres in a design that promises exactly one; an anchored re-entry bridge re-injects the prompt state into the loop each visit, a Paper-One-era mechanism the WEFT-1 forward does not use; and a read-only long-term memory writes retrieved content into the residual just before the coda. Every one of these is structurally switchable — the agent's bit-identical OFF tests prove the dense anchor is untouched — so nothing is broken. But "switchable" is not "ratified," and a production model must know which of its parts are the design and which are scaffolding. The rulings classify all four: the front Hadamard and the re-entry bridge are retired from the production configuration (kept as registered legacy arms, structurally OFF); the lane mixer is retired when the callosum lands; the long-term memory is kept as a registered, OFF-by-default arm because its write site is outside the loop, which is what the no-injection rule actually protects. That last point is a vocabulary fix worth stating plainly: "no injection" has always meant *no stored content into the recurrent state* — the engram's early-prelude write and a post-loop memory read are outside that boundary, and the rules now say so in one sentence.

The visit schedule also gets written down in one place, because the handoff's pseudocode lost its indentation somewhere between drafts and the two diagrams draw it slightly differently: per block — attention, FFN, lane update, sidecar; per visit — carrier rotor and write, then callosum. The agent's seam will assert that order as a receipt.

On the build order: the queue was logical but had one thing in the wrong slot and two things missing. The combiner belongs with the bicameral block, not three steps later — without it the integrated model cannot produce logits, and with it the dense-equivalence anchor can be tested the day the block lands. The objective stack — the staged-state loss that gives the loop its training signal, the diversity term, the halting head — is absent from the graph and was absent from the queue, and no training run can start without it. And the Hadamard core-expert arm with its matched-dense control is a registered *arm* whose control (the dense FFN) is already executable, so it belongs in the experiment queue, not on the critical path.

Then the honest question: have we assembled the pieces for a smart, small reasoner? My answer is that we have assembled the pieces for a *credible, falsifiable attempt* at one — which is the most anyone can truthfully claim before the first training run. §5 says what that means and what would change my mind.

---

# 1. What each diagram is — both adopted

| | strategy r2 (`8816ea41…`) | coding agent (`480852a8…`) |
|---|---|---|
| draws | the ratified target, one visit in detail | the target overlaid with executable status, plus the live `forward` strip and PF-3 evidence |
| role of record | **design of record** | **build state of record** |
| currency | current (post S5/S6 + A1) | snapshot at `6d00b1fc`: lists C-S5-1/2, C-S6-1/2 and #35 as OPEN and "Q K V O gate up down" (seven paired) — **stale on exactly those points**; an r2 is requested after the agent verifies S5/S6 + A1 |
| agrees on | every ratified surface: embed → prelude (engram after block 1) → h₀ → static K/V + bridge_in → shared core ×K (paired block, lanes, sidecar, rotor+write, callosum) → combine → bridge_out → coda → tied readout; rungs, widths, K curriculum, α_T = c/T, receipts | — |

**Naming:** the agent's "shared recurrent LOOM" and the handoff's `LOOM1` filename are the pre-rename vocabulary; the program is WEFT-1. Harmless, but the r2 figure should say WEFT.

# 2. One vocabulary — code name ↔ handoff name ↔ diagram name ↔ status

| code (`models/ablation_lm`) | handoff | diagram term | ratified? | status |
|---|---|---|---|---|
| `AblationLM` | the model | WEFT-1 | — | bring-up graph |
| tied embedding / readout | §5.1 `embed`, `readout` | tied embedding · tied readout | yes | integrated |
| `front_hadamard` (`ModifiedHadamardExpertBank` + learned router) | **not in the handoff** — §5.7 places Hadamard experts *in the core FFN slot* with the parameter-free occupancy router | "upfront modified Hadamard challenger" | **no (placement drifted)** | **retire from production config → legacy arm FRONT-WHT, structurally OFF** (R-1) |
| prelude blocks | §5.1 `prelude` | Prelude ×P | yes | integrated |
| `CausalTokenEngram` | §5.11 `M_lex` | M_lex engram | yes | integrated (token-ID keys; byte-span = registered ablation) |
| `h0` | §5.1 `h0` | h₀ anchor | yes | integrated |
| static K/V from `h0` (single-stream) | §5.3 | static K/V cache | yes — **shared consensus (S-3, A1)** | integrated single-stream; bicameral form pending step 2 |
| `bridge_in` / `PositionAlignedScratch` | §5.4, §5.12 `bridge_in`, lanes `s[t,k]` | bridge_in · lanes (agent: "scratch lanes") | yes | integrated — **"scratch" and "lanes" are the same object; one word from now on: lanes** |
| `TwoLaneBirkhoffMixer` | **not in the handoff** (§5.6: the callosum is "the only inter-hemisphere channel") | "narrow two-lane Birkhoff mixer" | **no** | **retire at step 4 when the per-band callosum lands** (R-5) |
| `AnchoredReentryBridge` | **not in the handoff** (core_pass re-injects nothing; h₀ enters only through K/V and bridge_in) | "anchored recurrent re-entry" | **no** | **retire from production config → legacy arm H0-REENTRY, structurally OFF** (R-2) |
| `SwapLinear` (μ, δ = UVᵀ, rank 32) | §5.5 | paired (μ ± δ) | yes | standalone; step 2 |
| `BicameralTransformerBlock` + recurrence seam | §5.1 core block, §5.5 | paired full-width block | yes | standalone; step 2 |
| — (absent) | §5.7 Hadamard core experts + `OccupancyRouter` + T12 control | (agent: "sequency expert alternative … absent") | yes, **as an arm** ("FFN *or* Hadamard bank") | experiment queue (R-6) |
| `Cl20Rotor` primitive; learned carrier absent | §5.2 rotor bank J = 8 + single rank-8 write | carrier rotor + one write | yes | primitive only; step 3 |
| `callosum.py` per-band Birkhoff | §5.6 | per-band corpus callosum | yes | standalone; step 4 |
| — (absent) | §5.8 loop sidecar | conditional shared sidecar | yes — **per-lane, shared bank (S-4′)** | step 5 |
| `trajectory_jet_metrics`, `plane_probe_features` (hidden-stream, frozen probes) | §5.9 pooled-lane second-order jet, learned probes | 2-jet geometry | yes (target form) | prototype on hidden stream; per-lane target with step 5 |
| — (absent) | §5.1 `combine` → S-2 unit-circle | final per-band combine | yes | **moves to step 2** (R-7) |
| — (absent) | §5.12 `bridge_out` | bridge_out | yes | step 3 |
| `ReadOnlyLatentMemory` | §2 "retained unchanged: read-only long-term memory with leave-one-record-out" | read-only long-term memory | **retained, not in the forward pass** | **registered arm LTM-RO, structurally OFF by default** (R-3) |
| coda, final norm, `_language_model_loss` (+ z-loss) | §5.1 `coda`, §6 | shared coda · readout | yes | integrated |
| — (absent) | §6 `L_stage`, `L_div`, auxiliaries; halting head | (agent: "both unbuilt") | yes | **step 6, new** (R-8) |
| `accounting.py`, `rng.py`, `observatory*.py` | §9, O-9, §10 | receipts | yes | integrated / harness |

# 3. Rulings

> **R-1 — Hadamard placement.** The ratified Hadamard experts live **in the core FFN slot** (§5.1 step 2, §5.7), addressed by the **parameter-free occupancy router**, as an arm against the dense FFN with the T12 matched-dense control. The executable `front_hadamard` module at the model input is **not a WEFT-1 surface**: it is retired from the production configuration and registered as legacy arm **FRONT-WHT** (structurally OFF; may be exercised only inside a registered contrast). **Catch #35 is thereby resolved for production** — the E×d learned router is not a production tensor; S-1's class (3′) remains bound so C1 can run on the bring-up graph as it stands today. *Veto note: if Mark wants an input-side WHT experiment, it is one line to register; it does not change the core's design.*

> **R-2 — the anchored re-entry bridge.** Re-injecting h₀ into the loop each visit is the Paper-One retrofit mechanism (`p` re-injected); the WEFT-1 forward pass admits h₀ only through the static K/V (the "fixed context") and `bridge_in` (the lanes' initial state). The bridge is retired from the production configuration and registered as legacy arm **H0-REENTRY**, structurally OFF, with a pre-registered reading if ever run: it is a *state-injection* path into the loop and therefore belongs in the same control family as MEM-INJ. The PF-1.5 eligibility rows it occupies become "legacy, OFF." *Veto note: same one-line register if Mark wants it live.*

> **R-3 — the no-injection vocabulary, in one sentence.** *No stored or retrieved content is ever added to the recurrent state — h_A, h_B, or the lanes — inside the loop, except through the ratified paths (lane update, sidecar-to-lanes, the single carrier write, the callosum).* Writes **outside** the loop are governed by ordinary gating rules, not by this prohibition: the engram's gated write at prelude block 1 (§5.11, *before* h₀) and a post-loop memory read (*after* the last visit, before the coda) are both outside it. Accordingly `ReadOnlyLatentMemory` is **registered as arm LTM-RO, structurally OFF by default**, permitted at its post-loop site only, never inside the loop.

> **R-4 — the visit schedule, bound.** Per executed visit k, per core block i: (1) attention — Q from the hemisphere's current state, K/V from the shared cache; (2) FFN (dense, or the Hadamard arm); (3) lane update reading h_A, h_B, engram values; (4) sidecar, per lane, conditional. **Once per visit, after the block loop:** (5) carrier — rotor then the single gated rank-8 write from lanes, per hemisphere, scaled α_T·γ; (6) callosum, once. Then per-visit `combine` → shared coda for step logits in training. The bicameral recurrence seam **asserts this order in a receipt** (`visit_schedule` line: the ordered list of executed module names per visit), so the two diagrams and the code cannot drift apart silently again.

> **R-5 — the two-lane mixer is a second callosum.** §5.6 makes the callosum *the only* inter-hemisphere channel; a lane-to-lane mixer is a second one. At step 4, when the per-band callosum integrates, `TwoLaneBirkhoffMixer` is **retired** (structurally OFF, receipt-marked absent). The C-JAC factor it contributed (exactly 1) drops out; CAL-BW-2's bypass caveat applies to it until then.

> **R-6 — the Hadamard core-expert arm is off the critical path.** §5.1 says "FFN **or** parameter-matched Hadamard expert bank": the dense FFN is a ratified configuration and the executable T12 control. The Hadamard-in-core arm + occupancy router (with its `(m, s)` calibration gate) is built and run as a **registered arm in the experiment queue**, after the first production model exists — not before it.

> **R-7 — combine moves to step 2.** The integrated bicameral block cannot emit logits without `combine`, and the T1 dense-equivalence anchor for the bicameral graph is precisely `δ = 0 ⇒ y = h`. S-2's unit-circle combiner lands **with** the block.

> **R-8 — the objective stack and halting head join the queue.** `L_stage` (per-visit step logits through the shared coda, §6.2), `L_div` with its interior target, the registered auxiliaries, and the halting head (inference-only, joint over both lanes' jets per A1) are absent from graph and queue alike. They are **step 6**; no training run starts without step 6.

# 4. The build queue, re-cut

| step | content | depends on |
|---|---|---|
| 1 ✓ | S5/S6 rulings + A1 | done |
| **2** | bicameral block into the recurrent path · shared consensus K/V cache · **S-2 combine** · retire FRONT-WHT and H0-REENTRY from the production config (structural OFF, registered as legacy arms) · A7 cells, eligibility rows, certificate factors, `visit_schedule` receipt | 1 |
| 3 | learned rotor carrier (J = 8) + single gated rank-8 write per hemisphere + `bridge_out` + fitted retention gauge `r ≥ 0.9` | 2 |
| 4 | per-band Birkhoff callosum on (h_A, h_B), once per visit · **retire the two-lane mixer** | 2 |
| 5 | sidecar per S-4′/S-5: per-lane experts, shared bank, per-lane jets + shared PQ codebook, invocation gate with the calibration/freeze gate, invocation-agreement line | 3, 4 |
| **6** | objective stack: `L_stage`, `L_div`, auxiliaries · halting head · K-curriculum runner · STOCH-K sampler (S2 arm) with its O-9 stream | 2 (L_stage needs combine + coda), 5 (halting reads jets) |
| 7 | certificate topology (#26) · full A7 matrix · first production T14b / OBS-INV receipts | 2–6 |
| exp. | Hadamard core-expert arm + occupancy router + T12 control (R-6) · LTM-RO · FRONT-WHT · H0-REENTRY · KV-PAIR · MEM-OP family · ROW ladder | first production model |

**Is the ordering logical?** Yes, with the three corrections above: it follows data dependencies (hemispheres before anything that reads two of them; lanes and carrier before the sidecar that writes lanes and the write that reads them; combine with the block; objectives before training), it keeps every step a controlled comparison against the live dense anchor (structural OFF at each step), and it puts every *arm* behind the first production model rather than in front of it.

# 5. Have we assembled a smart, small reasoner? — an honest assessment

**What is assembled, and why each piece is there.** *Depth on demand*: a weight-tied core executed K times, with α_T = c/T keeping the composition bounded, a curriculum to 4 and a knob to 8, and a scaling law with an external prior (GRT: most of the gain in the first two visits, degradation beyond the trained horizon) that tells us what to expect. *Two readers of one context*: paired hemispheres whose weights differ by a low-rank δ, reading a shared K/V, exchanging only through a rank-1-per-band bounded channel — diversity as an interior target, not a maximum, with the collapse mode (ρ = ½) understood in closed form. *Persistent scratch*: position-aligned lanes that carry state across visits without ever reading the future. *Procedural memory*: a shared library of low-rank operators consulted only when the trajectory's geometry says the work changed, writing to the lanes, never the carrier — memory as operator, with the killed content-injection mechanism kept as a control. *Sparse input vocabulary*: the engram, supplying pattern content before the loop without spending context tokens. *And the discipline*: every module structurally OFF is bit-identical to a dense transformer, every instrument calibrated on a planted effect before it sees the real model, every claim against a matched-compute control.

**What is not assembled: evidence.** Not one of the seven mechanisms has been shown to help at this scale on this data. The design gives each a mechanism-level reason and a kill-switch; the training run is what converts reasons into findings.

**Where I think the risk actually sits, in order.** (1) *Recurrence gain at 300 M / 10¹⁰ tokens may be modest* — GRT's own numbers say L₁→L₂ is the big step and visits 3–6 are small; a 34-tokens-per-parameter budget is not lavish, and the matched-compute dense control gets 64 % more data. The instrument for this is η_k against reallocation, and the honest branch is that K = 2 is most of the win. (2) *Bicameral collapse* — the hemispheres may converge (ρ̂ → 1) or the disagreement may be decorative; L_div's interior target and the callosum tripwire exist for this, and KV-PAIR is the registered lever if five paired projections cannot sustain diversity. (3) *The sidecar may be idle or uniform* — G-INV's "report an ordinary MoE" branch is the honest outcome if the geometry gate does not discriminate; the per-lane agreement line will say quickly whether private invocation is real. (4) *Complexity on a first from-scratch run* — seven interacting modules is a lot for one model; what makes it survivable is that the dense baseline is one flag away and each step of the queue is a controlled increment, so a bad module is a *finding*, not a failed program. (5) *Data* — English-only, 10¹⁰ tokens, reasoning traces only in a declared final phase: enough to test the mechanisms, not enough to make a strong reasoner by scale alone. That is the correct scope for a from-scratch mechanism study, and it should be described that way.

**My takeaway.** The pieces of a smart small reasoner are on the table, connected in a coherent factorization — *content anchors memory, state chooses computation, operators compose, hemispheres disagree within a bounded budget, depth is spent only when the trajectory says so* — with the controls and instruments to tell us which pieces earn their place. That is the strongest true statement available before training. What would change my mind, in either direction, is the first S2 result on the two headline instruments: the loop marginal gain against reallocation, and the recurrence exponent against GRT's bracket. If both land where the design predicts, the rest of the program is refinement; if the loop gain is flat at matched compute, we will have learned the most expensive fact early, cheaply, and with the receipts to prove it.

---

*Signature block*

**Strategy:** the diagrams were never the disagreement; the four inherited modules were, and nobody had listed them against the ratified forward pass until both pictures were on the table. R-1 through R-5 are the reconciliation; R-6 through R-8 fix the queue. On the big question I have tried to give the answer I would want to read in six months, whichever way the run goes.
**Coding agent:** verify bytes and hash; adopt the vocabulary table (one word: *lanes*); apply R-1/R-2 (retire to legacy arms, structural OFF, registered) and R-7 in step 2; add the `visit_schedule` receipt (R-4); schedule R-5 with step 4; add step 6; update the figure to r2 after verifying S5/S6 + A1 (paired set = five; C-S5/S6 and #35 closed; WEFT naming).
**Mark:** nothing blocking. Veto notes on R-1 and R-2 only. The rest is what you asked for: the two pictures now describe one model, the order has its dependencies in the right places, and the assessment is as honest as I can make it before the data arrives.
