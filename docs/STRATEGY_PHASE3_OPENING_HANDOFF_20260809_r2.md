# Strategy Handoff to Coding Agent — Phase 3 Opening: The Answer-Gap Program (r2)

Date: 2026-08-09, revision r2 same day. Governing: Phase 3 realignment r2 (Drive `1189Ah2swUw6DiK778VffygPAqeOhkQbe`), Guardrail Doctrine (Drive `1R40TawfW-ZJcec4dkRnxGir9c3v4a5IP`), the banked Phase 2 chain through the confirmation (commit `7f5743b4`). Companion figure: `PHASE3_TRAINING_SIGNALS_20260809.svg` (published alongside).

**r2, ratified by Mark 2026-08-09:** folds in amendments A1–A7 from the dialectical review (Drive `1_CeYQj_rUkau_clRzW5R7CFizCZmYjoL`) — claim coherence via pooled accounting, pinned estimators for χ_max and π, the false-stop tax on the battery floor, the verifier mask on distillation, sealed CONFIRM battery halves, the ridge-probe prior on the falsifier, and deltas reported beside ratios. Supersedes Drive `1oIHPwaZff6n1YvqwrHTI2EXWTl6TM7n_`. Amendment markers **[A1]**–**[A7]** appear where each binds. This is the phase's opening charter: aims, rationale, formalism, sequence, pseudocode, guardrails, and deliverables. Protocol locks per stage follow the usual pattern — this document tells you where the program is going and why, so the locks you draft aim at the same place.

---

## 1. The goal, stated once and numerically

**Build a better 0.5B model.** Specifically: the sidecar-augmented Qwen2.5-0.5B (496M stored base + 1,184,917 trainable sidecar parameters, ≈ 497M inference-active — "the 0.6B" colloquially, and the paper will state the exact count) must answer questions **measurably better than the frozen base**, with the improvement denominated against the 14B teacher:

```
gap_closed = (A_augmented − A_base) / (A_14B − A_base)
```

per battery, with document-bootstrap confidence intervals. Phase 2's result in this currency is ≈ 0 by construction — the system was trained under a preservation contract. Phase 3 exists to make this number positive, replicated, and eventually confirmed on the sealed partition. Training the recurrent bridge sidecar to move this number **is Paper Two's results section**. Everything below serves that sentence.

**Claim coherence [A1].** The confirmed headline claim — a better model — requires two things jointly: gap_closed > 0 with CI excluding zero on at least one target battery, **and** the pooled accounting across all batteries (target + general, weighted by item count) non-negative. A model that gains on math while quietly losing general knowledge is not a better model and will not be claimed as one. This binds the *claim*, not the training: exploration above the battery-floor cliff remains unconstrained. **[A7]** The raw delta A_augmented − A_base is reported beside gap_closed everywhere the ratio appears, and any battery whose teacher−base denominator CI touches zero reports delta only.

## 2. Why this is believed possible (the rationale, with receipts)

Four measured facts, all banked in Phase 2, jointly define the opportunity:

1. **The reach exists.** Oracle-aimed writes through the existing bounded bridge flip **60.85%** of student-wrong/teacher-right positions at the measured safe radius (c = 0.15, p99 cap), with collateral near zero under perfect aim. The Phase 1 oracle ceilings on final answers were **+3.91 to +5.61 points**. The channel physically carries several points of answer improvement.
2. **The state exists.** The trained flow moves the scratchpad **1.36 nats** toward the 14B teacher's future distribution. The information needed for aiming is being computed inside the forward pass already.
3. **Aim was never trained.** Phase 2 supervised the bridge with acceptance and preservation losses only. The oracle first-order direction — the strongest signal ever measured in this program (98.4% flip rate in the compatible quartile) — was used as a curriculum weight and never as a training target. It is barred from *inference*, not from *training*. Training the bridge to reproduce oracle aim is the same move EAGLE makes when it trains its head on teacher features: privileged signal at training time, learned generalization at inference time.
4. **The trainer works.** Staged curricula, directional gradient-share contracts, calibration-derived weights, the doctrine's rule inventory. The last campaign ran 80,000 steps with zero spurious stops. The optimization craft transfers whole.

The honest risk, stated with the rationale: Stage A proved *unaimed* writes are harm-dominated, and a trained aim is somewhere between oracle-perfect and unaimed. The entire phase keys off measuring where. That is Stage P3.3, the falsifier, and it is deliberately early and cheap.

## 3. Objectives (hierarchy, with the anti-trap clauses built in)

- **O1 — Aim capture (the falsifier).** Measure the *aim-capture ratio* π: the trained bridge's flip rate on held-out student-wrong/teacher-right positions, divided by the oracle flip rate on the same positions, at matched collateral accounting. **[A2]** π's estimator is pinned in the P3.3 lock in the same clause as its readings: held-out audit slice, flip-rate ratio at matched clamp and gate accounting, both seeds pooled, bootstrap CI. Exploration reading, not a gate: π ≥ 0.25 means the ceilings are a roadmap and the phase proceeds at full speed. π < 0.05 after the iteration budget means the approach has found its boundary and strategy writes the boundary memo. Between, one preregistered iteration on features/capacity, then re-read. **[A6]** Before P3.3 spends a training session, a CPU-only ridge/linear probe from cached (h_p, S) to cached d* on the training stratum reports the achievable cosine alignment — a forecast for π that gates nothing. It may run inside P3.2's session and is the phase's first creativity slot, pre-spent.
- **O2 — Gap closed (the phase currency).** gap_closed > 0 with CI excluding zero on the DEV battery, reported per battery and per recurrent depth K, with raw deltas beside every ratio **[A7]** and the general-knowledge component's trailing trend beside the target batteries **[A1]**. This is the number the paper carries, under the §1 claim-coherence requirement.
- **O3 — Acceptance retained as byproduct.** EAL tracked as telemetry. No acceptance gate exists in Phase 3. A model whose final distribution moves toward the teacher accepts longer drafts for free.
- **Anti-trap clauses (binding on every protocol drafted in this phase):** every threshold carries an annealing schedule or a named revisit condition. No confirmation gate is written until exploration shows life. Each report-back includes one **creativity slot**: a design variation outside the locked plan, proposed by either lane, runnable as a cheap probe without an amendment cycle — the doctrine's cliffs still apply to it, nothing else does.

## 4. Formalism

**Notation.** Frozen base f with residual stream h at the bridge's injection point (the same point the perturbation studies characterized). Sidecar: initializer I, flow F (K ≤ 4 iterations over scratchpad S ∈ R^{8×128}), bridge B with gated write Δh, draft head D, control state c. Teacher T = Qwen2.5-14B (32B per cascade where lattice provides it).

**Oracle direction (training-time only).** For position p with student top-1 token y_s and teacher top-1 token y_t ≠ y_s, define the margin m(h) = z_{y_t}(h) − z_{y_s}(h) on base-model logits, and the oracle direction

```
d*(p) = ∇_h m(h_p) / ‖∇_h m(h_p)‖
```

computed on the frozen base with sidecar inactive (stable, cacheable). The validated aimed write is Δh* = g · min(RMS(h_p), 0.5509) · d* with g the gate scalar. v1 caches d* per selected position (bf16, 896-dim, ≈ 1.8 KB per position). Known limitation, accepted for v1 and revisited in P3.5: as the bridge opens, the true optimal direction drifts from the sidecar-inactive gradient — a refresh pass on the current model is the named revisit.

**The Phase 3 loss set.**

- **Aim loss (new, primary for the bridge).** On flip-candidate positions (teacher-right, student-wrong, high-teachability): `L_aim = 1 − cos(u_p, d*(p))`, where u_p is the bridge's pre-gate write direction. Direction only — magnitude remains governed by the clamp and gate.
- **Gate supervision (new).** The gate is trained, not correlation-mined: binary target open on flip candidates, closed on student-correct positions, `L_gate = BCE(gate_p, open_p)`. This is the router question of Phase 2, now answered with training-time oracle labels instead of post-hoc feature hunting.
- **Answer distillation (primary for the system).** `L_KL = KL(p_T^τ ‖ p_{f+sidecar}^τ)` on the top-K = 128 lattice with tail mass, temperature τ = 1 initially [schedule open]. `L_CE` on teacher-verified answers where the lattice carries them. **[A4] Verifier mask:** both losses are masked (or sharply downweighted at the lock's discretion) on positions the lattice's verifier marks teacher-wrong. The student is never trained toward a verified teacher error — without this mask, L_KL teaches the teacher's mistakes on exactly the positions where L_CE and preservation fight it, the two-masters conflict that killed the Phase 2 joint pilot.
- **Preservation, annealed.** `λ_p(t) · KL(p_f ‖ p_{f+sidecar})` on student-correct positions, with λ_p on the annealing schedule below — pressure that relaxes as competence is demonstrated, never a fixed wall.
- Flow loss frozen-in (flow parameters frozen in P3.3, optionally unfrozen at 0.1× LR in P3.5).

**The annealing controller (the memo's "anneals instead of walling," made mechanical).** Every evaluation window, compute on the rolling DEV audit: aim capture π, collateral rate χ (student-correct positions flipped wrong by open-gate writes), and the battery floor check. Advance at most one rung per window:

```
rung:        0      1      2      3
requires:    —      π≥.10  π≥.25  π≥.40   (and χ ≤ χ_max at each)
gate ceil γ: 0.02   0.08   0.20   0.50
λ_preserve:  1.0    0.5    0.2    0.05
```

Rung table values are v1 defaults, revisit-labeled. **[A2]** χ_max is not left symbolic: the P3.3 lock assigns it a calibrated value derived from the measured collateral of *oracle* writes at the matched gate ceiling plus a stated margin, with its estimator (positions, window, seed pooling) in the same clause. No threshold enters any Phase 3 lock without its estimator in the same sentence. Demotion happens only on the battery floor (below), never on noise. The controller is code, not judgment: its state and transitions appear in every evidence record.

**Depth as reasoning.** Answer losses are applied at every loop k with weights w_k = k/K (deep supervision), and the phase's standing curve is accuracy-versus-K per battery — the program's founding measurement, finally on working machinery. Cost note for the P3.4 lock: supervising every k multiplies coda-forward cost by K, so the lock may subsample k per step (sampled k with w_k weighting, unbiased in expectation).

**The floor that never anneals (the doctrine's fourth cliff).** A reference battery (Paper One's Tier-1 style plus an ARC slice) with a hard bound: augmented-model score more than 3 points below the frozen base's score on two consecutive evaluations → stop. **[A3] The false-stop tax is paid at P3.1:** the reference battery is sized so that the probability of this floor tripping on two consecutive evaluations under the null (no true change) is below 1 in 10,000, computed from the battery's binomial noise at the base's measured score. If the natural sets are too small, the floor widens or the battery grows — the number is chosen, stated, and defended before the first training step. Everything else in the phase is telemetry, warning, or schedule. Retention is additionally tracked as a trailing-window trend in the standard curves **[A1]**, so damage-then-recovery transients are visible and distinguishable from decay.

## 5. Sequence

**P3.0 — Research sweep** (strategy lane, parallel, no GPU). Direction-supervised interventions, future-token and hidden-state distillation for answer quality, self-correction training, latent lookahead. One memo before the P3.3 lock. Any finding that changes the loss set enters through the lock, not mid-run.

**P3.1 — Currency assembly** (coding, mostly CPU + one eval-only GPU session). Assemble the batteries: ARC-Easy/Challenge from Paper One's infrastructure, a grade-school-math set, a code set [strategy will name exact sets in the P3.1 lock]. **[A5] Seal the confirmation partition at assembly:** every battery splits into a DEV half and a CONFIRM half. The CONFIRM halves are hashed and sealed *before any model — base, teacher, or augmented — is scored on them*, under the standard read-once lease. P3.6 spends CONFIRM once. This cannot be done honestly later. **[A3]** Battery sizing satisfies the false-stop tax computation in §4. Score the frozen base and the 14B teacher on the DEV halves with the same reader protocol. Build the gap_closed arithmetic with document-bootstrap CIs, deltas beside ratios **[A7]**. Deliverable: the reference table every later run is judged against, plus the CONFIRM seal hashes. Nothing trains before the currency exists.

**P3.2 — Oracle cache** (one A100 session). Compute and cache d*(p) and gate labels over the high-teachability stratum of the 181,969-anchor training pool (teacher-right/student-wrong selection from the lattice's existing quantities — the stratum was built for this and never used). Storage ≈ 1–2 GB at bf16. Score-blind with respect to all batteries. Hash-ledgered like every cache. **[A6]** The session closes with the CPU ridge-probe (cached (h_p, S) → cached d*, achievable cosine alignment reported with CI) — the forecast for P3.3's π, gating nothing.

**P3.3 — Aimed-writeback pilot (the falsifier; one A100 session).** Train bridge + gate + control only (flow and draft head frozen), loss set {L_aim, L_gate, λ_p·preserve} under a calibrated directional contract, both seeds. Read π and χ on a held-out audit slice, per the pinned estimators of §3/§4 **[A2]**. This is the cheapest experiment that can kill or fund the whole phase, which is why it runs third and not tenth. Exploration rules: doctrine cliffs only, everything else measured.

**P3.4 — Answer-distillation main run** (A100, the phase's center of mass). Staged: aim-pretrained bridge from P3.3 → joint {L_KL, L_CE, L_aim, L_gate, annealed preserve} with the annealing controller live → accuracy-versus-K and gap_closed on the DEV battery at every checkpoint window. The 20-point-curve pattern from the scaled continuation carries over: every run is a curve, not a verdict.

**P3.5 — Iterate with the annealing and creativity clauses** (report-back driven). Named levers, in rough priority: oracle-direction refresh on the current model, flow unfreeze at 0.1×, bridge capacity (rank or multi-point injection), data expansion via another teacher pass, per-loop gate schedules. Each report-back's creativity slot may jump this queue with a cheap probe.

**P3.6 — Confirmation on the sealed partition** (one shot, only when earned). When exploration shows a stable positive gap_closed across seeds, strategy drafts the confirmation preregistration with bands set from measured effects. **[A5]** The partition spent is the CONFIRM battery halves sealed at P3.1 — the partition that can adjudicate a battery claim. EVAL-E remains sealed until this same moment and is spent only for the acceptance-byproduct and retention claims it was built to carry. Both leases are read-once. The confirmed claim obeys the §1 claim-coherence requirement **[A1]**, and Paper Two's results section is whatever the confirmation says.

## 6. Pseudocode

**Oracle direction cache (P3.2):**

```python
@torch.no_grad_off()  # gradients required w.r.t. h
def oracle_direction(base, tokens, pos, y_teacher, y_student, inject_layer):
    h = base.forward_to_layer(tokens, inject_layer)      # sidecar inactive
    h_p = h[:, pos, :].requires_grad_(True)
    logits = base.forward_from_layer(h_swap(h, pos, h_p), inject_layer)
    margin = logits[:, pos, y_teacher] - logits[:, pos, y_student]
    (g,) = torch.autograd.grad(margin.sum(), h_p)
    d = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return d.to(torch.bfloat16)                          # cache with position key
```

**Aim + gate training step (P3.3):**

```python
def p33_step(batch, sidecar, cache):
    S = sidecar.flow(sidecar.init(batch.h0))              # flow frozen
    u, gate = sidecar.bridge.pre_gate_write(batch.h0, S)  # u: direction proposal
    d_star, open_label = cache[batch.pos_keys]
    flip = batch.is_flip_candidate                        # teacher-right, student-wrong
    L_aim  = (1 - F.cosine_similarity(u[flip], d_star[flip], dim=-1)).mean()
    L_gate = F.binary_cross_entropy(gate, open_label.float())
    L_pres = lam_p * kl_div(batch.base_logits, model_with_write(batch, u, gate))[batch.student_correct].mean()
    return w_aim*L_aim + w_gate*L_gate + L_pres           # weights by calibration, shares audited
```

**Verifier-masked distillation losses (P3.4) [A4]:**

```python
def distill_losses(batch, student_logits, lattice):
    ok = ~lattice.teacher_wrong[batch.pos_keys]           # verifier labels, already in the lattice
    L_KL = (kl_topk(lattice.topk_dist, student_logits, tau=1.0) * ok).sum() / ok.sum()
    ce_mask = lattice.has_verified_answer[batch.pos_keys] & ok
    L_CE = F.cross_entropy(student_logits[ce_mask], lattice.answers[ce_mask])
    return L_KL, L_CE                                     # never distill a verified teacher error
```

**Annealing controller (P3.4, evaluated per window):**

```python
def anneal(state, audit):
    pi  = audit.trained_flip_rate / audit.oracle_flip_rate
    chi = audit.collateral_rate
    if audit.battery_floor_breached_twice():   # the one cliff — stop, do not demote silently
        return state.stop("battery_floor")
    r = state.rung
    if r < 3 and pi >= REQ_PI[r+1] and chi <= CHI_MAX[r+1]:
        r += 1                                  # at most one rung per window
    state.apply(gate_ceiling=GAMMA[r], lam_preserve=LAM_P[r], rung=r)
    state.log(pi=pi, chi=chi, rung=r)           # controller state in every evidence record
    return state
```

**Gap evaluation (every checkpoint window):**

```python
def gap_closed(aug_scores, base_scores, teacher_scores, docs):
    g = {b: (aug_scores[b] - base_scores[b]) / (teacher_scores[b] - base_scores[b])
         for b in BATTERIES}
    ci = {b: doc_bootstrap_ci(aug_scores[b], base_scores[b], teacher_scores[b], docs, n=10_000)
          for b in BATTERIES}
    return g, ci   # reported per battery and per K
```

## 7. Diagram

`PHASE3_TRAINING_SIGNALS_20260809.svg` (house template) shows the three columns: the frozen base with the injection point, the trained sidecar (initializer → scratchpad → flow ×K → gated clamped bridge → writeback, with the draft head as the secondary path), and the training-time-only signal stack (canonicalized teacher states → flow targets, top-K lattice → KL, oracle gradient → aim loss, verified answers → CE, and the annealing controller driving the gate ceiling and preservation weight). The gap_closed formula sits in the footer. Panel captions carry layer ranges and parameter counts per the figure standard.

## 8. Guardrails (doctrine-compliant rule inventory, sketch for the P3.3 lock)

| Rule | Class | Grounding | Named cliff |
|---|---|---|---|
| Non-finite loss/state | stop | absolute | garbage training |
| Base/frozen-hash mutation | stop | absolute | corrupted lineage |
| Battery floor (−3 pts vs base, 2 consecutive; battery sized per the A3 false-stop computation) | stop | static reference (frozen base on fixed battery — a true constant), in-flight by design | quality damage escaping into a claim |
| CONFIRM-half or EVAL-E contact before P3.6 | stop | absolute | invalidated science |
| Gradient explosion (10× trailing-100 median, 3 consecutive) | stop | trajectory | garbage training |
| Directional share contract (primaries ≥ 50%, aux ≤ 25%) | warn → stop (two-tier) | matched estimator | objective identity |
| π, χ, rung, gap_closed, EAL, retention | telemetry | — | — (gauges, not brakes) |

Everything not in this table launches as observe-and-log. The annealing controller is not a guardrail — it is the experiment.

## 9. Deliverables and report-backs

P3.1: reference-score table (deltas beside ratios) + battery hashes + CONFIRM seal hashes + the A3 false-stop computation. P3.2: cache hashes + stratum counts + the ridge-probe alignment forecast. P3.3: π and χ with CIs per the pinned estimators, both seeds, the falsifier read, and the next creativity slot (the first was pre-spent on the probe). P3.4 onward: per-window curves (gap_closed and delta per battery per K, general-knowledge and retention trailing trends, π, χ, rung trajectory, EAL as telemetry), standard evidence-record discipline throughout. Strategy responds to each report-back with the next lock. Do-not-claim, standing: no gap_closed claim from exploration numbers, no "better model" claim without the §1 pooled accounting, no inference-time use of any oracle quantity, no CONFIRM or EVAL-E contact before P3.6, and π is an audit ratio, not a deployment property.

## 10. Plain-language summary

The program is now pointed at its real target: a 0.5B model that answers better than it did, with the 14B teacher as the yardstick. The plan uses everything already built — the scratchpad that anticipates the teacher, the safe write channel with several points of measured headroom, the training discipline that finally works — and adds the one thing never tried: directly teaching the write channel *where to aim*, using the aiming signal validated months ago, plus teaching the gate *when to fire* with real labels instead of guesswork. Safety rails loosen on a published schedule as skill is demonstrated, with exactly one unmovable floor: never meaningfully worse than the base model on the reference battery. The sequence is built to fail fast if it must — the third step is the cheap experiment that tells us whether trained aim captures enough of perfect aim to be worth the campaign, and a nearly-free probe forecasts its answer before we even run it — and to leave room for invention if it can, with a standing slot in every report for ideas outside the plan. The currency is one number: what fraction of the distance to the 14B has been closed. Four honesty clauses from the adversarial review of this plan are built in: the model only gets called better if it is better on the whole, not just where we aimed. Every tripwire states exactly how it is measured, so we never again stop a healthy run over an ambiguous number. The final exam questions are locked in a vault before anyone — including the models — ever sees them. And the student is never taught the teacher's own mistakes. When the number is stable and positive, it takes the one-shot exam on the sealed data, and that result is the paper.