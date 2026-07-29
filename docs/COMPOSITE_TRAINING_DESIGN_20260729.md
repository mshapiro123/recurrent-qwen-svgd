# Composite Training Design — One Router, Two Axes, Four Stages

Date: 2026-07-29. Strategy lane, ratified by Mark 2026-07-29. This memo is the governing forward design for the composite lane and supersedes the program shape of concept-note section 6 for that lane. It does not modify DC1 as issued (STRATEGY_TO_CODING_AGENT_DC0_BANK_DC1_AUTH_20260729.md, Drive `1Yu6EUsbFb9Z4tA2n_2F6moeM-EGyxNVK`) — DC1 is stage A of this design, verbatim. Companion figure: `composite_architecture_20260729.svg` (paper-grade; left panel vertical axis, right panel one decision episode, bottom strip the stages). Locked constants in this memo carry Mark's date and are not reopened by markup.

## 1. The unified control principle

The composite has one learned router, and it lives on the horizontal axis. Per-position vertical routing is retired — not the vertical loop, the vertical *router*. The evidence trail: D0 (disagreement-labeled routing failed), the causal allocation audit (a deployable per-position signal for in-place depth is worth +2 positions in 199,529; the substrate punishes every routing error at 3.5 hurts per help), and DC0 (the harm asymmetry replicated out of sample). The two mechanisms are therefore not independent and not separately routed:

- **k — the number of appended latent slots at a position — is the policy.** Learned per position, decided by control tokens, capped at 3 (locked, Mark 2026-07-29).
- **L — the vertical loop count inside every pass — is a setting.** Fixed globally per configuration: L = 1 through stage C, L = 2 at stage D. Every pass, at real positions and latent slots alike, runs prelude → recurrent block × L → coda.

Compute per position: (k + 1) · (12 + 12L) layer-applications. One fine-grained learned dial, one coarse configured dial.

The vertical loop's continuing value, stated so the demotion is not misread as retirement: the block is weight-tied, so L is parameter-free depth — each horizontal step can be made deeper at zero parameter cost, which a horizontal-only design on the uncut model (plain Coconut) cannot do. At L = 3 a single pass has 48 effective layers, the per-pass depth of Qwen2.5-14B. The vertical actuator is also the proven half of the program (T1-lite: 1,024/1,024 exact depth, 5,632/5,632 overrides); what failed twice was per-position *when*, which this design removes. And the hybrid gets a registered falsification test rather than an assumption — section 5's stage D shape test.

## 2. Decision points and the think-token signal

The control signal is never injected as an input. It is read as an internal output at every computed state: after position t's normal forward, the continue/stop logits at t decide whether slot 1 opens; after each slot computes, the same read at that slot decides whether the chain extends or the token is emitted from there. The slot is both the thought and the assessment of whether thinking is done — the latent analog of a reasoning model emitting its own end-of-think marker, and the resolution of the concept-note section 5.3 question (the think pair migrates into latent space as a *readout convention*, not as injected tokens). Grounding: the frozen-probe result showed the control signal emerges during computation, not before it, so the first decision comes after t's forward, never pre-computation.

Two structural guardrails come free. Stop is the default decision, and stop-at-t is the k = 0 path — bit-identical to the plain drafter, so any position the controller declines to think about is untouched by construction (D0's accepted-position damage mode is architecturally impossible there). And the control tokens remain masked from visible generation under the T0 contract; training their rows changes no visible k = 0 behavior.

Chain mechanics as in M7: fed state through the identity bridge at raw scale (DC0's one unambiguous interface lesson), slots attend over the entire prefix, transient eviction after emission with the position-id and downstream-identity assertions of the markup addendum.

## 3. Teacher policy: ladder, not mixture (Mark, 2026-07-29)

**The 7B is the training teacher through stages A–C.** Full 14B-target training is a named later phase, not a parallel signal. Reasoning, recorded because it will be questioned:

1. *Label actionability (the D0 lesson applied to teacher choice).* The recoverable fraction of a teacher's rejections shrinks as the teacher gap grows — the storage boundary was already ~79 percent against the 7B. Continue-labels derived from the 14B would carry a larger unreachable share, rebuilding D0's misalignment. The 7B maximizes actionable labels, which is what stages B and C consume.
2. *Naive mixing injects measured contradiction.* On approximately 16.5 percent of 7B rejections the 14B endorses the drafter's loop-1 token. Mixing exact-match signals gives that entire set contradictory supervision.
3. *Deployment is pairwise.* Spec-dec acceptance is defined against one target model; each phase names one teacher and is scored against it.

**The 14B's three standing roles before its phase arrives:**

1. **Crossover scoreboard.** Every stage's evaluation carries a descriptive 14B-scored column (post-processing where the cache covers the partition; one cached pass where it does not, commissioned per stage at markup). This watches generalization toward the larger teacher and gives the teacher-shift hypothesis its first test on a *trained* composite.
2. **Label-cleaning referee (optional, decided at the stage-C preregistration).** Positions where the 7B rejects but the 14B endorses the drafter are teacher-noise candidates, excluded from the continue class — the bank-audit's near-tie exclusion, operationalized with the second teacher as referee rather than competitor. This is the one endorsed form of signal mixing.
3. **Distillation-phase target.** After the composite qualifies against the 7B (stage C banked positive), the training signal switches to the 14B — and the signal form switches with it, from greedy exact-match to distribution distillation (KL against teacher logits), which carries more information per token and avoids sparse noisy labels against a much stronger teacher. Depth arithmetic is favorable: the 14B is exactly three loops of per-pass depth, inside the k ≤ 3 envelope. This phase gets its own preregistration and is not scoped further here.

## 4. The stages

Each stage's labels are measured on the substrate the previous stage built — the sequencing rule that closes D0's failure class. No stage starts before the previous stage's verdict is banked.

**Stage A — interface (= DC1, already issued; nothing changes).** Bridge-only, identity init at the preflight-selected scale, forced k = 1, teacher-CE at the slot readout. Question: can forced append be made safe? RG-4/RG-11 green before the loop; one registered pass on EVAL-C; three-way decision mapping as issued.

**Stage B — capability.** Trainable set: bridge, plus minimal adapters only if its preregistration argues for them (open for markup). Mixed forced k ∈ {0…3}, teacher-CE, on fresh dev material. Then the load-bearing measurement: the forced-k utility ledger (helps/hurts per position per k) computed **on the stage-B checkpoint** — these are stage C's labels, causally actionable by construction because they describe what this exact substrate does at each k. The λ grid for stage C is chosen from this ledger, not a priori. Go/no-go: do helps materialize beyond copy-through? If the trained interface only learns to reproduce k = 0, there is nothing for a controller to allocate, and the lane ends cheaply with a receipted negative.

**Stage C — controller.** Trainable set: control rows only; the interface is frozen (no joint fine-tune in the first pass — locked, Mark 2026-07-29; moving the substrate under the labels is how D0 went wrong). Targets: return-to-go continue/stop per decision point with registered cost λ — continue at slot j only if some deeper slot's match beats stopping now by more than the compute penalty; shallow ties stop. Supervised throughout, no RL; credit assignment is handled by the label construction. Class weights measured from the ledger (the P0 lesson: no inverse-frequency defaults). The full T1 forcing and override battery proves the actuator; the policy-level guardrail from the bank-audit precommitments binds (deployed policy's match on baseline-accepted positions within a stated margin of the plain drafter). Optional referee exclusion per section 3.

**Stage D — depth, and the vertical loop's own trial.** L = 2 inside every pass; the full RG battery reruns at the new configuration per the integration design's standing rule. The registered question is the iso-compute shape test: **(k = 2, L = 1) versus (k = 1, L = 2), both exactly 72 layer-applications per position** — does depth-per-step or number-of-steps buy more at fixed compute? Pre-stated readings: more-steps dominates everywhere — the vertical loop retires from the composite on evidence, and the design simplifies to horizontal-only; deeper-steps wins on any stratum — the hybrid is vindicated and L becomes a real second dial; mixed — both knobs are real and the distillation phase chooses shape per budget. Either outcome is a Paper Two result. The pinned third loop (L = 3) is not touched here; it remains gated on the prior pin.

## 5. Fallbacks and standing relations

- If stage A fails its bands: transient append retires and D1 (utility-labeled in-place) resumes, per DC1's decision mapping — the RTG label machinery transfers back to the in-place actuator unchanged.
- If stage B's go/no-go fails: same reversion, with the stage-B ledger banked as the append actuator's capability bound.
- Persistent scratchpad, RG-12, GRAM/width, retrieval-direction: pins unchanged. MoE exotic routers: retired (rung 0 remains the outstanding closure measurement).
- Partition discipline throughout: dev slices reusable within a stage, evaluation slices frozen and read-once, manifests hashed, no partition ever reused across a decision boundary.

## 6. Open for markup

1. Stage B trainable set: bridge-only versus bridge + R16-class adapters (my lean: start bridge-only; widen only if stage B's helps stall, as a registered amendment).
2. The λ grid's range and resolution, once the stage-B ledger exists.
3. The referee-exclusion decision at stage C lock.
4. Whether the stage-B 14B crossover column uses the existing calibration cache or commissions one fresh pass on the stage-B evaluation slice.
