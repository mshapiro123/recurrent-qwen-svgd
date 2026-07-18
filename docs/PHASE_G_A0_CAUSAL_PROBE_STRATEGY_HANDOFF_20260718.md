# Phase G-A0 and Forced-Injection Causal Probe

**Date:** 2026-07-18
**Audience:** strategy and deep-research review
**Program:** recurrent Qwen guided stochastic width
**Final status:** `blocked_no_authorized_successor`
**Decision:** close the current additive re-entry injection design

## Executive Summary

Phase G-A0 asked whether a target-conditioned posterior could impose a selected
valid latent trajectory on a repaired, frozen recurrent Qwen keeper and thereby
teach a stochastic prior. The corrected experiment used repeated identical
prompts paired with different valid selected targets, fixing the
non-identifiability of the first guided-width curriculum.

Both preregistered training arms failed posterior control:

- KL `0.001`: posterior selected-target fidelity `22.64%`, switching `4/32`.
- KL `0.0001`: posterior selected-target fidelity `23.58%`, switching `4/32`.

These values were effectively at the target-blind null. The posterior and prior
were nearly the same terminal decision function despite nonzero gradients,
separated latent statistics, no measured posterior collapse, and an injection
residual that grew during training.

The one authorized follow-up then multiplied the learned posterior residual at
inference by `1, 3, 10, 30, 100` on both preserved checkpoints. This established
the missing causal distinction:

- the additive residual **can** perturb terminal outputs at sufficiently large
  magnitude;
- it does **not** steer those outputs toward the selected valid target;
- output switching rises only as validity and target fidelity deteriorate.

The preregistered verdict is therefore `NO-CHANNEL`, more precisely:

> The current additive high-level re-entry route is not a useful
> magnitude-responsive target-control channel. It has raw causal influence at
> high dose, but that influence is destructive rather than target-aligned.

No same-route scale increase, additional KL/seed/duration sweep, coverage run,
selector, per-trajectory halting, particle method, or SVGD continuation is
authorized from these checkpoints. A future successor would require a new
conditioning route and a new preregistration.

## 1. Scientific Question

The original recurrent model produced one deterministic latent trajectory:

```text
input -> recurrent loop -> learned depth -> one latent trajectory -> answer
```

The GRAM-inspired extension proposed a distribution over recursive trajectories:

```text
input -> conditional latent trajectories -> multiple valid candidates
      -> vote, confidence, verifier, or process reward model
```

The early stochastic/SVGD work did not implement GRAM's essential
target-conditioned posterior teaching a conditional prior. It therefore tested
noise and geometric diversity, not guided stochastic reasoning.

After repairing loop closure, prelude re-injection, and bridge gradient flow,
Phase G returned to the original width question with:

- a frozen, working deterministic recurrent keeper;
- a learned conditional prior;
- a target-conditioned posterior available only during training/evaluation;
- per-transition gold chain states;
- high-level stochastic injection at the recurrent re-entry state.

The minimum causal requirement was posterior target control: changing only the
selected valid chain for an otherwise identical prompt must change the
posterior-guided terminal choice in the corresponding direction.

## 2. Why the Corrected A0 Was Necessary

The first guided-width training set contained only one selected chain per base
problem. That design could not distinguish:

- target-conditioned control;
- prompt memorization;
- generic stochastic variation;
- ordinary answer validity.

The corrected A0 generator enumerated distinct valid exact-depth terminals for
each base problem and emitted repeated prompt variants. Within a group, table,
start state, depth, prompt, and valid terminal set were identical. Only the
selected terminal and its valid intermediate chain changed.

### Frozen data

| Surface | Base prompts | Target variants | Multi-target groups |
|---|---:|---:|---:|
| Training | 512 | 1,899 | 512 |
| Held-out posterior control | 32 | 106 | 32 |

Train and held-out base-problem IDs were disjoint. Training sampled uniformly
over base problems and then over target variants so prompts with larger valid
sets did not dominate the dose.

## 3. Model and Training Boundary

### Deterministic keeper

```text
SHA-256
0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f
```

The repaired recurrent block, prelude, bridge substrate, and coda were frozen.
Only these Phase G parameters trained:

- `phase_g_prior_head.*`
- `phase_g_posterior_head.*`
- `phase_g_injection_scale`

The posterior received the current recurrent state plus the gold next-state
embedding. The prior received only the recurrent state. A sampled latent was
projected through a fixed orthonormal projection and added to the high-level
re-entry state.

### Shared settings

| Setting | Value |
|---|---:|
| Steps | 1,000 |
| Trajectories during training | 4 |
| Learning rate | `1e-4` |
| Initial injection scale | `0.001` |
| KL balance | `0.8` |
| EMA decay | `0.999` |
| Precision | bfloat16 |
| Sampling | base-problem uniform |
| Seed | `20260718` |

At every step, training asserted zero gradients on the frozen substrate. Both
arms recorded 1,000 successful assertions and unchanged deterministic-lineage
hashes.

## 4. A0 Preregistered Gate

The gate was locked before corrected multi-target training:

| Criterion | Requirement |
|---|---:|
| Repeated multi-target groups | `>=32` |
| Posterior selected-target fidelity | `>=0.60` |
| Posterior-minus-prior fidelity lift | `>=+0.15` |
| Paired one-sided sign test | `p<=0.05` |
| Posterior switching groups | `>=24/32` |

The primary arm used KL `0.001`. A single confirmation at KL `0.0001` was
allowed only after the primary blocked. If both blocked, coverage and all
downstream width machinery remained closed.

## 5. A0 Results

### Primary arm: KL `0.001`

| Metric | Posterior | Prior | Reading |
|---|---:|---:|---|
| Selected-target fidelity | 24/106, `0.2264` | 25/106, `0.2358` | `-0.0094` |
| Group selected-target rate | `0.2811` | `0.2941` | `-0.0130` |
| Switching groups | `4/32` | not gating | required `24/32` |
| K=1 validity | `0.7453` | `0.7830` | `-0.0377` |
| Paired target result | helped 1 | hurt 2 | tied 103, `p=0.875` |

### Confirmation arm: KL `0.0001`

| Metric | Posterior | Prior | Reading |
|---|---:|---:|---|
| Selected-target fidelity | 25/106, `0.2358` | 23/106, `0.2170` | `+0.0189` |
| Group selected-target rate | `0.2967` | `0.2707` | `+0.0260` |
| Switching groups | `4/32` | not gating | required `24/32` |
| K=1 validity | `0.7925` | `0.7736` | `+0.0189` |
| Paired target result | helped 2 | hurt 0 | tied 104, `p=0.25` |

Both arms missed every target-control threshold. The confirmation's favorable
point estimates were supported by only two discordant rows and remained close
to the target-blind null.

### Mechanistic diagnostics

The negative did not look like a dead training graph:

- posterior and prior head statistics separated;
- gradient norms were nonzero;
- posterior-collapse fraction was zero;
- injection residual RMS grew roughly fiftyfold;
- the learned scalar moved only slightly, from approximately `0.001000` to
  `0.00105`;
- frozen keeper lineage remained unchanged.

This localized the uncertainty to the interface between learned latent signal
and output control: either the residual was too small, or the additive route was
not a useful target-control channel.

## 6. Forced-Injection Causal Probe

### Question

If the already-learned posterior residual is multiplied at inference, does
terminal selection become target-responsive before model validity collapses?

### Frozen design

- No training, optimizer, backward pass, or checkpoint mutation.
- Both preserved A0 EMA checkpoints.
- Exact same 106 variants and 32 groups.
- Exact published per-row trajectory seeds.
- Residual multipliers: `1, 3, 10, 30, 100`.
- Metrics: switching groups, selected-target fidelity, K=1 validity.
- Factor `1` had to reproduce every published A0 posterior prediction.
- Deterministic-lineage hash had to match before and after.

### Locked reading

`CHANNEL-EXISTS` required:

- switching `>=16/32` at any factor; and
- K=1 validity strictly `>0.50` at that factor.

`NO-CHANNEL` applied if:

- switching stayed `<8/32` at every factor; or
- switching reached `>=16/32` only after validity fell below `0.50`.

All intermediate outcomes were `AMBIGUOUS` and closed by default.

## 7. Forced-Injection Results

### Complete table

| KL arm | Multiplier | Injection RMS ratio | Changed vs 1x | Switching | Target fidelity | K=1 validity |
|---|---:|---:|---:|---:|---:|---:|
| `0.001` | 1 | `0.00058` | 0/106 | 4/32 | `0.2264` | `0.7453` |
| `0.001` | 3 | `0.00173` | 7/106 | 3/32 | `0.2264` | `0.7547` |
| `0.001` | 10 | `0.00577` | 12/106 | 5/32 | `0.2075` | `0.7075` |
| `0.001` | 30 | `0.01726` | 45/106 | 15/32 | `0.1792` | `0.6321` |
| `0.001` | 100 | `0.05650` | 83/106 | 22/32 | `0.1321` | `0.4340` |
| `0.0001` | 1 | `0.00056` | 0/106 | 4/32 | `0.2358` | `0.7925` |
| `0.0001` | 3 | `0.00167` | 14/106 | 4/32 | `0.2358` | `0.7358` |
| `0.0001` | 10 | `0.00555` | 18/106 | 4/32 | `0.2170` | `0.7170` |
| `0.0001` | 30 | `0.01659` | 45/106 | 9/32 | `0.1604` | `0.6321` |
| `0.0001` | 100 | `0.05442` | 90/106 | 13/32 | `0.1038` | `0.4623` |

### Integrity

- Factor-1 predictions exactly matched the published A0 receipts for all 212
  arm-row evaluations.
- Guidance checkpoint hashes:
  - KL `0.001`:
    `186fd2bb2efcd17574e261cf68d5e9cf9b5afd5b001910d5787dc12422974d50`
  - KL `0.0001`:
    `405cc76f972522d091d10edb77b10d278c12b63b20f6ee05fa8fd29fd6f33b1e`
- Frozen deterministic-lineage hash matched before and after both arms.
- No training or coverage evaluation occurred.

### Final scorer repair

All GPU evaluation completed and both arm receipts landed before a final CPU
scoring error. The scorer incorrectly required JSON object keys to preserve the
numeric factor insertion order. The receipt writer used `sort_keys=True`, which
serialized keys lexicographically as `1,10,100,3,30`.

The repair:

- validates the exact factor set independently of key order;
- iterates factors in the locked numeric order;
- adds a regression test that round-trips summaries through sorted JSON;
- changes no model output, metric, threshold, or interpretation.

The corrected full test suite passed in GitHub Actions.

## 8. Interpretation

### 8.1 The additive route has raw causal influence

It would be incorrect to read `NO-CHANNEL` as "the injected residual never
affects the model." At factor 100:

- the primary arm changes 83/106 predictions;
- the confirmation arm changes 90/106 predictions.

The route can therefore perturb computation and terminal output.

### 8.2 The influence is not target-controllable

The central result is the direction of change:

- switching rises with dose;
- target fidelity falls with dose;
- validity falls with dose;
- neither arm shows a fidelity improvement at any multiplier.

The primary arm's factor-30 point is the closest apparent near miss:
switching reaches 15/32 while validity remains `0.632`. But target fidelity has
already fallen from `0.226` to `0.179`. This is not almost-successful control;
it is increasingly disruptive variation that still does not follow the selected
chain.

At factor 100 the primary passes the switching count but fails the validity
floor, exactly matching the preregistered destructive-intervention reading.
The confirmation arm never reaches the switching threshold and also collapses
below the validity floor.

### 8.3 Magnitude was not the missing ingredient

The probe resolves the A0 ambiguity. The small trained scale was not hiding a
useful target-aligned channel that simply needed amplification. Multiplying the
learned residual by up to 100 produces more changes, but those changes are less
valid and less faithful to the posterior target.

The current design's failure is therefore better localized as a conditioning
interface/objective problem:

- the posterior representation contains statistical target-dependent signal;
- the additive projected delta can perturb the state;
- the downstream recurrent computation does not interpret that perturbation as
  a branch-selection instruction.

## 9. What Is Established

1. The original single-target guided-width curriculum was non-identifying.
2. Corrected repeated-prompt multi-target A0 directly measured posterior target
   control.
3. Both KL arms failed that control gate near the target-blind null.
4. The failure was not explained by frozen-graph breakage, posterior collapse,
   checkpoint mutation, or a dead residual.
5. The additive residual has raw causal influence at high dose.
6. Amplification does not produce useful target control; it produces
   target-misaligned and eventually invalid output variation.
7. The current additive high-level re-entry route is closed under its locked
   gate.

## 10. What Is Not Established

- GRAM is false.
- Stochastic recurrent width is false.
- A pretrained recurrent transformer cannot support guided width.
- A multiplicative, gated, attention-conditioned, or otherwise redesigned
  conditioning path would fail.
- Width loses to depth at iso-compute; coverage and iso-compute comparisons
  were correctly never opened.
- The early SVGD results evaluate GRAM-style guidance.
- The posterior contains no target information. The result is about usable
  causal control, not representational mutual information.

## 11. Program Decisions

### Closed

- Larger trained scale on the same additive route.
- Further KL, seed, optimizer, or duration sweeps from A0.
- Coverage-at-K from these checkpoints.
- LPRM/selector, learned per-trajectory halting, particles, or SVGD on this
  line.

### Banked

The result is a useful Paper Two boundary:

> A target-conditioned latent head can learn separated statistics without
> acquiring terminal control. Amplifying its additive re-entry residual causes
> output variation but not target-aligned branching.

The gate order prevented a misleading coverage result from being interpreted as
guided stochastic reasoning.

### Priority

Paper One closure remains the primary program priority. Phase G should not
consume another training slot unless strategy explicitly authorizes a new
architecture and preregistration after weighing it against Paper Two assembly
and the think-token program.

## 12. Future Design Ideas, If Reopened

These are hypotheses for a new program, not authorized continuations.

### A. FiLM-style conditioned re-entry

Use the posterior/prior latent to modulate the re-entry state
multiplicatively:

```text
h' = gamma(z, h) * h + beta(z, h)
```

This gives the conditioning signal access to feature selection and suppression,
not only a small direction added to a high-confidence state.

### B. Per-loop branch-choice supervision

Add an auxiliary loss at every transition:

```text
Does the latent-conditioned state choose the selected chain's next branch?
```

This supplies credit exactly at divergence points rather than asking final CE
and KL to discover branch semantics indirectly.

### C. Per-loop conditioned gates

Allow the latent to control a small number of loop-specific scales or gates.
This tests whether a shared additive residual fails because the meaning of a
branch instruction changes across recurrent transitions.

### D. Rank-limited conditioned state transform

If FiLM fails, use a low-rank conditioned operator on the re-entry state. Keep
the recurrent block frozen and bound trainable capacity so any control gain
remains attributable to the interface.

### E. Causal probe before retraining

For any successor, test whether an oracle branch label injected through the new
route can causally alter the next-step branch while preserving validity before
training a variational prior/posterior. This separates route capacity from
amortized inference difficulty.

## 13. Questions for Strategy and Research

1. Is the `NO-CHANNEL` boundary sufficiently complete to close Phase G for
   Paper Two, or is one oracle-conditioned FiLM micro-probe worth its
   information value?
2. Should a future successor first prove next-transition control with explicit
   branch labels before reintroducing Gaussian sampling and KL?
3. Is a fixed orthonormal latent projection the wrong interface for a pretrained
   residual stream whose useful control directions may be highly anisotropic?
4. Would multiplicative modulation remain a clean transplant, or would it
   become enough new architecture that the scientific question should be
   reframed?
5. Should posterior conditioning include the entire selected next-state
   relation, not only its frozen token embedding?
6. Does the high-confidence keeper create an output basin that requires
   conditioning before or inside the recurrent block rather than at re-entry?
7. Could target control be measured at the next-loop hidden state or branch
   logits before terminal decoding, revealing where alignment disappears?
8. If target control passes in a redesigned route, should the existing gate
   order remain unchanged? Current evidence strongly supports retaining it:
   posterior control, preservation, coverage, then selection.
9. Is the strongest publication contribution now the methodological one:
   multimodal curricula must be identifying, and latent diversity must clear a
   causal control gate before coverage is interpretable?
10. Given opportunity cost, should engineering effort move now to Paper One and
    Paper Two assembly rather than another width intervention?

## 14. Recommended Strategy Reading

The current recommendation is:

1. Ratify `NO-CHANNEL` as the final result for the additive A0 transplant.
2. Bank the complete negative and its causal localization for Paper Two.
3. Do not authorize same-route training.
4. Finish Paper One closure and evidence compilation.
5. Decide separately whether Paper Two needs:
   - no further GPU work; or
   - one small oracle-conditioned interface probe before manuscript lock.

If one future probe is approved, the highest-EVPI version is not another
variational training run. It is a bounded oracle branch-control test comparing
the existing additive route with one FiLM route on the same frozen keeper. The
test should ask whether a supplied branch instruction can control the next
transition without destroying validity. Only a positive result would justify a
new guided-width training specification.

## 15. Receipt Map

| Artifact | Purpose |
|---|---|
| `docs/PHASE_G_A0_POSTERIOR_CONTROL_HANDOFF_20260718.md` | corrected A0 design and first verdict |
| `docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json` | locked A0 posterior-control thresholds |
| `outputs/stage5/stage5_phase_g_multitarget_control_20260718/summary.json` | final A0 status and arm lineage |
| `docs/STAGE5_PHASE_G_FORCED_INJECTION_PROBE_SPEC_20260718.md` | preregistered causal probe |
| `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/arms/kl_0p001/summary.json` | primary multiplier arm |
| `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/arms/kl_0p0001_confirmation/summary.json` | confirmation multiplier arm |
| `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/gate.json` | machine-readable final gate |
| `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/gate.md` | compact final table |
| `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/summary.json` | final run-level receipt |

## Bottom Line

The current Phase G design did not fail because the stochastic residual was
merely too quiet. When amplified, it became behaviorally loud but not
semantically aligned. The additive high-level re-entry path can perturb the
model, but it does not function as a selected-trajectory control channel.

That result closes the present transplant cleanly. It preserves the broader
research question while identifying what any successor must change: not just
the dose, but the causal interface and transition-level supervision.
