# Phase G-A0: Guided Stochastic Width Handoff

**Date:** 2026-07-18  
**Status:** completed and preregistered blocked  
**Audience:** strategy and deep-research review  
**Primary receipt:** \`outputs/stage5/stage5_phase_g_multitarget_control_20260718/summary.json\`

## Executive Decision

Phase G-A0 tested the essential element missing from the early GRAM-inspired work:
whether a target-conditioned posterior can select different valid latent paths for
the same prompt and teach a prior to do so. It did not pass. The primary
KL coefficient (\`1e-3\`) failed the locked posterior-control gate, and the
single pre-authorized confirmation (\`1e-4\`) also failed. The runner therefore
stopped before coverage-at-K, a selector, per-trajectory halting, or SVGD.

This is a meaningful negative for one bounded transplant:

> With a frozen repaired recurrent Qwen keeper, trainable prior/posterior heads,
> and small high-level bridge injection, multi-target transition supervision did
> not make the posterior reliably choose the selected valid terminal among
> alternatives for an otherwise identical prompt.

It is not a negative result about GRAM generally, stochastic recurrent reasoning
generally, or the earlier SVGD line. The early work lacked a target-conditioned
posterior. This experiment had one, but it deliberately forbids post-hoc sweeps.

The immediate decision is to close corrected G-alpha without a coverage rerun and
return GPU priority to Paper One, especially Arm E. Any future stochastic work
needs a new design and pre-registration rather than a continuation of this run.

## 1. Scientific Lineage

The original recurrent Qwen proposal was:

    input -> recurrent latent loop -> learned halting -> one trajectory -> answer

The GRAM-inspired extension proposed:

    input -> K sampled latent trajectories -> K candidate answers
          -> confidence / vote / verifier / selector

The intended principle was that recursive reasoning could be a distribution over
latent trajectories, rather than repeated refinement of one deterministic hidden
state.

The early code did add Gaussian latent perturbation, K-trajectory evaluation, and
later SVGD repulsion. It did not implement GRAM's learned conditional prior taught
by a target-conditioned posterior. It used a state-conditioned Gaussian
regularized toward a standard normal and attempted to induce diversity
geometrically. Those early experiments therefore show only that naive noise and
particle repulsion did not reliably create additional correct candidates on a weak,
initially miswired recurrent substrate.

The deterministic program subsequently repaired loop closure and prelude
re-injection, audited bridge gradients, and established a trainable
intermediate-state recurrence. That was the necessary substrate repair. The GRAM
divergence audit also established that GRAM's own "stochasticity only" ablation
fails on its multimodal task, while guided stochasticity succeeds. The missing
posterior/prior mechanism was thus the correct next test.

## 2. Why the First Guided-Width Run Was Not Admissible

The July 17 guided-width curriculum had one target chain per base problem. Its
posterior could not be identified as target-selective because a prompt was never
paired with a different valid selected target. Its post-run autopsy recorded:

| Training fact | Result |
|---|---:|
| Problem groups | 2,048 |
| Repeated-prompt groups | 0 |
| Groups with multiple selected targets | 0 |
| Teacher-minus-prior target-in-K at K=1 | -0.98 pp |

That run could not answer the posterior-control question. It was not accepted as
a clean negative and was replaced by the multi-target A0 experiment.

## 3. Corrected A0 Hypothesis and Gate Order

For an identical branching-relation prompt with several valid exact-depth
terminals, changing only the selected valid terminal and sampled valid chain
should change the posterior teacher's latent trajectory and first prediction. The
learned prior should approximate that behavior at inference.

The gate order was locked before training:

    posterior exact-target control
      -> deterministic K=1 preservation
      -> prior coverage vs entropy-matched answer sampling
      -> prior coverage vs iso-compute depth
      -> only then: selector, per-trajectory halting, or SVGD

A0 is therefore not a coverage experiment. It tests whether the posterior has
target-selective behavior worth distilling into a stochastic prior.

## 4. Task and Data

The task is a verbal branching relation over 20 named symbols. At each step the
current symbol may transition to either of two successors. A start symbol and
exact depth can have several valid terminals.

For each base problem, the generator enumerated every distinct exact-depth
reachable terminal and emitted one row for each selected terminal. Variants share
table, start, depth, prompt, and reachable set; only the selected terminal and
its valid chain differ. Each row contains \`base_problem_id\`, target-variant
metadata, an exact \`sampled_chain\`, and per-loop completions.

| Surface | Base prompts | Target variants | Role |
|---|---:|---:|---|
| Training | 512 | 1,899 | prior/posterior learning |
| Held-out posterior control | 32 | 106 | primary A0 decision |

All training and held-out groups were multi-target. Validation checked exact
chains, all reachable terminals, shared prompt/table/start/depth within each
group, and disjoint train/control base-problem IDs. Training sampled uniformly by
base prompt and then by target variant, so larger reachable sets did not receive
more updates.

## 5. Model and Training Boundary

The keeper was the repaired natural-surface recurrent Qwen checkpoint:

    SHA-256: 0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f

The recurrent block, prelude, and coda remained frozen. Only these parameters were
trainable:

- \`phase_g_prior_head.*\`
- \`phase_g_posterior_head.*\`
- \`phase_g_injection_scale\`

The prior observes recurrent state. The posterior additionally observes the
selected valid target/chain. Its sampled high-level residual enters at the
re-entry bridge while the deterministic update remains intact.

Training used K=4 trajectories, exact per-loop supervision, balanced
posterior-to-prior KL, EMA decay 0.999, bfloat16, base-problem-uniform sampling,
learning rate \`1e-4\`, and injection-scale initialization \`0.001\`. Both arms
ran 1,000 steps with the same seed and all settings other than KL coefficient.

At every step, the trainer asserted frozen gradients. It also checked the frozen
active-lineage hash before and after each arm.

## 6. Locked A0 Decision Rule

The held-out K=1 posterior-control surface had 32 prompt groups and 106 selected
target variants. The pre-registered requirements were:

| Criterion | Required value |
|---|---:|
| Repeated multi-target prompt groups | >= 32 |
| Teacher selected-target fidelity | >= 0.60 |
| Teacher-minus-prior selected-target lift | >= +0.15 |
| One-sided paired sign test | p <= 0.05 |
| Teacher switching groups | >= 24 of 32 |

The primary arm used KL \`1e-3\`. Only if it blocked, exactly one confirmation at
KL \`1e-4\` could run. No other coefficient, seed, optimizer, or duration sweep
was authorized. If both arms blocked, the experiment ended without a coverage
rerun.

Validity was report-only: it could identify an obvious regression but could not
pass posterior control by itself.

## 7. Execution Integrity

Two launch failures were repaired without changing the model or retraining:

1. The runner initially read the frozen-gradient receipt from the wrong JSON
   nesting level. The trainer had correctly stored it under
   \`summary.config.frozen_gradient_assertions\`.
2. The audit initially treated cached \`valid_samples\` as a list, although the
   cache stores an aggregate integer. The audit now calculates first-sample
   validity directly from the prediction and exact reachable set.

Both arms completed after these fixes. Receipts show:

- 1,000 frozen-gradient assertions in each arm;
- frozen active-lineage hash unchanged in each arm;
- raw and EMA checkpoints, RNG manifests, traces, row caches, audit, and gate
  files saved;
- held-out deterministic control screen: \`81/106 = 76.4%\`, above its locked
  pooled and depth-specific floors.

This rules out a base-weight mutation, dead frozen-gradient check, or failed
deterministic substrate screen as the explanation for the result.

## 8. Results

### Primary: KL = 0.001

| Measure | Posterior teacher | Prior | Difference/test |
|---|---:|---:|---:|
| Selected-target fidelity, 106 variants | 22.64% | 23.58% | -0.94 pp |
| Paired target result | helped 1 | hurt 2 | one-sided p=0.875 |
| Group selected-target rate | 28.11% | 29.41% | -1.30 pp |
| Teacher switching groups | 4 / 32 | required 24 / 32 | failed |
| K=1 valid answer rate | 74.53% | 78.30% | -3.77 pp |

The primary missed every posterior-control criterion and triggered the one
authorized confirmation.

### Confirmation: KL = 0.0001

| Measure | Posterior teacher | Prior | Difference/test |
|---|---:|---:|---:|
| Selected-target fidelity, 106 variants | 23.58% | 21.70% | +1.89 pp |
| Paired target result | helped 2 | hurt 0 | one-sided p=0.25 |
| Group selected-target rate | 29.67% | 27.07% | +2.60 pp |
| Teacher switching groups | 4 / 32 | required 24 / 32 | failed |
| K=1 valid answer rate | 79.25% | 77.36% | +1.89 pp |

The confirmation had a favorable but tiny point estimate. It remains far below
the 60% absolute-fidelity and 15-point lift requirements, with only two non-tied
comparisons and a non-significant paired test.

## 9. Diagnostics and Interpretation

This does not resemble a dead-gradient or simple posterior-collapse failure:

- training gradient norms were nonzero;
- \`phase_g_posterior_collapse_fraction\` was recorded as zero;
- posterior and prior head statistics separated during training;
- injection scale moved only from roughly 0.001000 to 0.001049.

The residual's RMS ratio increased from roughly \`8e-5\` initially to
\`0.0041-0.0043\` by step 1,000. The heads moved, but the
target-conditioned signal did not become a reliably controllable output-level
trajectory difference. This is diagnostic evidence, not proof that the small
injection was the cause.

Supported conclusions:

1. The initial single-target curriculum defect was real and is fixed.
2. The corrected A0 measurement directly tested target-specific posterior
   behavior rather than generic validity or diversity.
3. On this frozen-keeper, small-residual transplant, posterior conditioning did
   not control selected terminal choice enough to teach a useful prior.
4. It was correct not to run coverage-at-K: a coverage change without target
   control would not support a guided-width claim.

Not supported:

- GRAM as a general method is falsified.
- Stochastic recurrent reasoning or width is false.
- Width loses to depth at iso-compute; that test was intentionally not run.
- Posterior collapse caused the failure.
- The pretrained substrate can never support guided width.

## 10. Decisions and Next Steps

The following are locked by the A0 preregistration:

- Final A0 status: \`blocked_posterior_control_after_confirmation\`.
- Do not run A1 coverage, entropy-matched temperature comparison, iso-compute
  width/depth, selector, learned per-trajectory halting, or SVGD from these
  checkpoints.
- Do not run another KL, seed, optimizer, or duration sweep.
- Preserve and report the negative transparently.
- Return the primary GPU lane to Paper One Arm E, which is independent and
  closes the matched rank-16 adapter-budget comparison.

Questions for a *new* future Phase G specification:

1. Is a prior/posterior residual plus a roughly 0.4%-RMS bridge perturbation too
   weak to encode selected branch identity? What minimal trainable injection
   route would preserve attribution?
2. Is terminal/chain conditioning sufficiently local and causally connected to
   decode, or should posterior supervision target each transition more directly?
3. Is verbal branching relation the right first multimodal task, or should a
   future test use N-Queens, graph coloring, or exact non-injective abduction?
4. Does a successor need a pre-registered larger intervention or more training
   dose, with preservation guardrails, rather than treating this blocked run as
   a tuning baseline?
5. Should a successor retain target switching as a prerequisite, then evaluate
   exact oracle coverage, entropy-matched answer sampling, iso-compute depth,
   and sequential latency only after it passes?

## 11. Receipt Map

| Artifact | Purpose |
|---|---|
| \`docs/gram_divergence_audit_20260711.md\` | original GRAM divergence and do-not-claim record |
| \`docs/STAGE5_PHASE_G_MULTITARGET_CORRECTION_SPEC_20260718.md\` | corrected multi-target design |
| \`docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.md\` | locked thresholds and contingency |
| \`outputs/stage5/stage5_phase_g_multitarget_control_20260718/data/summary.json\` | manifests and data validation |
| \`.../deterministic/posterior_control/summary.json\` | keeper validity screen |
| \`.../train/kl_0p001/summary.json\` | primary training integrity |
| \`.../posterior_control/kl_0p001/{audit,gate}.json\` | primary result |
| \`.../train/kl_0p0001_confirmation/summary.json\` | confirmation integrity |
| \`.../posterior_control/kl_0p0001_confirmation/{audit,gate}.json\` | final verdict |
| \`.../summary.json\` | run-level final status |

## Bottom Line

The project now has a clean answer to a question the early stochastic work could
not ask: can a target-conditioned posterior impose a different valid latent
solution on the repaired recurrent Qwen keeper? Under the frozen,
small-residual A0 transplant, the answer is no. The negative is bounded,
well-instrumented, and more informative than an unrestricted coverage sweep
would have been. It closes the present guided-width correction without
conflating it with the deterministic recurrence contribution or GRAM's
task-specific demonstrations.

