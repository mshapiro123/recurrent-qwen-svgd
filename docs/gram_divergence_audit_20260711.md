# GRAM Divergence Audit

> **Queue amendment, July 12, 2026:** The return to guided stochastic width is no longer a separate future track. The abductive-injective deterministic gate is its substrate prerequisite, followed immediately by Phase G-alpha. See [PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md](PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md).

## From stochastic recurrent Qwen to a validated deterministic recurrent substrate

**Date:** July 11, 2026  
**Project:** `mshapiro123/recurrent-qwen-svgd`  
**Purpose:** Catalogue how the project diverged from its original GRAM-inspired plan, why each divergence occurred, what the experiments do and do not establish, and what would be required to return to the original probabilistic thesis.

**Audit amendment:** The strategy review subsequently verified the audit directly against arXiv:2605.19376 and supplied ledger amendments A39 and F6. This revision incorporates the paper's mechanism ablation, the amended Phase G gate, and the locked G-alpha comparators.

---

## Executive conclusion

The project is no longer running the model that the original research question described.

The original thesis was:

```text
input
  -> recurrent reasoning state
  -> stochastic latent transition at each loop
  -> K independently sampled trajectories
  -> possibly different halting depths
  -> K candidate answers
  -> selector or verifier
  -> final answer
```

The model currently being trained is closer to:

```text
input
  -> frozen Qwen prelude
  -> recurrent block with corrected input re-injection
  -> fixed, supervised loop count
  -> deterministic intermediate-state targets
  -> one trajectory
  -> one answer
```

That is a major divergence, but not an arbitrary one. Early stochastic experiments were performed on a weak and, for part of the program, structurally miswired recurrent substrate. They also implemented only a loose VAE-like approximation to GRAM rather than GRAM's target-conditioned variational trajectory objective. The project therefore paused stochastic width and repaired the deterministic machine first.

The correct scientific interpretation is:

> We have not falsified GRAM or stochastic recurrent reasoning. We obtained mostly negative results for naive Gaussian latent injection and SVGD-style particle repulsion on an incompletely trained, initially miswired recurrent Qwen substrate. We then established that the repaired deterministic substrate can learn and preserve genuine multi-step latent state transitions. The original stochastic-width thesis remains untested in its principled form.

The current program has produced a more credible launch point for the original idea, but a return to it requires an explicit new phase. It should implement a learned conditional prior, a target-conditioned posterior, transition-level KL training, independent trajectory sampling, and a selector trained on trajectory correctness. SVGD should be withheld until that cleaner baseline is measured.

---

## 1. The four systems that must not be conflated

### 1.1 The original deterministic proposal

The starting design was a pretrained Qwen converted into a recurrent-depth model:

```text
input -> recurrent loop -> learned halting depth -> one latent trajectory -> one answer
```

Its purpose was adaptive computation in latent space. The model would reuse a block, refine a persistent hidden state, and learn how many recurrent passes a problem required.

### 1.2 The original GRAM-inspired extension

The GRAM-inspired proposal added stochastic width:

1. Sample a latent perturbation inside the recurrent transition so identical inputs can follow different internal paths.
2. Run `K=2` or `K=4` trajectories in parallel instead of only increasing depth.
3. Prevent collapse into identical trajectories.
4. Decode multiple candidate solutions and select by confidence, consistency, or a verifier.

The intended claim was not that this exactly reproduced GRAM. It was a practical hybrid:

- Qwen as a pretrained language backbone.
- A PonderNet-style probabilistic halting process.
- A shared recurrent Qwen block connected by a bridge.
- LoRA and small auxiliary modules for economical adaptation.
- A lightweight variational latent head.
- Multi-trajectory sampling.

The phrase "GRAM-inspired" meant borrowing the central idea that reasoning should be represented as a distribution over recursive latent trajectories.

### 1.3 What GRAM actually implements

The GRAM paper defines a stochastic recursive transition, not merely noisy decoding. At each step it forms a deterministic proposal and then samples a learned residual:

```text
u_t = deterministic_transition(z_{t-1}, x)
epsilon_t ~ Normal(mu_theta(u_t), sigma_theta(u_t)^2 I)
z_t = u_t + epsilon_t
```

Its main instantiation uses hierarchical high- and low-level states, although the paper's abstract framework also permits flatter looped or universal-transformer forms. More importantly, training uses two trajectory distributions:

- A learned conditional prior `p_theta(trajectory | x)` used for generation.
- A target-conditioned posterior `q_phi(trajectory | x, y)` that guides training toward solution-bearing stochastic paths.

The objective is an ELBO-style surrogate with predictive/deep-supervision terms and transition-level posterior-to-prior KL terms. Width is obtained by independent samples from the learned prior. GRAM also evaluates majority voting and a latent process reward model that predicts final trajectory correctness. Its main adaptive-computation mechanism is Q-learning-style ACT rather than PonderNet's geometric halting distribution.

These details matter because the paper explicitly reports that naive random initialization or stochastic decoding does not reproduce the gains. Its evidence attributes the useful stochasticity to variational guidance, not noise by itself. See the [GRAM paper](https://arxiv.org/pdf/2605.19376) and [project page](https://ahn-ml.github.io/gram-website/).

The mechanism ablation makes the distinction unusually concrete. On 8x8 N-Queens with five samples, full GRAM scores `99.69`, while its "stochasticity only" Gaussian variant without learned guidance scores `50.27`. Guidance without stochasticity scores `0.00`; stochastic decoding and random initialization added to TRM also fail to match GRAM. On Sudoku, stochasticity alone remains competitive, so the failure is not simply that noise always destroys prediction. It appears specifically where structured navigation of a multi-solution space matters. This is strong external corroboration for our do-not-claim rule: our early Gaussian injection was closest to an ablation that GRAM itself shows is insufficient, not to the full method.

### 1.4 What the project currently runs

The current natural-surface training path uses:

- `max_loops=8`.
- `loop_loss_mode="per_loop_labels"`.
- One trajectory.
- No latent sampling.
- No SVGD particle update.
- No learned halting loss.
- No learned loop-control loss.
- A trainable bridge.
- A fully unfrozen recurrent block.
- Frozen prelude and coda.

The trainable parameter summary is approximately:

| Component | Trainable parameters |
|---|---:|
| Recurrent block | 178,948,608 |
| Bridge | 3,214,849 |
| Halting | 0 |
| Re-entry adapter | 0 |
| Latent trajectory module | 0 |
| **Total** | **182,163,457** |

This is no longer a LoRA-scale surgical conversion. It updates roughly one-third of a nominal 0.5B model, concentrated in the reused recurrent block.

---

## 2. Divergence catalogue

| Axis | Original GRAM-inspired plan | GRAM reference method | What we built or now run | Divergence type | Scientific consequence |
|---|---|---|---|---|---|
| Backbone | Pretrained Qwen | Task-oriented recurrent reasoning architectures | Qwen2.5-0.5B split into prelude, recurrent block, and coda | Intentional hybrid | Improves language reuse but introduces distribution-matching and re-entry problems absent from a natively recurrent model. |
| Recurrent state | One persistent Qwen hidden state | Main model uses hierarchical high/low latent state | Flat token-sequence hidden state | Intentional simplification | Compatible with GRAM's broad framework, but not its principal hierarchical implementation. |
| Stochastic transition | Sample latent at each loop | Learned conditional Gaussian residual at each recursion | One sequence-pooled Gaussian head plus broadcast hidden delta | Material approximation | The perturbation is global across token positions and is less expressive than a structured state transition. |
| Prior | Lightweight variational head | Learned `p_theta(epsilon_t | state, x)` | State-conditioned `mu/logvar`, regularized to standard normal | Material divergence | There is no learned reference trajectory distribution matching the inference process. |
| Posterior | Not fully specified initially | Separate target-conditioned `q_phi(epsilon_t | state, x, y)` | None | Missing core mechanism | Training cannot use the answer to discover solution-bearing stochastic paths. |
| Variational objective | KL and diversity regularization | Posterior/prior transition KL plus predictive supervision | KL to `N(0,I)` plus CE and pairwise diversity reward | Material divergence | This is VAE-like regularization, not GRAM's guided stochastic trajectory learning. |
| Width | `K=2` or `K=4` sampled paths | Independent samples from learned conditional prior | Repeated batch trajectories with sampled latent injection and optional shared SVGD interaction | Partial implementation | Independent sampling exists, but SVGD changes the sampling law by coupling trajectories. |
| Anti-collapse | Encourage meaningful differences | Variational posterior/prior training; candidate coverage; reward model | Pairwise cosine diversity, random noise schedules, SVGD kernels, projection geometry | Research extension | Much effort optimized geometric separation before establishing that separation carried correctness. |
| Particle interaction | Optional future idea | Not part of GRAM | SVGD attraction and repulsion after the recurrent block | New mechanism | This became a major side program and can no longer be interpreted as a direct GRAM test. |
| Halting | PonderNet-style probabilistic depth | Q-learning-style ACT per trajectory | PonderNet-style halt head initially; now disabled or forced depth | Initial hybrid, later pivot | Current results do not test independently sampled trajectory depths. |
| Decode aggregation | Multiple candidates plus selector | Majority vote or latent process reward model | Standard forward averages trajectory logits; separate best-of-K and vote evaluators added later | Incomplete alignment | Averaging can erase multimodality; it is not equivalent to selecting a coherent trajectory. |
| Selector | Confidence, self-consistency, verifier | Majority vote and learned latent correctness model | Exact-match/oracle best-of-K, ARC symbolic selectors, reliability voting, later depth selectors | Partial and task-specific | No general latent process reward model was trained for stochastic trajectories. |
| Training scope | LoRA plus small bridge/latent modules | Method trained as recurrent stochastic architecture | Full recurrent block later unfrozen | Major scope expansion | Stronger substrate, but the "small percentage of weights" claim needs careful qualification. |
| Data | Opus/Fable-like reasoning traces adapted to recurrence | Structured tasks with supervised outcomes and recursive dynamics | Early Opus JSONL, then ARC/MCQ diagnostics, then synthetic exact state chains and natural-surface relay tasks | Evidence-driven pivot | Current positives prove trainable recurrence more directly, but are farther from broad reasoning and from stochastic multi-solution coverage. |
| Primary benchmark | Hard reasoning and candidate diversity | ARC-AGI-like tasks plus N-Queens, graph coloring, width scaling, unconditional generation | Early 14-task exact suite and ARC MCQ; later deterministic synthetic and natural-surface transfer | Major evaluation pivot | The present evidence is strongest for deterministic state-transition learning, not stochastic reasoning breadth. |
| Multi-solution coverage | Central reason for width | Explicitly evaluated | Only small early prompts and candidate-conversion gates | Deferred | The most direct test of the original thesis remains largely undone. |
| Unconditional generation | Not central but compatible | Evaluated by GRAM | Not pursued | Omitted | No conclusion about generative latent recursion. |

---

## 3. How the divergence happened

### 3.1 June 15-18: a pragmatic stochastic hybrid was built

The first implementation correctly introduced the surface features of the proposal:

- A Qwen layer split.
- Repeated recurrent computation.
- A probabilistic halting head.
- A Gaussian latent policy and adapter.
- `K` trajectory replication.
- Per-trajectory logits and diversity diagnostics.
- Optional SVGD updates.

However, the latent module had one learned state-conditioned Gaussian and a KL to a standard normal. It did not have GRAM's target-conditioned posterior, learned conditional prior pair, or posterior/prior transition KL. This was the first and most important divergence from the published mechanism.

The standard model output also averaged logits across trajectories. That preserved compatibility with a causal-LM interface, but it did not preserve distinct candidate identities. Separate best-of-K tooling was needed to inspect whether any trajectory contained a correct answer.

### 3.2 June 18-20: anti-collapse became SVGD geometry research

Early trajectories were nearly identical:

- First-token trajectory diversity was commonly around `0.0007` to `0.002`.
- A phase-two validation reported trajectory diversity around `0.008`.
- On the first five exact tasks, deterministic phase one and phase-two best-of-four both achieved only `2/5`.

The response was to add:

- Noise schedules and per-step noise.
- Multi-seed evaluation.
- SVGD repulsion sweeps.
- Random and calibrated projections.
- Euclidean and spherical kernels.
- Within-group PCA calibration.
- Drift, repulsion, bandwidth, clipping, and pairwise-distance diagnostics.

This work was technically informative, but it changed the research question. GRAM asks whether a learned stochastic trajectory distribution can cover multiple solution modes. The SVGD program asked whether explicit particle interaction in selected hidden-space geometries could force trajectories apart.

The experiments repeatedly showed why those are not the same question:

- Repulsion was often only about `0.5%` to `2%` of recurrent drift.
- Many paired repulsion-on/off comparisons were ties.
- Stronger noise or longer schedules often produced four unique outputs while reducing correct candidate count.
- Spherical kernels could create large separation and severe accuracy collapse.
- Isolated projected settings looked promising, but gains were not stable enough to establish candidate conversion.

The most defensible conclusion was that geometric diversity alone was not useful diversity.

### 3.3 June 20-23: candidate selection and capability recovery displaced the latent objective

The project then built selectors, ARC candidate gates, reliability voting, and trajectory distillation. This remained aligned with the original need to convert candidate coverage into accuracy.

But the particle gates exposed a more basic problem: the recurrent model itself was weak, and K trajectories did not reliably add correct-bearing alternatives. Recovered phase-one and phase-two checkpoints were evaluated on larger ARC slices. Particles and SVGD frequently underperformed the deterministic recurrent baseline.

At this point, the working question changed from:

> Does stochastic width improve a competent recurrent reasoner?

to:

> Is the recurrent reasoner itself competent and structurally sound?

That was a necessary change, not mission drift for its own sake.

### 3.4 June 24-28: architecture repair revealed that earlier stochastic tests were confounded

The re-entry diagnostic program found multiple problems in the recurrent substrate:

- Entry/exit distribution mismatch across the reused block.
- Norm and tail drift.
- Bridge liveness and initialization problems.
- Re-entry stabilization requirements.
- Most importantly, missing prelude/input re-injection on recurrent passes.

The missing re-injection was fixed on June 28 in commit `96efe39` (`Fix recurrent re-entry prelude injection`). Before that correction, the loop was not implementing the intended recurrence over a state continually grounded in the input context.

This materially weakens any attempt to interpret early stochastic failures as evidence against the original architecture. Stochastic width was being tested on a recurrent system whose loop closure was not yet correct.

### 3.5 June 28-July 8: the project became a deterministic recurrence mechanism program

The next program deliberately turned off stochastic components and asked whether the repaired loop could learn an actual iterative state update.

It introduced:

- Forced loop counts.
- Exact per-loop labels.
- Synthetic function-iteration tasks.
- Staged depth curricula.
- Gradient-path audits and finite-difference checks.
- Chain supervision followed by outcome-only annealing.
- Depth extrapolation and hidden-state probes.
- Support ladders and seed replications.

This work established a positive result that the early stochastic phase never had: the recurrent block and bridge could learn a persistent, multi-step latent chain, and that chain could survive removal of intermediate supervision. It also established limits: behavior was often tied to trained support and required careful curriculum design.

This is scientifically important because it validates the substrate needed for a future stochastic-width test. It does not itself test stochastic reasoning.

### 3.6 July 8 onward: natural-surface transfer further consolidated the deterministic path

The present natural-surface experiments train deterministic recurrent state updates on verbal relay/pointer-style tasks mixed with synthetic transition data. The loop is still supervised through exact intermediate targets. Latent sampling, learned halting, re-entry adaptation, and particle updates are disabled.

The current model should therefore be described as a repaired and trained recurrent-depth Qwen substrate, not as an active GRAM-like stochastic reasoner.

---

## 4. What remained aligned with the original vision

The project did preserve several important commitments:

1. **Shared recurrent computation.** A fixed Qwen block is reused over latent state.
2. **Persistent hidden-state reasoning.** Intermediate computation occurs in hidden space rather than by emitting a full textual chain at every step.
3. **Depth and width were treated as distinct axes.** The program did not mistake more loops for more trajectories.
4. **Multi-trajectory infrastructure exists.** The wrapper can repeat inputs, sample per-loop latents, preserve trajectory logits, and evaluate candidate sets.
5. **Halting infrastructure exists.** PonderNet-style and later loop-control components were implemented and diagnosed.
6. **Candidate conversion was used as the correct gate.** The project learned not to accept pairwise distance or unique strings as evidence unless failed groups gained correct candidates.
7. **Selection infrastructure exists.** Best-of-K, voting, reliability, symbolic verification, and task-specific selectors were developed.
8. **The stochastic line was paused, not erased.** The latent and SVGD modules remain in the codebase.

These assets substantially lower the cost of returning to the original thesis.

---

## 5. What the existing evidence actually establishes

### 5.1 Supported conclusions

- A pretrained Qwen can be reorganized into a recurrent-depth architecture and made trainable.
- Reusing a middle block introduces a real loop-closure distribution problem that must be repaired.
- Correct input/prelude re-injection is essential.
- The repaired model can learn nontrivial intermediate latent transitions under direct chain supervision.
- Some learned chains survive outcome-only annealing.
- Geometric particle diversity is not sufficient; more unique trajectories can produce fewer correct candidates.
- Candidate-set quality and selector conversion are better metrics than hidden distance alone.
- A weak deterministic substrate makes stochastic-width results uninterpretable.

### 5.2 Unsupported conclusions

- GRAM does not work on Qwen.
- Stochastic latent reasoning does not help language models.
- Independent trajectory width cannot improve the repaired recurrent model.
- SVGD is necessary for GRAM-style reasoning.
- SVGD is definitively useless on a competent stochastic substrate.
- Learned per-trajectory halting has been fairly tested together with proper variational width.
- The current model has retained the original low-rank, small-update surgical character.
- Current synthetic and natural-surface positives imply broad reasoning gains on GPQA, ARC-AGI, or other hard benchmarks.

### 5.3 The proper label for the early negative result

Use:

> Naive state-conditioned Gaussian injection and SVGD-style repulsion did not reliably improve correct-candidate coverage on the early recurrent Qwen checkpoints. These tests predated full loop-closure repair and did not implement GRAM's target-conditioned variational posterior/prior training.

Do not use:

> GRAM-style stochastic recurrence failed.

### 5.4 Direct corroboration from GRAM's ablation

The paper's ablation predicts our qualitative failure mode:

| Mechanism | Sudoku | N-Queens 8x8 | Interpretation |
|---|---:|---:|---|
| Full GRAM | 93.96 | 99.69 | Learned guidance and stochastic sampling together |
| Without stochastic guidance | 82.87 | 72.91 | Deterministic recursive baseline behavior |
| Stochasticity only | 94.88 | 50.27 | Noise can preserve single-solution performance while collapsing multimodal coverage |
| Guidance only | 0.00 | 0.00 | Target-conditioned direction without stochasticity overfits |
| TRM with stochastic decoder | 82.87 | 71.66 | Output sampling is not latent trajectory learning |
| TRM with random initialization | 78.53 | 71.82 | Naive initial-state noise is insufficient |

This table changes the evidentiary status of our early experiments. Their negative outcome is not surprising evidence against GRAM. It is consistent with GRAM's claim that stochasticity must be trained through target-conditioned guidance.

### 5.5 The transplantation question is a distinct contribution

GRAM demonstrates the mechanism in small, task-specific recursive reasoners trained for structured problems. This project asks a different question: can the same probabilistic-width principle be transplanted into a retrofitted pretrained language model whose deterministic recurrent mechanism has already been installed and characterized?

That structural difference is the contribution opportunity. It also gives this project a supervision asset not normally available in terminal-answer training: exact intermediate state targets. A positive Phase G result would therefore establish more than replication. It would show that guided stochastic width can coexist with pretrained language representations, model surgery, and an explicitly trained deterministic transition substrate.

---

## 6. The largest strategic deviations

### 6.1 From learned stochastic trajectories to supervised deterministic chains

This is the largest conceptual shift. The project exchanged uncertainty over latent paths for exact intermediate-state labels and forced depths. It gained identifiability and debuggability, but stopped addressing the original question of multiple internally valid reasoning paths.

### 6.2 From independent trajectory sampling to interacting particles

SVGD was an extension, not a GRAM reproduction. In GRAM, trajectories are independent samples from the learned prior and are compared or selected after generation. In our SVGD mode, particles directly alter one another inside the recurrent computation. That may be useful, but it defines a different probabilistic object.

### 6.3 From variational guidance to standard-normal regularization

GRAM's posterior is conditioned on the target and teaches the prior where successful stochastic paths lie. Our latent head was conditioned only on the current hidden state and regularized against `N(0,I)`. The anti-collapse reward encouraged distance, but it did not provide a principled signal for diverse correctness.

### 6.4 From economical adaptation to full recurrent-block unfreezing

The early claim emphasized LoRA, a bridge, and a small latent module. Later capacity experiments showed that control modules alone were insufficient, and the full 12-layer recurrent block was unfrozen. The current 182M trainable parameters are a substantial adaptation budget.

This does not negate the architectural result, but it changes the claim from "minimal parameter surgery" to "partial-model retraining concentrated in a recurrently reused block."

### 6.5 From broad reasoning traces to mechanism-specific curricula

Opus/Fable and other distilled reasoning traces were initially central. The eventual keeper was trained primarily on programmatic state-transition tasks and their natural-language variants. This was the right way to isolate recurrence, but it postpones the question of whether the mechanism helps open-ended reasoning.

### 6.6 From multi-solution coverage to mostly single-answer scoring

The motivating advantage of stochastic width is strongest when several solution modes are valid or when a verifier can exploit a diverse candidate set. Much of the later evaluation used unique-answer MCQ or exact state-transition tasks. Those are appropriate for deterministic depth, but not decisive for stochastic width.

---

## 7. Recommended return path: Phase G

If the original thesis remains the goal, the next stochastic phase should be designed as a clean GRAM-inspired test rather than another SVGD sweep.

### 7.1 Opening gate

Phase G-alpha opens only when both conditions are met:

1. **Abductive-injective task gate:** the multimodal task family exists and returns exact valid preimage sets, giving a known denominator for coverage.
2. **Guardrail gate:** the selected deterministic keeper remains green on its locked regression receipts.

Router-B or any learned selector is explicitly removed as a G-alpha precondition. Oracle candidate coverage can test whether stochastic width discovers additional valid solutions before a selector is built.

G-alpha has two locked comparators:

1. **Latent width versus answer-head sampling:** learned latent `K` sampling must beat temperature sampling from the answer head at matched `K`.
2. **Width versus depth at iso-compute:** parallel trajectory width must beat additional deterministic recursion under a matched compute budget.

G-beta opens only after a G-alpha coverage win. It contains learned candidate selection, per-trajectory halting, and SVGD strictly as a later ablation.

### G0. Freeze a deterministic keeper

- Select the best repaired deterministic recurrent checkpoint.
- Record fixed K=1, forced-depth baselines.
- Freeze the prelude, coda, and initially most or all of the deterministic recurrent transition.
- Treat this checkpoint as the shared substrate for every width arm.

**Gate:** deterministic performance and state-transition receipts reproduce from a clean runtime.

### G1. Implement the missing variational mechanism

Add two distributions at each recurrent step:

```text
prior:     p_theta(epsilon_t | state_t, input)
posterior: q_phi(epsilon_t | state_t, input, target)
```

Training samples from the posterior. Inference samples from the prior. Optimize:

```text
predictive loss
+ beta_kl * KL(q_phi || p_theta)
+ optional deep supervision
```

Do not use SVGD in this phase. Do not use a standard-normal KL as the principal objective.

Our substrate provides an additional supervision option that the task-specific GRAM experiments did not require: exact intermediate states. The posterior can therefore condition each transition on the gold next state, not only on the terminal answer. On genuinely multimodal tasks, that target must be a sampled valid chain rather than an arbitrarily fixed canonical chain, or the posterior itself will collapse the valid modes.

The first implementation should inject stochasticity at the slow, high-level transition. GRAM reports that low-level noise did not help in its hierarchical model. The closest mapping in this architecture is the re-entry bridge/state update, not arbitrary noise inside attention or MLP sublayers. This mapping is our transplantation hypothesis, not a direct architectural equivalence asserted by the paper.

Training defaults to test, rather than assume:

- KL balancing initialized near the paper's reported `0.8` setting.
- Exponential moving average of model weights.
- Per-task KL coefficient sweeps because the paper reports task sensitivity.
- Posterior-collapse diagnostics at every loop.

**Gate:** K=1 stochastic inference retains deterministic correctness while different random seeds produce measurable but bounded state variation.

### G2. Test width without interaction

Run `K=1,2,4,8` independent prior samples at matched maximum depth. Report:

- Per-trajectory accuracy.
- Oracle candidate coverage.
- Unique correct candidates.
- Semantic or structural solution diversity.
- Compute-normalized comparison against deeper K=1 runs.
- Calibration of trajectory confidence.

Use at least one true multi-solution task, such as N-Queens or graph coloring, plus a verifier-backed reasoning task.

For the abductive-injective family, report the GRAM-style coverage measure directly:

```text
coverage = unique valid solutions found / total valid solutions
```

Our generator has an advantage for this test: it returns exact preimage sets, so the denominator is exact rather than estimated or recovered through external enumeration.

**Gate:** increasing K improves oracle coverage on failed K=1 cases without a material drop in per-trajectory correctness.

The primary budget comparison is modeled on GRAM's width result: many shallower independent samples versus one much deeper deterministic trajectory at comparable total recurrent transitions. Report both total transition count and sequential latency so width is not credited for hidden extra compute.

The reference claim to mirror is GRAM's comparison in which `N=20` samples at 16 iterations outperform deterministic baselines at 320 iterations under a comparable transition budget. Our preregistration must choose its own feasible values before results exist, but preserve the same arithmetic and latency distinction.

### G3. Train a latent process reward model

Train `v_psi(z_t)` or an equivalent trajectory-level scorer to predict final correctness. Compare:

- Mean-logit aggregation.
- Majority vote.
- Self-consistency.
- Oracle best-of-K.
- Learned latent scorer.
- External verifier where available.

**Gate:** the selector converts a statistically meaningful portion of oracle coverage into top-1 accuracy on held-out tasks.

This is G-beta, not a prerequisite for G-alpha. A failed selector must not obscure a positive or negative answer to whether stochastic width increased oracle coverage.

### G4. Restore adaptive depth per trajectory

Only after width works, train or calibrate per-trajectory halting. Compare:

- Fixed depth.
- PonderNet-style halting.
- Q-learning-style ACT closer to GRAM.

Report the joint distribution of trajectory depth, correctness, and selected candidate.

**Gate:** adaptive depth reduces expected compute at matched coverage or improves coverage at matched compute.

### G5. Test SVGD as an optional extension

Only if independently sampled trajectories still collapse after proper variational training, compare:

- Independent prior samples.
- Diversity regularization only.
- SVGD interaction.

The primary endpoint remains new correct candidates and selector-converted accuracy, not hidden-space distance.

**Gate:** SVGD must beat the independent-sampling baseline under paired seeds and matched compute.

This experiment remains strictly downstream of a G-alpha coverage win. If independent prior samples do not beat answer-head sampling or iso-compute depth, additional repulsion geometry is not justified.

---

## 8. Minimal architectural specification for Phase G

The minimum credible implementation should include:

1. A prior network and a separately parameterized or partially shared posterior network.
2. Target conditioning that cannot leak into inference.
3. One stochastic residual per loop and trajectory.
4. Explicit storage of per-loop prior/posterior statistics.
5. Transition-level KL, not only aggregate standard-normal KL.
6. Per-trajectory logits that are never averaged before candidate extraction.
7. Independent RNG streams and seed manifests per trajectory.
8. A K=1 identity/parity test.
9. A posterior-teacher/prior-student evaluation to measure amortization gap.
10. Candidate-level records sufficient for paired selector analysis.

Recommended implementation order:

```text
deterministic state transition
  -> posterior stochastic residual during training
  -> prior learns posterior paths through KL
  -> independent prior trajectories at inference
  -> candidate selector
  -> adaptive halting
  -> optional interacting-particle extension
```

---

## 9. Proposed claims discipline

### Claims supportable now

- A pretrained Qwen middle block can be converted into a recurrently reused latent transition with corrected re-entry and targeted retraining.
- The resulting substrate can learn multi-step hidden-state transformations under exact intermediate supervision.
- Loop-closure topology, gradient connectivity, and data support are first-order determinants of whether recurrence learns.
- Naive trajectory separation does not guarantee useful candidate diversity.

### Claims requiring Phase G

- Stochastic latent recurrence improves reasoning coverage.
- Width scaling complements depth scaling in recurrent Qwen.
- A learned selector converts latent diversity into accuracy.
- Structural diversity survives later SFT or reinforcement better than weight-encoded diversity.
- The recurrent-particle model outperforms an equivalently trained dense Qwen on hard-tail benchmarks.

---

## 10. Bottom line

The program diverged from its original GRAM-inspired destination in three layers:

1. **From the start, by approximation.** The June latent head was not GRAM's target-conditioned variational trajectory model.
2. **During early experimentation, by extension.** SVGD particle interaction and kernel geometry became a separate anti-collapse research line.
3. **After diagnostic failures, by strategic necessity.** Stochasticity, learned halting, and K>1 were disabled while the recurrent substrate was repaired and trained deterministically.

That detour produced valuable knowledge and a much better substrate. It also means the original research question remains open.

The project should now make an explicit choice:

- Continue the deterministic recurrence and natural-surface transfer program as its own contribution; or
- Use the validated deterministic keeper to run the first faithful test of the original stochastic-width thesis.

The strongest program is likely both, but with clean branch separation and a shared deterministic baseline. The deterministic branch establishes that the recurrent machine works. Phase G would establish whether a learned probability distribution over its trajectories adds something depth alone cannot.

---

## Source and code anchors

### Primary research source

- Baek, Jo, Kim, Ren, Bengio, and Ahn, [Generative Recursive Reasoning](https://arxiv.org/pdf/2605.19376).
- [GRAM project page](https://ahn-ml.github.io/gram-website/).

### Project implementation anchors

- `models/recurrent_wrapper.py`: recurrent loop, trajectory replication, halting-weighted decoding, latent injection, SVGD hook, trajectory-logit averaging.
- `models/latent_policy.py`: state-conditioned Gaussian latent head and standard-normal KL.
- `models/svgd.py`: interacting-particle update and kernel geometry.
- `models/bridge.py`: recurrent prelude/state bridge.
- `training/train_phase2_stochastic.py`: early stochastic multi-trajectory training.
- `training/train_unfrozen_recurrent.py`: current full recurrent-block training path.
- `colab/run_stage5_natural_surface_transfer.py`: current deterministic natural-surface configuration.
- Commit `277f5b4`: initial recurrent Qwen SVGD scaffold.
- Commit `96efe39`: corrected recurrent re-entry prelude injection.
