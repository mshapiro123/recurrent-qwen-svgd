# Arm G / Phase G Guided Stochastic Width: Publication Handoff

Date: 2026-07-18
Program: recurrent Qwen guided stochastic width
Analysis status: complete
Final registered reading: `BOTH_FAIL`
Final program status: current frozen-substrate re-entry conditioning line closed

## Technical summary

Arm G tested whether a repaired deterministic recurrent Qwen could be extended
from one latent trajectory to a learned distribution over valid trajectories.
The implementation borrowed GRAM's central training structure: a
target-conditioned posterior teaches a conditional prior, while inference uses
the prior alone. The pretrained Qwen keeper and its recurrent mechanism were
frozen so any coverage gain would be attributable to stochastic guidance.

The program produced one exploratory width result and three increasingly
specific causal tests:

1. The initial guided-width arm at `K=20` achieved mean exact-set coverage
   `0.3049`, beating one iso-compute deep trajectory by `+0.2506` coverage
   points, but losing to entropy-matched answer-head sampling by `-0.0401`.
   The test also missed one K=1 preservation cell.
2. A curriculum autopsy showed that all 2,048 training prompts were unique and
   each exposed only one selected target. The posterior therefore had no
   repeated-prompt counterfactual evidence from which to learn target control.
   This made the initial width result non-identifying rather than a valid
   negative on guided stochastic reasoning.
3. The corrected repeated-prompt A0 exposed every prompt with multiple valid
   selected chains. Both registered KL arms then failed direct posterior
   target control: selected-target fidelity remained `22.6-23.6%`, posterior
   lift over the prior stayed between `-0.9` and `+1.9` points, and only
   `4/32` groups switched with the requested target versus a required `24/32`.
4. Multiplying the learned posterior residual up to `100x` proved that the
   additive route had raw causal influence but no target-aligned influence.
   Predictions changed frequently only as target fidelity and validity
   deteriorated. The locked verdict was `NO-CHANNEL`.
5. The terminal oracle probe removed latent inference and KL entirely. It
   supplied the true next selected-chain symbol at every transition to
   parameter-matched additive and FiLM conditioners trained against per-loop
   chain labels. Both failed. Additive controlled `14.35%` of non-default
   transitions and FiLM `15.74%`, far below the `85%` gate. Overall control
   was `30.82%` and `28.52%`, versus `90%`; legality was `54.10%` and
   `56.39%`, versus `95%`.

The strongest claim supported by this sequence is:

> On this frozen recurrent Qwen substrate, neither a learned additive
> stochastic residual nor a parameter-matched oracle additive or FiLM
> conditioner established reliable control over non-default recurrent
> transitions through the tested high-level re-entry interface. Increasing
> perturbation magnitude changed outputs but did not create target-aligned
> branching.

This is a bounded architecture-and-interface result. It does not refute GRAM,
stochastic recurrent width, or guided latent reasoning in general.

## 1. Scientific question

The deterministic recurrent model implements:

```text
input -> recurrent state update -> selected depth -> one trajectory -> answer
```

Arm G asked whether the same installed mechanism could support:

```text
input -> conditional latent transition distribution
      -> K independently sampled recurrent trajectories
      -> multiple valid candidate solutions
```

Three requirements were separated deliberately:

1. **Preservation:** installing stochastic guidance must not destroy the
   deterministic keeper.
2. **Target control:** when the training posterior is shown one selected valid
   chain, it must causally steer computation toward that chain.
3. **Useful width:** prior samples must cover more distinct valid solutions
   than output sampling and must beat extra depth at matched compute.

The gate order was preservation, posterior control, coverage, then selection.
This prevented generic output variation from being mislabeled as guided latent
reasoning.

## 2. Relation to GRAM and the early stochastic work

The early particle/SVGD experiments injected stochasticity and encouraged
trajectory separation, but did not include the target-conditioned posterior
that GRAM's own ablations identify as essential on multi-solution tasks. Those
runs therefore tested noise and geometric diversity, not GRAM-style learned
guidance.

Arm G was the first rigorous return to the original width question after:

- repairing loop closure and prelude re-injection;
- verifying bridge gradient flow;
- installing and characterizing deterministic recurrence;
- creating a task with exact multimodal solution sets;
- freezing a deterministic keeper that passed the task's substrate gate.

Arm G remains a transplantation experiment rather than a reproduction. GRAM's
models are task-specific recursive reasoners trained from scratch. This work
uses a surgically modified pretrained Qwen2.5-0.5B-Instruct, a repaired
recurrent bridge, explicit intermediate-chain labels, and a frozen deterministic
substrate.

## 3. Shared experimental substrate

### 3.1 Keeper

Every canonical Arm G experiment used the same deterministic keeper:

```text
SHA-256
0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f
```

The active frozen-lineage hash was:

```text
b9ab5ba8630670197c3036631831949292113d60e48b6c9e7acc8dc113e37ec9
```

The recurrent block, repaired bridge substrate, coda, embeddings, reader,
halting modules, and pretrained Qwen weights remained frozen during the guided
and oracle-interface experiments.

### 3.2 Task

The task was a natural-language branching relation over 20 symbols. Each row
specified a relation table, a start symbol, and a composed depth. Multiple
terminal symbols could be valid, giving exact finite target sets against which
coverage and selected-chain control could be measured.

The initial frozen coverage sets contained:

- 512 calibration rows;
- 512 test rows;
- depths 1-4;
- reachable-set strata `2`, `3-4`, `5-8`, and `9-16`.

The corrected control curriculum contained:

| Split | Base prompts | Selected-chain variants | Multi-target groups |
|:---|---:|---:|---:|
| Training | 512 | 1,899 | 512 |
| Held out | 32 | 106 | 32 |

Within each group, the prompt, relation table, start state, depth, and full valid
terminal set were fixed. Only the selected valid terminal and intermediate
chain changed. Train and held-out base prompts were disjoint.

## 4. Experiment G-1: initial guided-width coverage

### 4.1 Model and training

Only the following parameters trained:

- `phase_g_prior_head.*`;
- `phase_g_posterior_head.*`;
- `phase_g_injection_scale`.

The prior saw the current recurrent state. The posterior additionally saw the
gold next-state embedding. A latent sample was projected through a fixed
orthonormal map and added to the high-level re-entry state.

Three KL coefficients were trained for 1,000 steps:

```text
0.0001, 0.001, 0.01
```

Shared settings:

| Setting | Value |
|:---|---:|
| Latent dimension | 64 |
| Trajectories per training row | 4 |
| Learning rate | `1e-4` |
| Initial injection scale | `0.001` |
| KL balance | `0.8` |
| EMA | `0.999` |
| Precision | bfloat16 |

The selected calibration arm was KL `0.001`, EMA weights.

### 4.2 Locked comparators

At `K=20`, latent-prior coverage was compared on identical frozen rows with:

1. entropy-matched answer-head temperature sampling at `K=20`;
2. one deterministic trajectory with `20x` the recurrent depth.

The absolute mean-coverage margin was locked at `0.05`. K=1 preservation also
required pooled accuracy within 3 points and every depth/stratum cell within
8 points of the deterministic keeper.

### 4.3 Result

| K=20 arm | Mean coverage |
|:---|---:|
| Learned latent prior | `0.3049` |
| Entropy-matched answer sampling | `0.3450` |
| Iso-compute depth | `0.0543` |
| Posterior teacher | `0.3063` |

Paired comparisons:

- prior minus iso-depth: `+0.2506`; helped 337, hurt 20, tied 155;
  one-sided `p=9.78e-76`;
- prior minus temperature: `-0.0401`; helped 24, hurt 63, tied 425;
  one-sided `p=0.999994`;
- posterior minus prior: `+0.00145`.

The selected arm failed the K=1 preservation gate despite close pooled
performance:

- learned arm: `383/512`;
- deterministic expectation: `389/512`;
- pooled absolute difference: `1.17` points, within tolerance;
- depth 4, reachable set `5-8`: `65.63%` versus `75.00%`, a `9.38` point
  difference above the `8` point cell tolerance.

The registered verdict was that G-alpha did not clear both comparators.

### 4.4 Why this was not the final causal verdict

The post-run curriculum audit found:

```text
2,048 rows
2,048 prompt groups
0 repeated-prompt groups
0 groups with multiple selected targets
```

The posterior was no more faithful to the selected target than the prior:

| K | Prior target present | Posterior target present | Posterior-prior |
|---:|---:|---:|---:|
| 1 | `29.49%` | `28.52%` | `-0.98` pp |
| 20 | `32.81%` | `32.62%` | `-0.20` pp |

The training data never required the model to distinguish two selected chains
for the same prompt. The apparent width-over-depth result could therefore arise
from ordinary stochastic output variation rather than learned conditional
trajectory control. The result remains a useful exploratory receipt but is not
evidence that guided latent width succeeded or failed.

## 5. Experiment G-A0: corrected multi-target posterior control

### 5.1 Direct causal question

For repeated identical prompts with different selected valid chains, does the
posterior change its terminal decision toward the selected target, relative to
the target-blind prior?

The locked gate required:

| Criterion | Requirement |
|:---|---:|
| Repeated multi-target groups | `>=32` |
| Posterior selected-target fidelity | `>=0.60` |
| Posterior-minus-prior fidelity lift | `>=+0.15` |
| One-sided paired sign test | `p<=0.05` |
| Posterior switching groups | `>=24/32` |

The primary arm used KL `0.001`. One confirmation arm at KL `0.0001` was
authorized only if the primary blocked.

### 5.2 Training integrity

Both arms:

- trained for 1,000 steps;
- used four trajectories per row;
- sampled uniformly over base prompts, then target variants;
- recorded 1,000 frozen-gradient assertions;
- preserved the exact active-lineage hash;
- showed nonzero gradient norms;
- showed separated posterior/prior statistics;
- recorded zero posterior-collapse fraction;
- increased the injection residual RMS substantially.

The failure was not a dead graph or a missing optimizer group.

### 5.3 Results

| Metric | KL `0.001` posterior | Prior | KL `0.0001` posterior | Prior |
|:---|---:|---:|---:|---:|
| Selected-target fidelity | 24/106, `22.64%` | 25/106, `23.58%` | 25/106, `23.58%` | 23/106, `21.70%` |
| Posterior-prior lift | `-0.94` pp |  | `+1.89` pp |  |
| Switching groups | `4/32` |  | `4/32` |  |
| Validity | `74.53%` | `78.30%` | `79.25%` | `77.36%` |
| Paired helped / hurt / tied | 1 / 2 / 103 |  | 2 / 0 / 104 |  |
| One-sided paired p | `0.875` |  | `0.25` |  |

Both arms missed every target-control threshold. Coverage was correctly not
rerun.

## 6. Experiment G-A0b: forced-injection causal probe

### 6.1 Question and gate

The learned posterior residual was multiplied at inference by:

```text
1, 3, 10, 30, 100
```

No parameter was trained or mutated. `CHANNEL-EXISTS` required at least `16/32`
switching groups while K=1 validity remained strictly above `0.50`.
`NO-CHANNEL` applied if switching remained below 8 at every dose or reached 16
only after validity fell below 0.50.

### 6.2 Results

| KL | Multiplier | Changed rows | Switching | Target fidelity | Validity |
|:---|---:|---:|---:|---:|---:|
| `0.001` | 1 | 0/106 | 4/32 | `22.64%` | `74.53%` |
| `0.001` | 3 | 7/106 | 3/32 | `22.64%` | `75.47%` |
| `0.001` | 10 | 12/106 | 5/32 | `20.75%` | `70.75%` |
| `0.001` | 30 | 45/106 | 15/32 | `17.92%` | `63.21%` |
| `0.001` | 100 | 83/106 | 22/32 | `13.21%` | `43.40%` |
| `0.0001` | 1 | 0/106 | 4/32 | `23.58%` | `79.25%` |
| `0.0001` | 3 | 14/106 | 4/32 | `23.58%` | `73.58%` |
| `0.0001` | 10 | 18/106 | 4/32 | `21.70%` | `71.70%` |
| `0.0001` | 30 | 45/106 | 9/32 | `16.04%` | `63.21%` |
| `0.0001` | 100 | 90/106 | 13/32 | `10.38%` | `46.23%` |

At `100x`, the primary residual changed 83 predictions and reached 22 switching
groups, but validity had fallen to `43.4%` and target fidelity to `13.2%`.
The route was causally active but destructive. The registered verdict was
`NO-CHANNEL`.

### 6.3 Scorer erratum

The GPU evaluation completed before a CPU scorer error. The scorer expected
numeric multiplier keys in insertion order, while `sort_keys=True` serialized
them lexicographically. The repair validated the exact factor set independent
of object-key order and iterated factors numerically. It changed no predictions,
metrics, thresholds, or verdict and added a sorted-JSON regression test.

## 7. Experiment G-T: terminal oracle re-entry interface probe

### 7.1 Purpose

The terminal probe removed the remaining variational ambiguity. Instead of
asking a prior or posterior to infer a useful latent instruction, it supplied
the true next selected-chain symbol directly at every transition.

Two parameter-matched routes were trained:

1. **Additive:** two learned command branches were folded into an additive
   residual.
2. **FiLM:** the same branches emitted `gamma - 1` and `beta`, applying
   `gamma * h + beta`.

Both consumed:

- the pooled current loop-input state;
- the frozen token embedding of the true next symbol.

Both output layers were zero-initialized, giving exact step-zero identity.

### 7.2 Matched contract

| Setting | Value |
|:---|---:|
| Training variants | 1,899 |
| Held-out variants | 106 |
| Held-out groups | 32 |
| Held-out transitions | 305 |
| Steps | 1,500 |
| Seed | `20260718` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| EMA | `0.999` |
| Bottleneck | 256 |
| Trainable parameters per route | 1,385,728 |
| Objective | per-loop commanded-chain CE |

Only `oracle_reentry_conditioner.*` trained. Each arm recorded:

- 1,500 frozen-gradient assertions;
- 1,500 conditioner-gradient-liveness assertions;
- exact zero-conditioning identity;
- identical frozen lineage before and after.

No KL, latent sampling, coverage, selector, halting, particles, or SVGD was
present.

### 7.3 Locked gates

An arm passed only if every gate passed:

| Gate | Threshold |
|:---|---:|
| Non-default branch control | `>=0.85` |
| Overall transition control | `>=0.90` |
| Transition legality | `>=0.95` |
| Terminal validity | `>=0.71` |
| Zero-conditioning identity | exact |
| Frozen keeper lineage | exact |

The terminal reading was locked as:

- FiLM pass, additive fail: interface localized to combination rule;
- both fail: re-entry conditioning on the frozen substrate closed;
- both pass: prior A0 failure localized to variational objective/amortization;
- additive pass, FiLM fail: unexpected asymmetry and pause.

### 7.4 Primary results

| Metric | Additive | FiLM | Gate |
|:---|---:|---:|---:|
| Non-default control | 31/216, `14.35%` | 34/216, `15.74%` | `85%` |
| Overall control | 94/305, `30.82%` | 87/305, `28.52%` | `90%` |
| Transition legality | 165/305, `54.10%` | 172/305, `56.39%` | `95%` |
| Terminal validity | 74/106, `69.81%` | 79/106, `74.53%` | `71%` |
| Zero-conditioning identity | exact | exact | exact |
| Frozen lineage | exact | exact | exact |

FiLM passed terminal validity but failed all command and legality gates.
Additive failed terminal validity as well. Both failed the terminal probe, and
the registered reading was `BOTH_FAIL`.

### 7.5 Localization by command type

The unconditioned keeper's prediction defined the default transition. A
non-default command required the conditioner to redirect the keeper.

| Route | Default control | Non-default control | Gap |
|:---|---:|---:|---:|
| Additive | `70.79%` | `14.35%` | `56.43` pp |
| FiLM | `59.55%` | `15.74%` | `43.81` pp |

The conditioners retained much more of the keeper's existing behavior than they
could override. Non-default transitions were the core causal test, and both
routes remained far below gate.

### 7.6 Localization by loop index

| Route | Metric | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|:---|:---|---:|---:|---:|---:|
| Additive | control | `58.49%` | `20.00%` | `16.18%` | `7.32%` |
| Additive | legality | `97.17%` | `38.89%` | `29.41%` | `17.07%` |
| FiLM | control | `49.06%` | `24.44%` | `11.76%` | `12.20%` |
| FiLM | legality | `97.17%` | `48.89%` | `22.06%` | `24.39%` |

The first transition remained mostly legal, but both control and legality
collapsed after re-entry. This is post-hoc localization, not a separate
registered gate. It supports the interpretation that the trained command did
not persist as a stable transition instruction through deeper recurrence.

### 7.7 Training was live

The first and last 100-step means were:

| Route | Loss, first 100 | Loss, last 100 | Grad norm, first 100 | Grad norm, last 100 | Final-window residual RMS ratio |
|:---|---:|---:|---:|---:|---:|
| Additive | `4.4775` | `1.9300` | `89.46` | `22.28` | `0.1103` |
| FiLM | `4.5759` | `1.6712` | `81.47` | `28.80` | `0.1270` |

The conditioners trained, gradients remained live, losses fell, and the
conditioned residual became substantial. The held-out failure is not explained
by zero gradients, identity-locking, or a missing optimizer group.

## 8. Integrated interpretation

### 8.1 What the sequence rules out

The evidence removes four progressively narrower explanations:

1. **The initial negative was only missing GRAM guidance.** A proper prior and
   target-conditioned posterior were added.
2. **The posterior lacked identifying examples.** Repeated identical prompts
   with multiple selected valid chains were added.
3. **The learned residual was simply too small.** It was amplified up to
   `100x`, changing most outputs at high dose.
4. **Additive combination alone was the bottleneck.** A parameter-matched FiLM
   oracle route received the true next symbol and still failed.

The resulting boundary is not "stochasticity cannot alter the model." It can.
The boundary is "the tested frozen high-level re-entry interfaces did not turn
a selected-chain command into reliable non-default transition control."

### 8.2 What remains unresolved

The terminal probe did not evaluate command accuracy on the training rows. It
therefore does not distinguish perfectly between:

- inability of the route to fit selected-chain commands at all;
- fitting the training variants but failing held-out prompt generalization.

The falling training loss and live residual show optimization occurred, but are
not substitutes for a train-set transition-control evaluation.

The experiment also cannot separate:

- conditioning location from frozen-substrate rigidity;
- bottleneck rank from interface semantics;
- token-embedding command representation from the combination rule;
- the effect of allowing a small part of the recurrent block to adapt.

These are new architecture questions, not continuations of the completed A0
run.

### 8.3 Why the initial width-over-depth number is not a success claim

The latent prior beat extra depth at matched compute, but answer-head sampling
beat the latent prior, the posterior did not control the chosen target, and the
training curriculum was non-identifying. The result demonstrates that sampling
can increase exact-set coverage on this task. It does not demonstrate that the
model learned a target-guided distribution over reasoning trajectories.

## 9. Claims supported

The paper can state:

> A GRAM-inspired prior/posterior transplant on a frozen recurrent Qwen
> produced stochastic coverage but did not outperform entropy-matched output
> sampling. A corrected repeated-prompt intervention showed that the posterior
> did not control the selected valid target. Amplifying the learned additive
> residual changed outputs only by reducing target fidelity and validity.
> Finally, even parameter-matched oracle additive and FiLM conditioners supplied
> with the true next symbol failed to control held-out non-default transitions,
> localizing the current boundary to conditioning a frozen recurrent substrate
> through the tested re-entry interfaces.

A shorter abstract-safe form:

> Guided stochastic width did not transfer through the tested frozen re-entry
> interface: latent perturbations increased variation without target-aligned
> control, and oracle additive and FiLM conditioners also failed held-out
> non-default transition gates.

## 10. Claims not supported

Do not claim:

- GRAM has been reproduced;
- GRAM is false;
- stochastic recurrent width is impossible;
- the recurrent state contains no branch information;
- the posterior contains no target-dependent statistics;
- FiLM is superior to additive because its terminal validity was higher;
- extra depth generally loses to width;
- the initial `+0.2506` width-over-depth coverage delta is guided reasoning;
- SVGD or particles were tested under the corrected posterior-control design;
- a conditioner cannot fit the training data, because train-set command control
  was not evaluated;
- unfreezing part of the recurrent block would fail;
- the result generalizes beyond this synthetic branching-relation family;
- seed robustness, because each canonical training design used one locked seed.

## 11. Publication placement

Arm G should not be merged into the central deterministic claim of Paper One.
It is best treated as:

1. a separate Paper Two result on learned stochastic width and causal
   controllability; or
2. an appendix boundary experiment that motivates future conditioning
   architectures.

The strongest contribution is methodological as well as empirical:

- multi-solution curricula must contain repeated prompts with different
  selected targets;
- latent diversity must pass a target-control intervention before coverage is
  interpreted as guided reasoning;
- output changes under larger perturbations are insufficient if validity and
  target fidelity fall;
- oracle route-capacity tests should precede expensive variational retraining.

Recommended figure sequence:

1. initial K-coverage curves for latent prior, temperature sampling, and
   iso-compute depth, visibly labeled exploratory/non-identifying;
2. corrected A0 posterior-versus-prior target fidelity;
3. forced-injection dose curves showing switching, fidelity, and validity;
4. terminal oracle control and legality by loop index for additive and FiLM.

## 12. Program decision

The current Arm G line is complete. No automatic successor is authorized.

Closed:

- additional KL, scale, seed, optimizer, or duration sweeps on the same A0
  route;
- coverage evaluation from the corrected A0 checkpoints;
- selector/LPRM, learned per-trajectory halting, particles, or SVGD on this
  route;
- another additive-versus-FiLM variational training run on the unchanged frozen
  keeper.

Banked:

- the initial exploratory coverage receipt;
- the corrected posterior-control negative;
- the forced-injection `NO-CHANNEL` result;
- the terminal oracle `BOTH_FAIL` result;
- the gate-order methodology.

Recommended immediate work:

1. freeze Arm G artifacts and manuscript language;
2. finish Paper One around deterministic recurrence and Arm E;
3. prepare Paper Two's causal sequence using the canonical receipts below;
4. decide separately whether a new architecture program is worth opening.

## 13. If the line is reopened

No further GPU run is required to interpret the current experiment. If strategy
authorizes a new architecture, the highest-information sequence is:

1. **Train-set oracle readout.** Evaluate the existing oracle checkpoints on a
   fixed training subset to separate fitting failure from held-out
   generalization failure.
2. **Earlier or internal conditioning.** Inject the command before or inside the
   recurrent block rather than only at high-level re-entry.
3. **Direct transition supervision.** Train against branch logits or a
   transition classifier at the divergence point, not only token CE after the
   recurrent update.
4. **Minimal adaptive substrate.** If a frozen block remains rigid, unfreeze a
   tightly bounded low-rank or normalization/gating subset and rerun the oracle
   gate before restoring a variational objective.
5. **Only after oracle control passes:** reintroduce posterior/prior KL,
   preservation, coverage versus temperature and depth, then selection.

These would constitute a new preregistered program. They are not authorized
extensions of the current one.

## 14. Questions for strategy review

1. Should Arm G be a Paper Two central negative or an appendix boundary study?
2. Is the train-subset oracle evaluation worth one small read-only GPU job, or
   is held-out `BOTH_FAIL` sufficient for manuscript closure?
3. Should the publication headline emphasize `frozen-substrate
   controllability` or the broader `diversity is not guidance` lesson?
4. Does the initial exploratory width-over-depth curve belong in the main
   figure if it is explicitly marked non-identifying, or would it distract
   from the causal sequence?
5. Should the scorer key-order repair be documented in the appendix artifact
   ledger or only in reproducibility notes?
6. Is the single-seed limitation acceptable given that the two KL arms and two
   interface classes converge on the same causal verdict?
7. If future work reopens the line, should the first new intervention be
   earlier conditioning or a minimally adaptive recurrent block?

## 15. Canonical artifact map

### Initial guided width

- Run summary:
  `outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/summary.json`
- Test summary:
  `outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/test/kl_0p001/ema/summary.json`
- Curriculum autopsy:
  `outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/autopsy/multimodal_supervision.json`
- Narrative handoff:
  `docs/STAGE5_PHASE_G_ALPHA_RESULT_AND_CURRICULUM_AUTOPSY_HANDOFF_20260718.md`
- Final run commit:
  `38cad7119704de7e861a5d5cd76ba208fe3feb3f`

### Corrected posterior control

- Run summary:
  `outputs/stage5/stage5_phase_g_multitarget_control_20260718/summary.json`
- Gate lock:
  `docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json`
- Narrative handoff:
  `docs/PHASE_G_A0_POSTERIOR_CONTROL_HANDOFF_20260718.md`
- Final receipt commit:
  `adec0ecc7bced9fc2bef780268ee125e50b61c7f`

### Forced injection

- Run summary:
  `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/summary.json`
- Final gate:
  `outputs/stage5/stage5_phase_g_forced_injection_probe_20260718/gate.json`
- Combined causal handoff:
  `docs/PHASE_G_A0_CAUSAL_PROBE_STRATEGY_HANDOFF_20260718.md`
- Preregistered spec:
  `docs/STAGE5_PHASE_G_FORCED_INJECTION_PROBE_SPEC_20260718.md`
- Final verdict commit:
  `8fc19aca88fb6a1bc0480e829687b42c55c0d6fe`

### Terminal oracle interface

- Preregistration:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/preregistration.json`
- Run summary:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/summary.json`
- Final gate:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/gate.json`
- Additive evaluation:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/eval/additive/summary.json`
- FiLM evaluation:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/eval/film/summary.json`
- Post-hoc localization:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/posthoc_localization.json`
- Probe spec:
  `docs/STAGE5_PHASE_G_ORACLE_INTERFACE_PROBE_SPEC_20260718.md`
- Preregistration commit:
  `94be5922330af30e97dcef0b76bb49c288ee2d74`
- Terminal receipt commit:
  `6aefe0d87bd0dc525caaaa172f9456556dfa8c03`

### Reproducibility

```bash
python eval/analyze_phase_g_oracle_localization.py \
  --gate_json outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/gate.json \
  --additive_trace outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/train/additive/training_trace.jsonl \
  --film_trace outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/train/film/training_trace.jsonl \
  --output_json outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/posthoc_localization.json
```

## Bottom line

Arm G did not establish guided stochastic reasoning. The initial latent sampler
increased coverage relative to extra depth, but ordinary output sampling did
better and the training curriculum did not identify target-conditioned control.
After correcting that curriculum, the posterior remained effectively
target-blind. Larger residuals changed outputs by degrading them. Finally, even
oracle additive and FiLM conditioners supplied with the true next symbol failed
to redirect held-out non-default transitions through the frozen re-entry
interface.

The defensible conclusion is specific and useful: useful stochastic width
requires a controllable transition interface, and this frozen high-level
re-entry design did not provide one. The broader GRAM-inspired question remains
open, but any successor must change the conditioning location or allow a
bounded adaptive substrate before spending compute on another variational
training run.
