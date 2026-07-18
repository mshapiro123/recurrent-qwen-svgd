# Handoff: Phase G-alpha Result and Multimodal-Curriculum Autopsy

**Date:** 2026-07-18
**Program:** Recurrent Qwen, deterministic depth, and guided latent width
**Status:** Phase G-alpha complete; G-beta remains closed
**Primary run:** `stage5_phase_g_alpha_guided_width_20260717`
**Final run commit:** `38cad711`
**Audit implementation:** `47634046`
**Audit receipts:** `048b0210`

## 0. Executive decision

Phase G-alpha did not clear both preregistered comparators.

The result contains one clean positive and one clean negative:

1. **Latent width strongly beat extreme deterministic depth at matched nominal
   compute.** At K=20, prior-sampled latent trajectories exceeded the
   iso-compute depth arm by `+0.2506` mean exact coverage, with `337` helped
   rows, `20` hurt rows, and one-sided sign-test
   `p = 9.78e-76`.
2. **The learned latent prior did not beat ordinary answer-head sampling.** At
   K=20, latent prior coverage was `0.3049`, versus `0.3450` for
   entropy-matched answer-head sampling, a delta of `-0.0401`. The paired
   result was `24` helped, `63` hurt, `425` tied.

The second result does **not** yet support the stronger conclusion that
GRAM-style guided width fails to transplant to the recurrent Qwen substrate.
The post-run audit found that the training curriculum did not expose the model
to conditional multimodality:

- `2,048` training rows formed `2,048` distinct problem groups.
- No prompt appeared more than once.
- No problem appeared with multiple supervised targets.
- Each row stored one sampled valid chain and one terminal target.
- All K=4 trajectories in an optimizer step were trained against that same
  fixed target chain.

The target-conditioned posterior consequently showed no target selectivity.
On the 512-row test set, the posterior teacher was slightly worse than the
prior at reproducing the row's conditioned target:

- K=1 target rate: posterior `0.2852`, prior `0.2949`.
- K=20 target-in-set rate: posterior `0.3262`, prior `0.3281`.
- K=20 paired target-in-set result: `4` helped, `5` hurt, `503` tied.

The defensible verdict is therefore:

> The implemented latent sampler creates width and beats excessive recurrence,
> but the first learned-guidance curriculum did not identify a
> target-conditioned multimodal posterior and did not beat output sampling.
> One bounded, explicitly multi-target correction is warranted before closing
> the GRAM-transplant question.

## 1. Why this experiment was run

The original program sought more than repeated deterministic refinement:

```text
input -> recurrent loop -> learned depth -> one trajectory -> one answer
```

The GRAM-inspired extension asks whether reasoning can instead be represented
as a distribution:

```text
input -> conditional latent prior -> K recursive trajectories
      -> multiple valid candidate solutions -> coverage or selection
```

Phase G-alpha was the first rigorous return to this question after repairing
and characterizing the deterministic recurrent substrate. Its primary test was
oracle coverage, so it did not require a learned selector. G-beta, including
selection, per-trajectory halting, and SVGD as an ablation, was preregistered to
open only if guided latent width beat both locked comparators.

## 2. Experiment design

### 2.1 Frozen deterministic substrate

- Backbone: `Qwen/Qwen2.5-0.5B-Instruct`.
- Keeper checkpoint:
  `outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/restored/natural_step2000.pt`.
- Keeper SHA-256:
  `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
- Recurrent block and established deterministic mechanism remained frozen.
- Active-lineage hash was unchanged across training:
  `b9ab5ba8630670197c3036631831949292113d60e48b6c9e7acc8dc113e37ec9`.
- Frozen-gradient assertions ran throughout training.

Only the following components trained:

- Conditional latent prior head.
- Target-conditioned posterior head.
- Scalar latent injection scale.

The latent was injected at the high-level re-entry state, not inside low-level
transformer sublayers.

### 2.2 Task and frozen sets

The task was a forward, multi-valued branching-relation problem:

- A prompt defines a binary successor table.
- A start symbol and exact depth are given.
- Any symbol reachable after exactly that many transitions is valid.
- Reachable sets are exact, so oracle coverage has an exact denominator.
- Depths: `1, 2, 3, 4`.
- Reachable-set strata: `2`, `3-4`, `5-8`, `9-16`.
- Training rows: `2,048`.
- Calibration rows: `512`.
- Test rows: `512`.

The data generator stored one sampled valid chain per row for posterior
conditioning and per-loop supervision.

### 2.3 Guidance training

Three KL coefficients were tested:

```text
0.0001
0.001
0.01
```

Shared settings:

- `1,000` steps per arm.
- K=4 seeded trajectories per optimizer step.
- Latent dimension `64`.
- Learning rate `1e-4`.
- KL balance `0.8`.
- EMA decay `0.999`.
- Initial injection scale `0.001`.
- Weight decay `0`.
- `bfloat16` model execution.

Final observed training values were approximately:

| KL coefficient | Final KL | Final injection scale |
|---:|---:|---:|
| `0.0001` | `80.49` | `0.001069` |
| `0.001` | `9.59` | `0.001058` |
| `0.01` | `1.38` | `0.001041` |

Raw and EMA checkpoints were evaluated. All three EMA variants passed the
calibration K=1 parity gate; all three raw variants failed it. The selected arm
was:

```text
KL=0.001, EMA weights
```

Selection rule:

```text
highest calibration K=20 prior mean coverage among K=1-parity arms
```

### 2.4 Locked comparators

The selected latent-prior arm was compared on identical frozen test rows
against:

1. **Entropy-matched answer-head sampling.**
   - Target mean entropy: `0.1432`.
   - Achieved mean entropy: `0.1432`.
   - Solved temperature: `0.9999869511`.
2. **Iso-compute deterministic depth.**
   - K trajectories at depth T versus one deterministic trajectory at depth
     K times T.
3. **Posterior teacher.**
   - Used to measure whether target-conditioned posterior information was
     present and transferred to the prior.

Sample counts were:

```text
K = 1, 2, 4, 8, 20
```

The locked absolute mean-coverage margin was `0.05`.

### 2.5 Preservation gate

K=1 behavior was required to remain close to the deterministic keeper:

- Pooled tolerance: `0.03`.
- Per depth-by-stratum cell tolerance: `0.08`.

Final result:

- Current correct: `383/512`.
- Expected correct: `389/512`.
- Pooled absolute delta: `0.01172`, within tolerance.
- One per-cell failure:
  - depth 4, reachable-set `5-8`: delta `0.09375`.

Thus the formal K=1 parity gate failed on one cell even though pooled behavior
was preserved.

## 3. Main results

### 3.1 Overall coverage by K

| Arm | K=1 | K=2 | K=4 | K=8 | K=20 |
|---|---:|---:|---:|---:|---:|
| Latent prior | `0.2716` | `0.2805` | `0.2898` | `0.2953` | `0.3049` |
| Temperature | `0.2713` | `0.2885` | `0.3053` | `0.3234` | `0.3450` |
| Iso-compute depth | `0.2754` | `0.1857` | `0.1315` | `0.0671` | `0.0543` |
| Posterior teacher | `0.2712` | `0.2817` | `0.2885` | `0.2957` | `0.3063` |

At K=20:

| Arm | Mean coverage | Mean unique valid | Valid-sample rate | Full coverage | Duplicate rate |
|---|---:|---:|---:|---:|---:|
| Latent prior | `0.3049` | `0.8750` | `0.7463` | `0.0176` | `0.9386` |
| Temperature | `0.3450` | `0.9531` | `0.7543` | `0.0742` | `0.9326` |
| Iso-compute depth | `0.0543` | `0.1895` | `0.1895` | `0.0000` | `0.0000` |
| Posterior teacher | `0.3063` | `0.8809` | `0.7497` | `0.0176` | `0.9380` |

### 3.2 Paired comparator results at K=20

**Latent prior minus iso-compute depth**

```text
mean coverage delta = +0.250623
helped = 337
hurt = 20
tied = 155
one-sided sign-test p = 9.78e-76
```

This is a decisive width-over-extreme-depth result on the current task.

**Latent prior minus entropy-matched answer sampling**

```text
mean coverage delta = -0.040124
helped = 24
hurt = 63
tied = 425
one-sided sign-test p = 0.999994
```

The learned latent prior did not add value over ordinary output sampling.

**Posterior teacher minus prior**

```text
mean coverage delta = +0.001449
```

This is effectively no amortization advantage.

### 3.3 Coverage by depth at K=20

| Depth | Latent prior | Temperature | Iso-depth | Posterior teacher |
|---:|---:|---:|---:|---:|
| 1 | `0.5234` | `0.6406` | `0.0742` | `0.5273` |
| 2 | `0.3268` | `0.3600` | `0.0456` | `0.3197` |
| 3 | `0.2026` | `0.2224` | `0.0519` | `0.2055` |
| 4 | `0.1667` | `0.1570` | `0.0454` | `0.1728` |

The latent arm lost most strongly at depth 1, lost modestly at depths 2 and 3,
and narrowly exceeded temperature at depth 4. The depth-4 delta was only
`+0.00965`, below the locked `+0.05` margin.

### 3.4 Coverage by reachable-set size at K=20

| Reachable set | Latent prior | Temperature | Iso-depth | Posterior teacher |
|---|---:|---:|---:|---:|
| 2 | `0.3951` | `0.4682` | `0.0618` | `0.3951` |
| 3-4 | `0.2572` | `0.2638` | `0.0450` | `0.2620` |
| 5-8 | `0.1486` | `0.1593` | `0.0530` | `0.1496` |
| 9-16 | `0.1207` | `0.0998` | `0.0347` | `0.1207` |

The latent arm exceeded temperature only on the largest `9-16` stratum, by
`+0.02085`, again below the locked margin. This is a weak but relevant tail
signal for future work, not a passed result.

## 4. Post-run curriculum and posterior audit

The audit was added after observing three linked facts:

- Latent prior coverage was below temperature sampling.
- Posterior-teacher coverage was almost identical to prior coverage.
- Candidate duplication remained very high.

### 4.1 Curriculum exposure

The audit groups rows by the complete underlying problem:

```text
depth + rendered question + start symbol + successor table
```

Result:

| Diagnostic | Value |
|---|---:|
| Training rows | `2,048` |
| Distinct problem groups | `2,048` |
| Groups with repeated prompt | `0` |
| Groups with multiple targets | `0` |
| Maximum distinct targets per problem | `1` |
| Rows in multi-target groups | `0` |
| Mean supervised fraction of valid targets | `0.3678` |

Every training problem appeared exactly once with one sampled chain. The
`posterior_chain_sampling = uniform_local_branch_choice` field described how
that one chain was generated; it did not mean a new target chain was sampled
when the row was revisited.

### 4.2 Posterior target fidelity

The original evaluator scored whether samples were any valid reachable answer.
The new audit instead asks whether the posterior teacher reproduced the exact
target chain endpoint it was conditioned on.

| Metric | Prior | Posterior teacher | Posterior minus prior |
|---|---:|---:|---:|
| K=1 exact target | `0.2949` | `0.2852` | `-0.0098` |
| K=20 target appears | `0.3281` | `0.3262` | `-0.0020` |
| K=20 mean target sample rate | `0.2900` | `0.2903` | `+0.0003` |

Paired K=20 target-in-set result:

```text
helped = 4
hurt = 5
tied = 503
```

The posterior was not materially target-selective.

### 4.3 Structural reading

The training objective had no repeated-input evidence from which to learn:

```text
same input x -> target chain y1
same input x -> target chain y2
same input x -> target chain y3
```

Instead it saw:

```text
input x1 -> one sampled chain y1
input x2 -> one sampled chain y2
...
```

Moreover, all K=4 trajectories for a sampled row received the same fixed
posterior target tensor. The setup could learn noise around a single
input-target association, but it did not explicitly identify alternate valid
reasoning modes for a fixed input.

This matches the empirical signature:

- Posterior and prior are behaviorally indistinguishable.
- KL can be reduced by coefficient choice without creating target selectivity.
- Latent samples add some width but not more useful width than output sampling.
- Duplicate rates remain around `0.94` at K=20.

## 5. What is and is not established

### 5.1 Supported conclusions

1. The repaired recurrent Qwen substrate can execute seeded latent trajectories
   and produce measurable candidate width.
2. Width is much more useful than forcing a single deterministic trajectory to
   extreme depth on this task and checkpoint.
3. The first learned latent guidance implementation did not beat
   entropy-matched answer sampling.
4. The trained posterior was not target-selective.
5. The first Phase G curriculum did not present multiple target chains for the
   same input, so it did not cleanly train conditional multimodality.
6. G-beta remains closed under the preregistration.

### 5.2 Unsupported conclusions

Do not claim:

- GRAM-style guided width cannot transplant to pretrained recurrent Qwen.
- Learned latent width is generally inferior to output sampling.
- The architecture requires SVGD or repulsion.
- Posterior collapse alone explains the result.
- More training steps on the same curriculum would repair the result.
- The narrow depth-4 or large-reachable-set advantages are confirmed tail wins.

### 5.3 Relation to the early stochastic/SVGD negatives

The early stochastic and SVGD runs remain non-dispositive for GRAM-style
width. They:

- preceded the repaired loop-closure path;
- lacked a target-conditioned posterior;
- did not train conditional multimodal guidance; and
- often measured diversity without candidate conversion.

Phase G-alpha is stronger because it used the repaired substrate and proper
prior/posterior machinery. However, its curriculum still omitted the
same-input/multiple-target exposure needed to identify the intended mechanism.

## 6. Recommended bounded correction

Run one correction before closing the GRAM-transplant question. Do not open
G-beta, add SVGD, expand model capacity, or start a broad hyperparameter sweep.

### Stage A: multi-target objective micro-test

Purpose: prove that the posterior can follow a selected valid chain when the
same problem is paired with multiple chains.

Design:

- Freeze the deterministic keeper exactly as in Phase G-alpha.
- Select a small set of base branching problems across depths and reachable-set
  strata.
- For each base problem, construct multiple rows with the identical prompt and
  successor table but distinct valid target chains.
- Prefer exhaustive targets when the reachable set is `2` or `3-4`.
- Sample at least four distinct targets for larger reachable sets.
- Balance sampling by base problem so large target sets do not dominate.
- Train only prior head, posterior head, and injection scale.
- Keep exact-target posterior scoring separate from oracle-validity scoring.

Required diagnostic:

```text
posterior K=1 exact selected-target rate
versus
prior K=1 exact selected-target rate
```

Decision:

- If the posterior cannot become strongly target-selective on this bounded
  micro-test, stop. The objective or conditioning path is defective.
- If it can, proceed once to Stage B.

### Stage B: matched multi-target Phase G-alpha rerun

Purpose: determine whether target-selective posterior modes amortize into a
prior that improves oracle coverage.

Design controls:

- Same keeper and keeper hash.
- Same frozen test rows.
- Same entropy-matched temperature comparator.
- Same iso-compute depth comparator.
- Same K values.
- Same K=1 preservation gate.
- Same frozen-block gradient assertions.
- Use the corrected repeated-input/multiple-target curriculum.
- Start with the selected `KL=0.001` EMA recipe.
- Use checkpoints at a small number of prospective doses, for example
  `250`, `500`, and `1,000`, rather than another unconstrained sweep.

Gate order:

1. Posterior exact-target selectivity.
2. K=1 deterministic preservation.
3. Prior oracle coverage versus temperature at matched K.
4. Prior oracle coverage versus iso-compute depth.

Interpretations:

- **Posterior selectivity fails:** objective or conditioning path failure.
- **Posterior succeeds, prior fails:** amortization failure; guided modes exist
  but the inference prior does not learn them.
- **Prior beats depth but not temperature:** width exists but remains
  uneconomical or redundant.
- **Prior beats both:** G-beta opens.

## 7. Questions for strategy review

1. Should the correction remain on the current forward branching-relations
   family for exact comparability, or return to non-injective abduction, whose
   inverse structure was the original Phase G gate?
2. Should alternate chains be materialized as frozen repeated rows, resampled
   online per base problem, or both? Frozen rows improve receipts; online
   resampling improves support coverage.
3. What exact posterior target-selectivity margin should be locked before the
   micro-test? The gate form is clear, but the numeric threshold should be
   power-checked prospectively.
4. Should prior training use all target chains uniformly, or weight base
   problems equally and targets conditionally within each problem?
5. Is the current entropy-matched answer sampler the final null comparator, or
   should a second temperature selected for maximal validation coverage be
   reported as a stronger practical baseline?
6. For paper framing, should Phase G-alpha be presented as a failed guidance
   result followed by a curriculum-identifiability correction, or held out of
   the deterministic paper and reserved entirely for the width paper?

## 8. Recommendation

Authorize one bounded multi-target correction, beginning with the posterior
target-selectivity micro-test. This is not a request for a new exploratory
program. It is the minimum experiment required to determine whether the first
Phase G-alpha negative belongs to the architecture or to a curriculum that
never exposed the conditional multimodality the architecture was meant to
learn.

If the posterior fails after explicit same-input/multiple-target supervision,
close the guided-width line on this substrate. If it succeeds, run one matched
coverage comparison. G-beta remains closed unless the corrected prior beats
both locked comparators.

## 9. Artifact map

Primary session summary:

```text
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/summary.json
```

Final selected-arm test:

```text
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/test/kl_0p001/ema/summary.json
```

Training-arm summaries:

```text
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/train/kl_0p0001/summary.json
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/train/kl_0p001/summary.json
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/train/kl_0p01/summary.json
```

Curriculum and posterior audit:

```text
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/autopsy/multimodal_supervision.json
outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/autopsy/multimodal_supervision.md
```

Audit implementation and tests:

```text
eval/analyze_phase_g_multimodal_supervision.py
tests/test_analyze_phase_g_multimodal_supervision.py
```

Drive backup:

```text
/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/
  stage5_phase_g_alpha_guided_width_20260717/autopsy/
```
