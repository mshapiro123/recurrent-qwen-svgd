# Phase G Corrected-Width Handoff

**Date:** 2026-07-18

**Scope:** close the first GRAM-inspired guided-width attempt accurately, record
the curriculum confound, and define the one bounded corrected test.

**Primary receipts:**

- Original final run: `38cad711`.
- Original audit implementation: `47634046`.
- Original result/autopsy handoff: `3601fdf3` and
  `docs/STAGE5_PHASE_G_ALPHA_RESULT_AND_CURRICULUM_AUTOPSY_HANDOFF_20260718.md`.
- Corrected repeated-prompt curriculum preparation: `0112a636`.

## Executive State

The original G-alpha experiment established that sampling several latent
trajectories is substantially better than spending the same nominal budget on
one excessively deep deterministic trajectory. It did **not** establish that
the learned latent prior improves on ordinary answer-head sampling. The latter
comparison was negative.

That negative does not answer the intended GRAM-style question because the
training data lacked the minimum condition for conditional multimodality: no
prompt appeared with more than one target. The posterior was therefore never
asked to distinguish different valid solutions for the same input.

One corrected repeated-prompt curriculum is now implemented and validated on
CPU. It is the only justified follow-up before either advancing to selection
and particle variants or closing the GRAM-transplant question.

## What the Completed G-alpha Run Measured

### Frozen setup

- Base architecture: recurrent Qwen2.5-0.5B-Instruct with the repaired
  deterministic loop mechanism.
- Frozen keeper: the recurrent block and installed deterministic mechanism.
- Trainable elements only: latent prior head, target-conditioned posterior head,
  and injection scale.
- Training arms: KL coefficients `0.0001`, `0.001`, and `0.01`; 1,000 steps;
  K=4 trajectories; latent dimension 64; KL balance 0.8; EMA decay 0.999.
- Selected model: KL `0.001`, EMA weights, selected only among calibration K=1
  parity arms.
- Test: exact oracle coverage on 512 frozen branching-relation rows at
  `K=1,2,4,8,20`.

### Locked K=20 results

| Comparison | Mean exact coverage | Reading |
|---|---:|---|
| Latent prior | `0.3049` | Reference arm |
| Entropy-matched answer sampling | `0.3450` | Latent prior loses by `-0.0401` |
| Iso-compute deterministic depth | `0.0543` | Latent prior wins by `+0.2506` |
| Posterior teacher | `0.3063` | Only `+0.0014` over prior |

Paired latent-prior versus iso-depth coverage had 337 helped, 20 hurt, and
155 tied rows (`p=9.78e-76`, one-sided sign test). This is strong evidence for
width relative to extreme iterative depth. Latent prior versus answer sampling
had 24 helped, 63 hurt, and 425 tied rows (`p=0.99999` in the preregistered
one-sided direction), so G-alpha did not clear its joint gate.

Other important readings:

- K=20 duplicate rates were high for both methods: latent `0.9386`,
  temperature `0.9326`.
- The pooled K=1 preservation delta was `0.0117`, within the pooled tolerance,
  but one depth-4 reachable-set cell exceeded its per-cell tolerance.
- The posterior teacher was not more target-faithful than the prior. This was
  the first warning that the posterior condition was not carrying usable
  information.

## Curriculum Autopsy

The audit of the completed data and trace cache is decisive:

| Check | Observed |
|---|---:|
| Training rows | 2,048 |
| Unique prompt/problem groups | 2,048 |
| Repeated prompts | 0 |
| Prompt groups with multiple targets | 0 |
| Targets in one K=4 optimizer step | One fixed chain, repeated for all trajectories |

The original generator constructed a fresh table and prompt for every row, then
stored one locally sampled valid chain. The target-conditioned posterior could
in principle see that row's selected target, but it never saw two valid targets
for a shared input. Its only learnable behavior was an ordinary one-target
mapping, not conditional solution selection.

Consequently, the defensible conclusion is narrow:

> The first implementation generated latent width and outperformed very deep
> deterministic recurrence, but its learned guidance curriculum was
> non-identifying for conditional multimodality and did not outperform ordinary
> answer sampling.

Do not claim that GRAM-style guidance fails on the recurrent-Qwen substrate.
Do not use this run as evidence for or against SVGD.

## Corrected Multi-Target Assets

Commit `0112a636` adds a CPU-validated correction.

### Data construction

`training/phase_g_multitarget_task.py` expands each one branching-relation
problem into one row per exact reachable terminal. Variants share the complete
prompt problem while each contains a distinct terminal target and one valid
exact-depth intermediate chain.

Required row metadata:

```text
base_problem_id
target_variant_index
target_variant_count
sampled_chain
loop_completions
posterior_chain_sampling=enumerated_distinct_terminal_target
```

The validator proves that prompt groups are identical except for the selected
valid chain, targets are distinct, all stored chains are legal, and full target
support is present. The default preparation contract requires every reachable
terminal rather than an arbitrary subset.

### Sampling correction

`training/phase_g_sampling.py` introduces:

```text
base problem uniformly -> target variant uniformly
```

This prevents a prompt with many valid terminals from receiving more updates
than a prompt with few terminals. `row_uniform` remains the default for legacy
run compatibility. Only the non-default policy is written into the resume
contract, so historical G-alpha checkpoints remain resumable without a false
contract mismatch.

### Held-out posterior control

`colab/run_stage5_phase_g_multitarget_prepare.py` writes separate training and
posterior-control splits. The latter is a repeated-prompt held-out set and is
used before coverage analysis to establish whether the posterior changes its
K=1 prediction when only the selected valid target changes.

The enhanced `eval/analyze_phase_g_multimodal_supervision.py` reports:

- number of actual multi-target prompt groups;
- mean distinct first predictions within each group;
- rate at which every selected-target variant is matched;
- prior versus posterior-teacher values on the same rows.

The unit test constructs a prompt with targets A and B and verifies that a
teacher selecting A/B scores two distinct predictions and full group match,
while an invariant prior selecting A scores one prediction and fails the group
match.

### Verification completed

CPU tests passed:

```text
12 passed in 0.84s
```

The scaled preparation smoke also passed. For 64 base prompt groups it created
226 training variants; for 32 held-out prompt groups it created 106 variants.
Every group had at least two targets and full reachable-target support.

The detailed implementation contract is in:
`docs/STAGE5_PHASE_G_MULTITARGET_CORRECTION_SPEC_20260718.md`.

## The Next GPU Experiment

### Stage G-A0: posterior-control micro-test

This is not yet a coverage claim. It must train the same frozen keeper and
trainable set as original G-alpha, but use:

```text
repeated-prompt multi-target train data
--sampling_policy base_problem_uniform
```

It must evaluate both the posterior teacher and the prior on held-out
repeated-prompt control rows at K=1. Before launch, lock powered effect-size
thresholds for teacher selected-target fidelity, teacher-minus-prior fidelity,
and target-group switching, plus a one-sided paired sign-test threshold for
teacher-minus-prior selected-target lift. None can be selected after viewing
the run.

The independent unit for both the effect-size comparison and the paired sign
test is the base prompt, not the target-row variant: variants sharing a table
and prompt are correlated by construction. The scorer therefore averages
selected-target fidelity within each prompt group before comparing posterior
and prior.

Mandatory launch assertions:

1. Exact keeper checkpoint SHA-256 matches the frozen original keeper.
2. Only prior head, posterior head, and injection scale have gradients.
3. Block gradients are identically zero.
4. Curriculum validator reports every base prompt has multiple target variants
   and full support.
5. Sampling policy is `base_problem_uniform`.
6. Train and posterior-control base-prompt manifests differ.
7. Existing frozen coverage/calibration rows are present with their expected
   manifests before any final coverage calculation.
8. A committed posterior-control gate-lock receipt matches the regenerated
   repeated-prompt control-row manifest exactly; raw environment thresholds are
   not accepted.

### Gate A0

**Pass:** the posterior teacher reliably follows distinct selected targets on
the repeated-prompt holdout and materially exceeds the prior under the
pre-locked paired criterion.

**Fail:** the posterior is still target-invariant or cannot transfer any
conditional information even on properly identified data. Close the
GRAM-guidance transplantation line at this substrate and report the boundary;
do not add SVGD, a selector, or another optimizer sweep.

### Stage G-A1: one corrected coverage run

Only if A0 passes, run one matched Phase G-alpha coverage test on the unchanged
original frozen coverage rows. The locked comparisons are unchanged:

1. Latent K versus entropy-matched output temperature K.
2. Latent width K x depth T versus one deterministic trajectory at depth K x T.

Gate order remains:

```text
posterior control -> K=1 preservation -> latent versus temperature
-> latent versus iso-depth
```

G-beta opens only if the corrected prior clears both coverage comparisons.

## Questions For Strategy Review

1. What posterior-control success margin is powered and appropriate for the
   planned held-out prompt count? Lock it before the A0 launch.
2. Should A0 retain the original KL sweep (`0.0001`, `0.001`, `0.01`) or use a
   short dose-finding phase with a single fixed KL and a pre-registered rule to
   select the one coverage arm? The scope must remain one bounded correction,
   not a new sweep program.
3. Is the original forward branching task sufficiently diagnostic once it has
   repeated targets, or should the corrected study instead move directly to the
   non-injective abduction family? The latter is more semantically aligned with
   multimodality, but changes both the deterministic substrate requirement and
   the continuity of the existing coverage result.
4. Does the one depth-4 K=1 preservation-cell failure require a tightened
   injection-scale constraint in A0, or should it remain reported as a
   secondary preservation diagnostic until the multi-target posterior is shown
   to function at all?

## Explicitly Closed Or Deferred

- Original G-alpha is complete and its original joint gate is failed.
- Original G-alpha does not justify a claim that learned guidance beats output
  sampling.
- G-beta selection, per-trajectory halting, and SVGD are deferred.
- The deterministic mechanism remains frozen during A0 and A1; no substrate
  adaptation is permitted because that would destroy attribution.
