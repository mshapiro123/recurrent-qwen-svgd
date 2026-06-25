# Deep Research Handoff - Master Sequence Reset, June 25 2026

## Purpose

This handoff is the short strategy packet for the current recurrent-Qwen
program state. It supersedes the older "current action" instincts that pointed
back to ARC-mix, direct-preservation, particle-noise, or kernel-geometry
sweeps. Those results remain useful evidence, but they are not the front of the
dependency chain.

The current program order is:

```text
Phase 0: loop-closure re-entry repair
Phase 1: deterministic depth recovery and dense control
Phase 2: breadth and multistability diagnostics
Phase 3: particles/SVGD and selector conversion
```

The active blocker is Phase 0.

## Central Thesis

The program is testing a narrow architectural claim:

> A pretrained dense Qwen model can be converted into a recurrent latent-depth
> model, preserving base behavior at depth 1 while using learned recurrent depth
> to recover or exceed larger-model reasoning on problems where additional
> sequential composition matters.

This is not a claim that recurrence replaces all benefits of scale. The
intended substitution is for sequential composition depth, not stored
knowledge, feature width, or parallel circuit capacity.

Particles/SVGD are downstream. They should only return after deterministic
recurrent depth is stable and produces correct-bearing alternatives.

## Current Evidence

### Strong positive evidence

- The one-pass identity wrapper is solved under strict settings. The recurrent
  split can exactly reproduce base Qwen when `max_loops=1`.
- Deterministic recurrent training can be numerically stable when trainable
  modules stay fp32 and the frozen base remains low precision.
- Depth signals have appeared on ARC-Challenge-style harder rows. Earlier
  fixed-depth and selected-depth runs showed that deeper loops sometimes
  contain correct answers that loop 1 misses.
- Debiased scoring exposed that some apparent regressions and gains were MCQ
  label/position artifacts. The project now tracks content and cyclic scoring
  side by side.

### Negative or cautionary evidence

- Inference-time particle noise and SVGD on the current checkpoint create
  superficial diversity more reliably than correct-bearing diversity.
- The current recovered checkpoint's loop dynamics look expansive, not
  multistable. Effective pathway spread before re-entry repair is not clean
  evidence of breadth.
- Stage 1 re-entry drift found the bridge effectively dead:
  `bridge_gate=0.0`, zero bridge delta, and zero bridge projection/bias/gate
  gradients.
- Easy-item preservation remains the main benchmark blocker. The architecture
  has shown hard-slice signals, but not a clean release-grade win over base.

## Why Re-entry Is First

The recurrent block was pretrained to receive the distribution produced by the
Prelude. After the first recurrent loop, the model feeds the recurrent block
its own output. If that re-entry state is off the expected input manifold, then:

- depth training acts on a distorted loop input;
- halting learns around a broken transition;
- breadth measurements reflect expansion/noise rather than basins;
- particles/SVGD push apart states the base recurrent map cannot use.

Therefore loop closure is the root dependency. It must be repaired before
depth, breadth, or particles can be interpreted.

## Current Concrete State

Latest reviewer state:

```text
latest stage: stage2_norm
latest status: entry_rms_safe_for_smoke
current source summary: outputs/stage5/stage5_reentry_norm_20260625_013527/summary.json
next target: reentry_repair_smoke
```

Stage 2 compared no re-entry normalization against eval-only `entry_rms`
normalization and did not find a major candidate-conversion regression. That
clears only a tiny trainable repair smoke, not full recovery training.

## Next GPU Sequence

Use the maintained GitHub-fetched launcher and change only
`STAGE5_CURRENT_A100_TARGET`.

### 1. Stage 3: `reentry_repair_smoke`

Purpose:

- reset bridge to an identity-preserving but gradient-live path;
- train only `bridge,reentry,halt`;
- enable `entry_rms` loop re-entry normalization;
- enable the identity-initialized re-entry adapter;
- verify loop-1 preservation.

Runtime:

- L4/T4 is enough.

Pass condition:

```text
review_stage5_reentry.py --no_write
recommendation = run_bounded_recovery_training_with_reentry_repair
```

Stop or rerun only bounded Stage 3 if:

- bridge gradients are still dead;
- re-entry adapter gradients are dead;
- bridge or adapter is live but unmoved;
- loop-1 preservation evidence is missing;
- loop-1 behavior regresses.

### 2. Stage 4: `reentry_recovery_training`

Purpose:

- bounded deterministic recovery SFT from the Stage 3 repaired checkpoint;
- particles off;
- learned loop control on;
- target-loop NLL supervision on;
- `entry_rms` re-entry normalization and re-entry adapter carried forward.

Pass condition:

- finite train and validation metrics;
- target-loop/depth gradient present;
- direct/easy behavior does not collapse;
- wrapper summary publishes as `kind=stage5_reentry_recovery_training`.

### 3. Benchmark: `debiased_benchmark_suite`

Purpose:

- compare base Qwen 0.5B against repaired recurrent Qwen;
- use ARC-Easy, ARC-Challenge, and GPQA-lite;
- read cyclic debiased scores and content scores together;
- learned loop control should be enabled, because Stage 4 trains it.

After this publishes, run `master_sequence_status`. Its `Phase 1 Gate Review`
section decides whether the recurrent-vs-base benchmark is sufficient to spend
on the dense same-curriculum control or whether deterministic recovery still
needs repair.

### 4. Control: `dense_mcq_trace_sft_control`

Purpose:

- train/evaluate standard dense Qwen LoRA on the same curriculum;
- separate architecture lift from data-recipe lift.

The recurrent architecture earns the Phase 1 claim only if it beats this
same-curriculum dense control on hard/depth-shaped rows without easy regression.

After the dense control publishes, run `master_sequence_status` again. The
`Phase 1 Gate Review` section is the handoff to strategy review: only
`hard_tail_lift_vs_dense` is an architecture signal; matching or losing to the
dense control means the data recipe helped but the recurrence claim is not yet
proven.

## Strategic Questions For Deep Research

1. After loop closure is gradient-live, what is the right depth-label target?
   The current high-level proxy is:
   - base-solvable rows -> depth 1;
   - Qwen-1.5B-only verified rows -> depth 2;
   - Qwen-3B/7B-only verified rows -> depth 3/4.
   Is there a better target derived from trace structure, verifier difficulty,
   or rescue-by-chain-of-thought?

2. If Stage 4 recovers easy behavior but hard-depth lift remains weak, is the
   limiting factor:
   - 0.5B capacity;
   - insufficient depth labels;
   - missing conditioning in the recurrent block;
   - or the bridge/re-entry adapter still being too low-rank/simple?

3. What benchmark split best tests "recurrence substitutes for scale" without
   confounding stored knowledge?
   Preferred candidates are verified ARC-style reasoning, synthetic
   compositional problems, math with controlled operation counts, and
   capability-ladder rows where larger Qwen models solve examples smaller Qwen
   models miss.

4. What is the cleanest same-recipe dense control?
   The control must see the same positive rows and similar LoRA budget, but not
   receive recurrent depth or re-entry modules.

5. If deterministic depth clears Phase 1, what should count as
   correct-bearing breadth in Phase 2?
   The key metric should not be raw diversity. It should be effective pathway
   count split by correctness and selector-convertible candidates.

6. If deterministic depth fails after re-entry repair, what scale probe should
   be run first?
   The current preference is a no-training 1.5B identity/loop-preservation
   probe as information, not an immediate pivot to large-model training.

## What Not To Do Yet

- Do not run more SVGD/kernel-geometry sweeps before deterministic recurrence
  is repaired and benchmarked.
- Do not run GPQA Diamond or public-release benchmarks before the repaired
  recurrent checkpoint is base-competitive on the bounded debiased suite.
- Do not treat content-score improvement as sufficient if cyclic-debiased
  scoring regresses.
- Do not claim architecture lift until the dense same-curriculum control is
  measured.

## Minimal Decision Tree

```text
Stage 3 fails:
  fix bridge/re-entry liveness; rerun bounded repair smoke only.

Stage 3 passes, Stage 4 fails:
  inspect recovery training, target-loop supervision, and easy/direct collapse.

Stage 4 passes, recurrent loses to base:
  improve deterministic depth recovery or test 1.5B capacity; no particles yet.

Stage 4 passes, recurrent beats base but dense control matches/beats it:
  data recipe works; architecture contribution not proven.

Stage 4 passes, recurrent beats base and dense control on hard rows:
  proceed to Phase 2 breadth diagnostics.

Phase 2 shows correct-bearing breadth:
  return to particles/SVGD and selector conversion.

Phase 2 shows diversity without correctness:
  train pathway supervision or regime shaping; do not tune inference noise.
```
