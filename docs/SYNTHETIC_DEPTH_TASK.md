# Synthetic Depth Task: Iterated Function Application

## Purpose

The deterministic ARC line showed that fixed deeper recurrence is harmful on ARC-style MCQ benchmarks. That result does not by itself prove that recurrence cannot supply sequential computation. It may instead mean ARC does not reward extra sequential depth.

This experiment tests the remaining mechanism claim directly: can the corrected recurrent architecture use additional loops to solve a task whose required sequential depth is known and tunable?

## Task

Each example samples a finite function `f: S -> S`, presents it as a shuffled lookup table, chooses a start value `x`, and asks for `f^d(x)`.

The generator enforces a distinct orbit prefix:

```text
x0 -> x1 -> x2 -> ... -> xd
```

So the target cannot be reached early through a short cycle. The remaining function-table rows are random distractors. This makes the required dependent lookup chain exactly depth `d` for the generated prefix.

## Prediction

Let `A(d, k)` be accuracy at task depth `d` using forced recurrent loop count `k`.

The success signature is a staircase:

- loop 1 solves shallow depths,
- higher forced loops solve deeper depths,
- the solved-depth frontier is non-decreasing and preferably strictly expanding with `k`,
- held-out functions show the same pattern.

The failure signature is the ARC shape:

- loop 1 is best or tied,
- deeper forced loops do not extend the solved-depth frontier,
- deeper loops mostly degrade.

## Scope

A positive result validates the mechanism in principle. It does not rescue ARC, and it does not prove natural benchmark value. It means the next question is which natural tasks reward this kind of sequential depth, and whether scale improves transfer.

A negative result is a clean refutation of the depth-substitution thesis under a task that removes the task-reward and base-competence confounds.

## Implementation

Core files:

- `training/synthetic_depth_task.py`
- `training/generate_synthetic_depth_task.py`
- `eval/eval_synthetic_depth_matrix.py`
- `colab/STAGE5_SYNTHETIC_DEPTH_TASK_CELL.py`

The generator writes both free-answer and MCQ-aligned SFT formats. The MCQ
option-text format is the preferred primitive diagnostic because it matches the
matrix evaluator's default scoring target:

- `train_sft.jsonl`: free-answer prompt, completion is the final value.
- `train_mcq_option_text_sft.jsonl`: MCQ prompt, completion is the correct option text.
- `train_mcq_label_sft.jsonl`: MCQ prompt, completion is the correct label.
- `train_mcq_label_and_text_sft.jsonl`: MCQ prompt, completion is `label. text`.

Launch target:

```python
os.environ["STAGE5_CURRENT_A100_TARGET"] = "synthetic_depth_task"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

The default target is a pilot-sized L4/T4 run:

- `N = 16` symbols
- depths `1..8`
- loops `1,2,4,8`
- `24` train/val/test rows per depth
- `25` training steps

Increase `STAGE5_SYNTH_DEPTH_ROWS_PER_DEPTH` and `STAGE5_SYNTH_DEPTH_MAX_STEPS` only after the pilot proves the plumbing and format are sane.

Primitive competence gate:

Before spending time on deeper staircase runs, use `max_depth=1`,
`max_loops=1`, `STAGE5_SYNTH_DEPTH_TRAIN_FORMAT=mcq_option_text`, and
`STAGE5_SYNTH_DEPTH_RUN_BASE_EVAL=1`. This checks whether the base model and
the recurrent model can learn the lookup primitive under the same MCQ format
used at evaluation. If depth-1 does not clear the threshold, deeper recurrence
claims are not yet interpretable.

## Two-Phase Discipline

The first staircase run changed symbol space, depth, and supervision pressure at
once. That made the result ambiguous: depth-1 accuracy collapsed before the
depth mechanism could be read. The corrected sequence is:

1. **Phase 1, primitive-generalization curve.** Keep `max_depth=1`,
   `max_loops=1`, and MCQ option-text SFT fixed. Vary only symbol count,
   usually `N=8,12,16`. The launch target is:

   ```python
   os.environ["STAGE5_CURRENT_A100_TARGET"] = "synthetic_depth_primitive_curve"
   exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
   ```

   The decision bar is `0.71`, because depth-4 four-option accuracy is floored
   when the primitive is below roughly `chance ** (1 / 4)`. Prefer `0.90+` for
   margin. The output recommends the largest `N` that clears the primitive bar.

2. **Phase 2, staged-depth forced-loop staircase.** Only after Phase 1 identifies
   a solid primitive `N`, add depths through a staged curriculum. Keep forced
   loop evaluation and keep router supervision off. This tests the mechanism,
   not the router.

3. **Phase 3, depth router.** Only if Phase 2 shows a staircase, turn on
   `halt_target_nll_weight` and a small nonzero `beta` to test whether the model
   can allocate loops without forcing.

## Decision Rule

Proceed only if:

1. depth-1 examples are learnable,
2. loop-1 accuracy falls at larger depths,
3. higher forced loops recover accuracy at those larger depths,
4. the frontier-by-loop strictly expands on held-out functions.

Otherwise close the deterministic recurrence-depth thesis rather than continuing to search benchmark slices.
