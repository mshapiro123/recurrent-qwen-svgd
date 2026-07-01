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

## Decision Rule

Proceed only if:

1. depth-1 examples are learnable,
2. loop-1 accuracy falls at larger depths,
3. higher forced loops recover accuracy at those larger depths,
4. the frontier-by-loop strictly expands on held-out functions.

Otherwise close the deterministic recurrence-depth thesis rather than continuing to search benchmark slices.
