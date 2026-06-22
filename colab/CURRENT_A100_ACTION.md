# Current GPU Action

## Preferred Launch Path

Use the safe-continue cell from a normal Drive-backed or blank Colab notebook:

[`colab/STAGE5_SAFE_CONTINUE_CELL.md`](STAGE5_SAFE_CONTINUE_CELL.md)

Keep the runtime disconnected while editing the cell. Attach an L4/T4 if
available; use A100 only if that is the practical available runtime. This is a
diagnostic eval, not a training run.

Set:

```python
RUN_A100_ACTION = True
```

only when you intentionally want the guarded action to execute. The cell pulls
latest GitHub, authenticates GitHub/Hugging Face, mounts Drive when needed,
runs the go/no-go guard, executes one allowlisted action, backs up/commits safe
artifacts, and disconnects by default.

## Source Summary

The current source of truth is the failed ARC-mix recovery proxy:

```text
outputs/stage5/stage5_arc_mix_recovery_once_20260622_030628/summary.json
```

Key result:

```text
status = no_proxy_lift
decision = stop_and_revise_objective
best proxy = 66/128
base proxy = 68/128
start proxy = 68/128
margin delta vs base = -0.308232
```

This means: do **not** extend ARC-mix training and do **not** run GPQA,
Phase 2/SVGD, or scale-up. The next useful paid-GPU action is a small routing
diagnostic.

## Experiment

Run exactly one bounded depth/width routing diagnostic:

```bash
python colab/run_stage5_routing_diagnostic.py
```

Default limits:

```text
STAGE5_ROUTING_ARC_EASY_LIMIT=64
STAGE5_ROUTING_ARC_CHALLENGE_LIMIT=64
STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS=1
```

The runner restores the recovered deterministic Phase 1 checkpoint from Drive
if needed, delegates to `colab/run_stage5_benchmark_suite.py`, then writes:

```text
outputs/stage5/<run_id>/benchmark_summary.json
outputs/stage5/<run_id>/routing_assessment.json
outputs/stage5/<run_id>/routing_assessment.md
```

## How To Interpret The Result

The routing assessment has a machine-readable `status`:

| Status | Meaning | Next action |
|---|---|---|
| `needs_direct_halting_repair` | Direct/base-confident rows drift or over-loop. | Train Phase 1 direct-mode recovery with base-logit distillation and shallow halt supervision. |
| `needs_deep_narrow_recovery` | Direct rows look acceptable but deep/numeric rows do not improve. | Train Phase 1 deep-narrow recovery with repulsion off and non-collapsed halt-depth targets. |
| `routing_diagnostic_pass` | Small diagnostic found no direct/deep blocker. | Consider a larger confirmation or the bounded direct/deep recovery ladder. |

This diagnostic is the bridge from the failed generic ARC-mix proxy to the new
depth/width curriculum. It tells us which part of the recurrent model to train
next instead of spending A100 on another undifferentiated continuation.

After a diagnostic summary lands, the same safe-continue path will select the
next action:

```bash
python colab/run_stage5_routing_repair.py
```

only for `needs_direct_halting_repair` or `needs_deep_narrow_recovery`. The
repair runner maps the diagnosis to one bounded deterministic Phase 1 profile
and keeps particles/SVGD off. That repair now writes typed ARC rows with
explicit loop-depth supervision: ARC-Easy/direct rows target loop `1`, while
ARC-Challenge rows target loop `2` for direct-halting probes or loop `3` for
deep-narrow recovery.
