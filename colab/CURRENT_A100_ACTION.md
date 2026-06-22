# Current GPU Action

## Preferred Launch Path

Use the safe-continue cell from a normal Drive-backed or blank Colab notebook:

[`colab/STAGE5_SAFE_CONTINUE_CELL.md`](STAGE5_SAFE_CONTINUE_CELL.md)
or the directly fetchable plain cell
[`colab/STAGE5_SAFE_CONTINUE_CELL.py`](STAGE5_SAFE_CONTINUE_CELL.py).

If the runtime has reset or Drive authorization is stale, first run the cheap
preflight cell on CPU or a low-cost runtime:

[`colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md`](STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md)
or
[`colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py`](STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py).

That cell mounts Drive, verifies the recovered deterministic Phase 1 checkpoint
is visible, runs the A100 go/no-go guard, and disconnects. Only attach an
A100/H100 after `checkpoint_preflight.available` is `True`.

Keep the runtime disconnected while editing the cell. This is one bounded
deterministic Phase 1 repair run; use an A100/H100 only when you intentionally
want to spend paid GPU on it. If cheaper L4/T4 capacity is immediately
available, it is acceptable, but expect a slower run.

Set:

```python
RUN_A100_ACTION = True
```

only when you intentionally want the guarded action to execute. The cell pulls
latest GitHub, authenticates GitHub/Hugging Face, mounts Drive when needed,
runs the go/no-go guard, executes one allowlisted action, backs up/commits safe
artifacts, and disconnects by default.

If the go/no-go output is `routing_checkpoint_missing_no_go`, do not keep the
GPU session alive. Disconnect and run the Drive/checkpoint preflight first.

## Source Summary

The current source of truth is the completed routing diagnostic:

```text
outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json
```

Key result:

```text
status = needs_direct_halting_repair
next_action = Train Phase 1 direct-mode recovery with base-logit distillation and shallow halt supervision.
ARC-Easy direct delta = -2, mean direct loops = 2.58, mean direct margin delta = -2.49
ARC-Challenge direct delta = -3, mean direct loops = 2.62, mean direct margin delta = -2.02
ARC-Challenge conceptual delta = +2
```

This means: do **not** run GPQA, Phase 2/SVGD, wide-particle training, or
scale-up yet. The model is still harming base-confident direct rows and
over-looping on them. The next useful paid-GPU action is a direct-mode
deterministic repair.

## Experiment

Run exactly one bounded direct-mode halting repair:

```bash
python colab/run_stage5_routing_repair.py
```

The selected profile for `needs_direct_halting_repair`:

```text
repair_mode=direct_halting
STAGE5_ARC_MIX_ARC_EASY_REPEAT=8
STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT=1
STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP=1
STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP=2
STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE=direct
STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE=deep_narrow_probe
STAGE5_ARC_MIX_EVAL_CONFIG=ARC-Easy
STAGE5_ARC_MIX_ARMS=arc_mix_response_w02_lr2e6
```

The proxy eval is ARC-Easy for this direct-halting repair. That is deliberate:
the source diagnostic showed the model over-looping and regressing on
base-confident direct rows, so the bounded repair must clear the direct/Easy
proxy before a larger ARC-Easy/ARC-Challenge confirmation benchmark.
The A100 go/no-go summary should show
`routing_repair_profile.expected_arc_eval_config = "ARC-Easy"` before you
allow the paid action to run.

The runner restores the recovered deterministic Phase 1 checkpoint from Drive
if needed, delegates to `colab/run_stage5_balanced_arc_mix_gate.py`, keeps
particles/SVGD off, and writes:

```text
outputs/stage5/<run_id>/repair_run/summary.json
outputs/stage5/<run_id>/summary.json
outputs/stage5/<run_id>/summary.md
```

## How To Interpret The Result

The repair summary wraps the child ARC-mix gate status:

| Status | Meaning | Next action |
|---|---|---|
| `repair_proxy_lift` | Direct repair lifted proxy accuracy and passed calibration. | Run the full balanced ARC confirmation. |
| `repair_proxy_matches_base` | Direct repair restored proxy to base without calibration warning. | Run the full balanced ARC confirmation. |
| `repair_proxy_lift_calibration_warning` | Accuracy lifted but base calibration degraded. | Stop and tighten preservation/distillation. |
| `repair_proxy_matches_base_calibration_warning` | Accuracy matched base but calibration degraded. | Stop and tighten preservation/distillation. |
| `repair_no_proxy_lift` | Repair did not improve the proxy. | Stop and revise direct-loop supervision. |

Do not proceed to width/particles until direct rows stop regressing. This is
the calibration floor for the wider depth/width curriculum.

## Optional Constructed-Curriculum Lever

If the routing repair still reports direct/deep calibration problems, the repo
also contains a bounded constructed-curriculum runner:

```bash
STAGE5_PROGRAMMATIC_SOURCE_SUMMARY=outputs/stage5/<source_run>/summary.json \
python colab/run_stage5_programmatic_depth_repair.py
```

That runner generates verified direct/deep-narrow arithmetic-chain rows on CPU,
exports only `positive_direct` and `positive_depth` traces, performs one short
Phase 1 continuation with base-logit distillation, and evaluates on a held-out
constructed split. Treat it as a calibration ingredient only. Any checkpoint it
produces still needs the ARC routing/benchmark gate before particles, SVGD,
or wider data should resume.
