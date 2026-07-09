"""Evaluate saved natural-surface transfer checkpoints without retraining.

This is the resume path for an interrupted/early-stopped natural-surface
transfer run.  It expects the training checkpoints to still exist locally in
the Colab runtime under the source run directory, evaluates selected steps on
relay, pointer, and symbolic-rehearsal readouts, and publishes only lightweight
evidence artifacts to GitHub.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import (  # noqa: E402
    DEFAULT_DATA_SUMMARY,
    active_diag_min,
    data_paths,
    eval_active_checkpoint,
    path_for_cli,
    publish_run,
    read_json,
    root_path,
    score_experiment,
    write_json,
)


DEFAULT_SOURCE_RUN_ID = "stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812"


def parse_steps(raw: str) -> list[int]:
    steps: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError(f"Checkpoint steps must be positive, got {step}")
        steps.append(step)
    if not steps:
        raise ValueError("No checkpoint steps requested")
    return sorted(dict.fromkeys(steps))


def checkpoint_path_for_step(source_run_dir: Path, step: int) -> Path:
    checkpoint = source_run_dir / "train" / "verbal_rung_zero" / f"unfrozen_recurrent_step_{step}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Missing natural-transfer checkpoint. This eval-only resume must run in the "
            f"same Colab runtime or after restoring checkpoints. Missing: {checkpoint}"
        )
    return checkpoint


def eval_step(
    output_dir: Path,
    *,
    source_run_dir: Path,
    paths: dict[str, Path],
    step: int,
    dtype: str,
    train_depth_max: int,
    eval_depth_max: int,
) -> dict[str, Any]:
    checkpoint = checkpoint_path_for_step(source_run_dir, step)
    step_output_dir = output_dir / f"step_{step}"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    loop_counts_eval = ",".join(str(idx) for idx in range(1, eval_depth_max + 1))
    loop_counts_train = ",".join(str(idx) for idx in range(1, train_depth_max + 1))
    return {
        "step": step,
        "checkpoint": path_for_cli(checkpoint),
        "relay": eval_active_checkpoint(
            step_output_dir,
            name=f"step{step}_relay",
            checkpoint=checkpoint,
            data_jsonl=paths["relay_test_chain_mcq"],
            loop_counts=loop_counts_eval,
            value_prefix="name:",
            dtype=dtype,
        ),
        "pointer": eval_active_checkpoint(
            step_output_dir,
            name=f"step{step}_pointer",
            checkpoint=checkpoint,
            data_jsonl=paths["pointer_test_chain_mcq"],
            loop_counts=loop_counts_eval,
            value_prefix="name:",
            dtype=dtype,
        ),
        "synthetic_rehearsal": eval_active_checkpoint(
            step_output_dir,
            name=f"step{step}_synthetic_rehearsal",
            checkpoint=checkpoint,
            data_jsonl=paths["synthetic_rehearsal_chain_symbol_sft"],
            loop_counts=loop_counts_train,
            value_prefix="letter:",
            dtype=dtype,
        ),
    }


def step_decision(
    *,
    frozen_baseline: dict[str, Any],
    step_eval: dict[str, Any],
    train_depth_max: int,
    eval_depth_max: int,
) -> dict[str, Any]:
    post = {
        "relay": step_eval["relay"],
        "pointer": step_eval["pointer"],
        "synthetic_rehearsal": step_eval["synthetic_rehearsal"],
    }
    scored = score_experiment(
        frozen=frozen_baseline,
        post=post,
        train_depth_max=train_depth_max,
        eval_depth_max=eval_depth_max,
    )
    experiment_1 = dict(scored["experiment_1"] or {})
    experiment_1["step"] = step_eval["step"]
    return experiment_1


def write_summary_markdown(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Natural-Surface Checkpoint Curve - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Steps: `{payload['steps']}`",
        "",
        "## Frozen Baseline",
        "",
        f"- Relay: `{payload['frozen_baseline']['n24']['relay']['active_diagonal']}`",
        f"- Pointer: `{payload['frozen_baseline']['n24']['pointer']['active_diagonal']}`",
        f"- Synthetic rehearsal: `{payload['frozen_baseline']['n24']['synthetic_rehearsal']['active_diagonal']}`",
        "",
        "## Checkpoint Curve",
        "",
    ]
    for row in payload.get("checkpoint_evals", []):
        decision = row.get("decision_read", {})
        lines.extend(
            [
                f"### Step {row['step']}",
                "",
                f"- Relay diagonal: `{row['relay']['active_diagonal']}`",
                f"- Pointer diagonal: `{row['pointer']['active_diagonal']}`",
                f"- Synthetic rehearsal diagonal: `{row['synthetic_rehearsal']['active_diagonal']}`",
                f"- Decision read: `{decision}`",
                "",
            ]
        )
    if payload.get("best_by_metric"):
        lines.extend(["## Best By Metric", "", f"`{payload['best_by_metric']}`", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def best_by_metric(checkpoint_evals: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "relay_train_depth_min",
        "relay_extrap_depth_min",
        "pointer_train_depth_min",
        "pointer_extrap_depth_min",
        "synthetic_rehearsal_min",
        "synthetic_rehearsal_min_delta",
    ]
    out: dict[str, Any] = {}
    for metric in metrics:
        candidates = [
            (float(row["decision_read"].get(metric, float("-inf"))), int(row["step"]))
            for row in checkpoint_evals
            if row.get("decision_read") and row["decision_read"].get(metric) is not None
        ]
        if candidates:
            value, step = max(candidates)
            out[metric] = {"step": step, "value": value}
    return out


def main() -> int:
    source_run_id = os.environ.get("STAGE5_NATURAL_TRANSFER_EVAL_SOURCE_RUN_ID", DEFAULT_SOURCE_RUN_ID)
    source_run_dir = ROOT / "outputs" / "stage5" / source_run_id
    source_summary_path = source_run_dir / "summary.json"
    if not source_summary_path.exists():
        raise FileNotFoundError(f"Missing source natural-transfer summary: {source_summary_path}")

    source_summary = read_json(source_summary_path)
    data_summary = os.environ.get("STAGE5_NATURAL_TRANSFER_DATA_SUMMARY") or source_summary.get(
        "data_summary", DEFAULT_DATA_SUMMARY
    )
    steps = parse_steps(os.environ.get("STAGE5_NATURAL_TRANSFER_EVAL_STEPS", "2000,4000,6000"))
    dtype = os.environ.get("STAGE5_NATURAL_TRANSFER_DTYPE", source_summary.get("dtype", "bfloat16"))
    train_depth_max = int(
        os.environ.get("STAGE5_NATURAL_TRANSFER_TRAIN_MAX_DEPTH", source_summary.get("train_depth_max", 8))
    )
    eval_depth_max = int(
        os.environ.get("STAGE5_NATURAL_TRANSFER_EVAL_MAX_DEPTH", source_summary.get("eval_depth_max", 12))
    )

    output_id = os.environ.get("STAGE5_NATURAL_TRANSFER_EVAL_OUTPUT_ID") or time.strftime(
        f"{source_run_id}_checkpoint_curve_%Y%m%d_%H%M%S"
    )
    output_dir = ROOT / "outputs" / "stage5" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = data_paths(data_summary)
    frozen_baseline = source_summary.get("frozen_baseline")
    if not frozen_baseline or "n24" not in frozen_baseline:
        raise KeyError(f"Source summary lacks frozen_baseline.n24: {source_summary_path}")

    checkpoint_evals: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "kind": "stage5_natural_surface_checkpoint_curve",
        "run_id": output_id,
        "status": "started",
        "source_run_id": source_run_id,
        "source_summary": path_for_cli(source_summary_path),
        "data_summary": data_summary,
        "steps": steps,
        "dtype": dtype,
        "train_depth_max": train_depth_max,
        "eval_depth_max": eval_depth_max,
        "frozen_baseline": frozen_baseline,
        "checkpoint_evals": checkpoint_evals,
        "best_by_metric": {},
    }
    write_json(output_dir / "summary.json", payload)

    for step in steps:
        print(f"\n===== NATURAL SURFACE CHECKPOINT EVAL step={step} =====", flush=True)
        row = eval_step(
            output_dir,
            source_run_dir=source_run_dir,
            paths=paths,
            step=step,
            dtype=dtype,
            train_depth_max=train_depth_max,
            eval_depth_max=eval_depth_max,
        )
        row["decision_read"] = step_decision(
            frozen_baseline=frozen_baseline,
            step_eval=row,
            train_depth_max=train_depth_max,
            eval_depth_max=eval_depth_max,
        )
        checkpoint_evals.append(row)
        payload["status"] = f"evaluated_step_{step}"
        payload["checkpoint_evals"] = checkpoint_evals
        payload["best_by_metric"] = best_by_metric(checkpoint_evals)
        write_json(output_dir / "summary.json", payload)
        write_summary_markdown(output_dir, payload)
        publish_run(
            output_dir,
            message=f"Record natural-surface checkpoint curve {output_id} step {step} [skip ci]",
            update_pointer=False,
        )

    payload["status"] = "finished"
    payload["checkpoint_evals"] = checkpoint_evals
    payload["best_by_metric"] = best_by_metric(checkpoint_evals)
    write_json(output_dir / "summary.json", payload)
    write_summary_markdown(output_dir, payload)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    publish_run(
        output_dir,
        message=f"Record natural-surface checkpoint curve {output_id} final [skip ci]",
        update_pointer=False,
    )
    print(json.dumps({"run_id": output_id, "summary": path_for_cli(output_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
