"""Train a depth-1..6 support route and score it on a frozen depth-1..10 set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import (
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    write_json,
)


NONREGRESSION_FLOORS = {
    "1": 0.93,
    "2": 0.93,
    "3": 0.93,
    "4": 0.93,
    "5": 0.85,
    "6": 0.85,
}
SELECTION_MIN_CORRECT = {
    "7": 52,
    "8": 19,
    "9": 14,
    "10": 14,
}


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def diagonal_counts(active_summary: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    matrix = active_summary.get("active_matrix", {})
    out: dict[str, dict[str, float | int]] = {}
    for depth, by_loop in matrix.items():
        cell = by_loop.get(depth, {}) if isinstance(by_loop, dict) else {}
        out[str(depth)] = {
            "correct": int(cell.get("correct", 0)),
            "total": int(cell.get("total", 0)),
            "accuracy": float(cell.get("accuracy", 0.0)),
        }
    return out


def locked_thresholds(rows_per_depth: int) -> dict[str, Any]:
    return {
        "selection_min_correct": dict(SELECTION_MIN_CORRECT),
        "selection_min_accuracy": {
            depth: correct / float(rows_per_depth)
            for depth, correct in SELECTION_MIN_CORRECT.items()
        },
        "nonregression_floors": dict(NONREGRESSION_FLOORS),
        "rows_per_depth": int(rows_per_depth),
        "selection_depths": sorted(SELECTION_MIN_CORRECT, key=int),
        "nonregression_depths": sorted(NONREGRESSION_FLOORS, key=int),
    }


def score_route(active_summary: dict[str, Any], *, rows_per_depth: int) -> dict[str, Any]:
    counts = diagonal_counts(active_summary)
    nonregression = {
        depth: {
            "accuracy": float(counts.get(depth, {}).get("accuracy", 0.0)),
            "floor": floor,
            "pass": float(counts.get(depth, {}).get("accuracy", 0.0)) >= floor,
        }
        for depth, floor in NONREGRESSION_FLOORS.items()
    }
    selection = {
        depth: {
            "correct": int(counts.get(depth, {}).get("correct", 0)),
            "total": int(counts.get(depth, {}).get("total", rows_per_depth)),
            "min_correct": min_correct,
            "accuracy": float(counts.get(depth, {}).get("accuracy", 0.0)),
            "pass": int(counts.get(depth, {}).get("correct", 0)) >= min_correct,
        }
        for depth, min_correct in SELECTION_MIN_CORRECT.items()
    }
    return {
        "diagonal_counts": counts,
        "locked_thresholds": locked_thresholds(rows_per_depth),
        "nonregression": nonregression,
        "selection": selection,
        "nonregression_pass": all(item["pass"] for item in nonregression.values()),
        "selection_pass": all(item["pass"] for item in selection.values()),
        "overall_pass": all(item["pass"] for item in nonregression.values())
        and all(item["pass"] for item in selection.values()),
    }


def ensure_frozen_eval_set(
    *,
    frozen_id: str,
    n_symbols: int,
    max_depth: int,
    rows_per_depth: int,
    value_prefix: str,
) -> dict[str, Any]:
    run_dir = ROOT / "outputs" / "stage5" / frozen_id
    data_dir = run_dir / "data"
    summary_path = run_dir / "summary.json"
    test_chain = data_dir / "test_chain_mcq.jsonl"
    if summary_path.exists() and test_chain.exists():
        return read_json(summary_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "training/generate_synthetic_depth_task.py",
            "--output_dir",
            path_for_cli(data_dir),
            "--n_symbols",
            str(n_symbols),
            "--max_depth",
            str(max_depth),
            "--rows_per_depth",
            str(rows_per_depth),
            "--seed",
            os.environ.get("STAGE5_ROUTE_FROZEN_EVAL_SEED", "20260704"),
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
        ]
    )
    payload = {
        "kind": "stage5_synthetic_depth_frozen_eval_set",
        "run_id": frozen_id,
        "status": "finished",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "value_prefix": value_prefix,
        "seed": os.environ.get("STAGE5_ROUTE_FROZEN_EVAL_SEED", "20260704"),
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "test_chain_mcq": path_for_cli(test_chain),
        "test_mcq": path_for_cli(data_dir / "test_mcq.jsonl"),
    }
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Frozen Synthetic Depth Eval Set - {frozen_id}",
                "",
                f"- Depths: `1..{max_depth}`",
                f"- Rows per depth: `{rows_per_depth}`",
                f"- Test chain set: `{payload['test_chain_mcq']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Stage 5 frozen synthetic eval set {frozen_id} [skip ci]")
    return payload


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    score = payload.get("route_score", {})
    lines = [
        f"# Depth Support Route Comparison - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Train support depth: `1..{payload.get('train_max_depth')}`",
        f"- Frozen eval set: `{payload.get('frozen_eval_set', {}).get('run_id')}`",
        f"- Active diagonal: `{payload.get('frozen_active_eval', {}).get('active_diagonal')}`",
        f"- Non-regression pass: `{score.get('nonregression_pass')}`",
        f"- Selection pass: `{score.get('selection_pass')}`",
        f"- Overall pass: `{score.get('overall_pass')}`",
        f"- Locked thresholds: `{score.get('locked_thresholds')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_ROUTE_RUN_ID") or time.strftime("stage5_depth_support_route_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_symbols = int(os.environ.get("STAGE5_ROUTE_N_SYMBOLS", "16"))
    train_max_depth = int(os.environ.get("STAGE5_ROUTE_TRAIN_MAX_DEPTH", "6"))
    eval_max_depth = int(os.environ.get("STAGE5_ROUTE_EVAL_MAX_DEPTH", "10"))
    rows_per_depth = int(os.environ.get("STAGE5_ROUTE_ROWS_PER_DEPTH", "256"))
    frozen_rows_per_depth = int(os.environ.get("STAGE5_ROUTE_FROZEN_ROWS_PER_DEPTH", "128"))
    total_steps = int(os.environ.get("STAGE5_ROUTE_TOTAL_STEPS", "2000"))
    dtype = os.environ.get("STAGE5_ROUTE_DTYPE", "bfloat16")
    threshold = float(os.environ.get("STAGE5_ROUTE_THRESHOLD", "0.71"))
    value_prefix = os.environ.get("STAGE5_ROUTE_VALUE_PREFIX", "letter:")
    frozen_id = os.environ.get("STAGE5_ROUTE_FROZEN_EVAL_ID", "stage5_synthetic_depth_frozen_eval_v1")

    frozen = ensure_frozen_eval_set(
        frozen_id=frozen_id,
        n_symbols=n_symbols,
        max_depth=eval_max_depth,
        rows_per_depth=frozen_rows_per_depth,
        value_prefix=value_prefix,
    )

    payload: dict[str, Any] = {
        "kind": "stage5_depth_support_route_comparison",
        "run_id": run_id,
        "status": "started",
        "train_max_depth": train_max_depth,
        "eval_max_depth": eval_max_depth,
        "rows_per_depth": rows_per_depth,
        "frozen_rows_per_depth": frozen_rows_per_depth,
        "total_steps": total_steps,
        "threshold": threshold,
        "frozen_eval_set": frozen,
        "route_hypothesis": (
            "Training support through depth 6 from the same pre-anneal checkpoint should extend "
            "the finite recurrent operator farther than matched 1-4 continuation."
        ),
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [sys.executable, "colab/run_stage5_chain_anneal_to_outcome.py"],
        env={
            "STAGE5_ANNEAL_RUN_ID": run_id,
            "STAGE5_ANNEAL_LOOP_LOSS_MODE": "per_loop_labels",
            "STAGE5_ANNEAL_INIT_CHECKPOINT": os.environ.get(
                "STAGE5_ROUTE_INIT_CHECKPOINT",
                "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            ),
            "STAGE5_ANNEAL_N_SYMBOLS": str(n_symbols),
            "STAGE5_ANNEAL_MAX_DEPTH": str(train_max_depth),
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": str(rows_per_depth),
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": os.environ.get("STAGE5_ROUTE_HELDOUT_ROWS_PER_DEPTH", "64"),
            "STAGE5_ANNEAL_TOTAL_STEPS": str(total_steps),
            "STAGE5_ANNEAL_HOLD_FRAC": os.environ.get("STAGE5_ROUTE_SAVE_MID_FRAC", "0.5"),
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": os.environ.get("STAGE5_ROUTE_PRELUDE_LR_MULT", "10.0"),
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_ROUTE_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ANNEAL_DTYPE": dtype,
            "STAGE5_ANNEAL_VALUE_PREFIX": value_prefix,
            "STAGE5_ANNEAL_SEED": os.environ.get("STAGE5_ROUTE_TRAIN_SEED", "20260705"),
        },
    )
    payload = read_json(run_dir / "summary.json")
    final_checkpoint = payload.get("final_checkpoint")
    if not final_checkpoint:
        raise RuntimeError("Route training did not produce final_checkpoint")

    artifact_summary = run_dir / "eval" / "frozen_depth10_artifact_check.json"
    active_rows = run_dir / "eval" / "frozen_depth10_active_rows.jsonl"
    active_summary = run_dir / "eval" / "frozen_depth10_active_summary.json"
    test_chain = ROOT / str(frozen["test_chain_mcq"])
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            str(final_checkpoint),
            "--output_summary",
            path_for_cli(artifact_summary),
            "--max_loops",
            str(eval_max_depth),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            str(final_checkpoint),
            "--output_jsonl",
            path_for_cli(active_rows),
            "--output_summary",
            path_for_cli(active_summary),
            "--loop_counts",
            ",".join(str(idx) for idx in range(1, eval_max_depth + 1)),
            "--threshold",
            str(threshold),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    active_payload = read_json(active_summary)
    route_score = score_route(active_payload, rows_per_depth=frozen_rows_per_depth)
    payload.update(
        {
            "kind": "stage5_depth_support_route_comparison",
            "status": "finished_with_frozen_eval",
            "train_max_depth": train_max_depth,
            "eval_max_depth": eval_max_depth,
            "frozen_eval_set": frozen,
            "frozen_artifact_check": read_json(artifact_summary),
            "frozen_active_eval": {
                "active_rows": path_for_cli(active_rows),
                "active_summary": path_for_cli(active_summary),
                "active_diagonal": {
                    str(depth): float(value)
                    for depth, value in active_payload.get("active_diagonal", {}).items()
                },
                "active_total": active_payload.get("active_total", {}),
                "above_diagonal": active_payload.get("above_diagonal", {}),
            },
            "route_score": route_score,
            "decision_read": {
                "question": "Does extending support from depth 1-4 to 1-6 extend held-out validity at depths 7-10?",
                "single_variable": "training_support_depth",
                "comparator": "stage5_chain_continuation_attribution_20260704_163056",
                "selection_depths": sorted(SELECTION_MIN_CORRECT, key=int),
                "nonregression_depths": sorted(NONREGRESSION_FLOORS, key=int),
                "overall_pass": route_score["overall_pass"],
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 depth support route comparison {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
