"""Run eval-only synthetic-depth extrapolation from the corrected chain checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from colab.stage5_chain_consolidation_utils import (
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    resolve_checkpoint_reference,
    write_json,
)


def run(cmd: list[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    diag = payload.get("active_eval", {}).get("active_diagonal", {})
    lines = [
        f"# Synthetic Depth Extrapolation - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source checkpoint: `{payload['checkpoint']}`",
        f"- Artifact check pass: `{payload.get('artifact_check', {}).get('pass')}`",
        f"- Active diagonal: `{diag}`",
        f"- Depth 5 read: `{payload.get('extrapolation_read', {}).get('5')}`",
        f"- Depth 6 read: `{payload.get('extrapolation_read', {}).get('6')}`",
        f"- Controls pass: `{payload.get('control_read', {}).get('pass')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def classify_depth(value: float, *, lower: float, bar: float) -> str:
    if value >= lower and value >= bar:
        return "inside_or_above_conservative_band"
    if value >= bar:
        return "partial_extrapolation_below_conservative_band"
    return "below_bar"


def main() -> int:
    run_id = os.environ.get("STAGE5_EXTRAP_RUN_ID") or time.strftime("stage5_depth_extrapolation_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    n_symbols = int(os.environ.get("STAGE5_EXTRAP_N_SYMBOLS", "16"))
    max_depth = max(int(item) for item in os.environ.get("STAGE5_EXTRAP_DEPTHS", "1,2,3,4,5,6").split(","))
    rows_per_depth = int(os.environ.get("STAGE5_EXTRAP_ROWS_PER_DEPTH", "64"))
    max_loops = int(os.environ.get("STAGE5_EXTRAP_MAX_LOOPS", str(max_depth)))
    value_prefix = os.environ.get("STAGE5_EXTRAP_VALUE_PREFIX", "letter:")
    dtype = os.environ.get("STAGE5_EXTRAP_DTYPE", "bfloat16")
    threshold = float(os.environ.get("STAGE5_EXTRAP_THRESHOLD", "0.71"))
    source_ref = os.environ.get(
        "STAGE5_EXTRAP_CHECKPOINT",
        "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
    )
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_ref,
        run_dir / "restored" / "scaled_corrected_final.pt",
        label="scaled_corrected_final",
    )

    payload: dict[str, Any] = {
        "kind": "stage5_depth_extrapolation_eval",
        "run_id": run_id,
        "status": "started",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "max_loops": max_loops,
        "threshold": threshold,
        "value_prefix": value_prefix,
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "pre_registered_bands": {
            "5": {
                "mle_prediction": 0.946,
                "mle_interval": [0.890, 1.000],
                "conservative_prediction": 0.903,
                "conservative_interval": [0.831, 0.976],
            },
            "6": {
                "mle_prediction": 0.935,
                "mle_interval": [0.875, 0.996],
                "conservative_prediction": 0.885,
                "conservative_interval": [0.807, 0.963],
            },
        },
    }
    write_json(run_dir / "summary.json", payload)

    data_dir = run_dir / "data"
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
            os.environ.get("STAGE5_EXTRAP_SEED", "20260703"),
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
        ]
    )
    test_chain = data_dir / "test_chain_mcq.jsonl"
    artifact_summary = run_dir / "artifact_check.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(artifact_summary),
            "--max_loops",
            str(max_loops),
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
    active_rows = run_dir / "eval" / "depth_extrapolation_active_rows.jsonl"
    active_summary = run_dir / "eval" / "depth_extrapolation_active_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(test_chain),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(active_rows),
            "--output_summary",
            path_for_cli(active_summary),
            "--loop_counts",
            ",".join(str(idx) for idx in range(1, max_loops + 1)),
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
    active = read_json(active_summary)
    diag = {str(key): float(value) for key, value in active.get("active_diagonal", {}).items()}
    bands = payload["pre_registered_bands"]
    extrapolation_read = {
        depth: {
            "observed": diag.get(depth, 0.0),
            "classification": classify_depth(
                diag.get(depth, 0.0),
                lower=float(bands[depth]["conservative_interval"][0]),
                bar=threshold,
            ),
            "conservative_interval": bands[depth]["conservative_interval"],
        }
        for depth in ("5", "6")
    }
    control_values = [diag.get(str(depth), 0.0) for depth in range(1, 5)]
    payload.update(
        {
            "status": "finished",
            "data_summary": path_for_cli(data_dir / "summary.json"),
            "test_chain_mcq": path_for_cli(test_chain),
            "artifact_check": read_json(artifact_summary),
            "active_eval": {
                "active_rows": path_for_cli(active_rows),
                "active_summary": path_for_cli(active_summary),
                "active_diagonal": diag,
                "active_total": active.get("active_total", {}),
                "above_diagonal": active.get("above_diagonal", {}),
            },
            "control_read": {
                "depths": [1, 2, 3, 4],
                "min_active_diagonal": min(control_values) if control_values else 0.0,
                "pass": bool(control_values) and min(control_values) >= threshold,
            },
            "extrapolation_read": extrapolation_read,
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 depth extrapolation {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
