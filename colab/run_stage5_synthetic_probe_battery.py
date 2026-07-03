"""Run the synthetic-depth state probe battery on a corrected chain checkpoint."""

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


def grid_diagonal(summary: dict[str, Any]) -> dict[str, float]:
    grid = summary.get("grid", {})
    out: dict[str, float] = {}
    for loop, by_target in grid.items():
        if loop in by_target:
            out[loop] = float(by_target[loop]["accuracy"])
    return out


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    probe = payload.get("probe", {})
    lines = [
        f"# Synthetic Depth Probe Battery - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source checkpoint: `{payload['checkpoint']}`",
        f"- Probe summary: `{payload.get('probe_summary')}`",
        f"- Diagonal state->f^k accuracies: `{payload.get('probe_diagonal')}`",
        f"- Loop-index probe: `{probe.get('loop_index_probe')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_PROBE_RUN_ID") or time.strftime("stage5_synthetic_probe_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    n_symbols = int(os.environ.get("STAGE5_PROBE_N_SYMBOLS", "16"))
    max_depth = int(os.environ.get("STAGE5_PROBE_MAX_DEPTH", "6"))
    rows_per_depth = int(os.environ.get("STAGE5_PROBE_ROWS_PER_DEPTH", "64"))
    value_prefix = os.environ.get("STAGE5_PROBE_VALUE_PREFIX", "letter:")
    dtype = os.environ.get("STAGE5_PROBE_DTYPE", "bfloat16")
    source_ref = os.environ.get(
        "STAGE5_PROBE_CHECKPOINT",
        "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
    )
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_ref,
        run_dir / "restored" / "scaled_corrected_final.pt",
        label="scaled_corrected_final",
    )
    payload: dict[str, Any] = {
        "kind": "stage5_synthetic_depth_probe_battery",
        "run_id": run_id,
        "status": "started",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
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
            os.environ.get("STAGE5_PROBE_SEED", "20260704"),
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
        ]
    )
    probe_summary = run_dir / "probe" / "synthetic_depth_state_probe.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_probe.py",
            "--data_jsonl",
            path_for_cli(data_dir / "test_chain_mcq.jsonl"),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(probe_summary),
            "--loop_counts",
            os.environ.get("STAGE5_PROBE_LOOP_COUNTS", "1,2,3,4,5,6"),
            "--target_steps",
            os.environ.get("STAGE5_PROBE_TARGET_STEPS", "0,1,2,3,4,5,6"),
            "--n_symbols",
            str(n_symbols),
            "--value_prefix",
            value_prefix,
            "--permutations",
            os.environ.get("STAGE5_PROBE_PERMUTATIONS", "20"),
            "--ridge_l2",
            os.environ.get("STAGE5_PROBE_RIDGE_L2", "0.01"),
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
    probe = read_json(probe_summary)
    payload.update(
        {
            "status": "finished",
            "data_summary": path_for_cli(data_dir / "summary.json"),
            "probe_summary": path_for_cli(probe_summary),
            "probe_diagonal": grid_diagonal(probe),
            "probe": {
                "rows": probe.get("rows"),
                "state_records": probe.get("state_records"),
                "loop_counts": probe.get("loop_counts"),
                "target_steps": probe.get("target_steps"),
                "loop_index_probe": probe.get("loop_index_probe"),
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 synthetic probe battery {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
