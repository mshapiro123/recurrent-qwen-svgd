"""Run the N24 permutation zero-shot control.

The control asks whether the N24 recurrent operator transfers from arbitrary
finite functions to bijective/permutation functions without extra calibration.
It is eval-only and compares each depth to the arbitrary-table diagonal from
the same N24 checkpoint.
"""

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

from colab.stage5_chain_consolidation_utils import (  # noqa: E402
    ROOT,
    path_for_cli,
    publish_run,
    read_json,
    resolve_checkpoint_reference,
    write_json,
)
from colab.stage5_frontier_metrics import diagonal_counts_to_accuracy  # noqa: E402


DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json"


def run(cmd: list[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def latest_n24_diagonal(payload: dict[str, Any]) -> dict[str, float]:
    evals = [item for item in payload.get("checkpoint_evals") or [] if isinstance(item, dict)]
    if not evals:
        raise KeyError("N24 source summary has no checkpoint_evals")
    latest = max(evals, key=lambda item: int(item.get("step") or -1))
    counts = ((latest.get("score") or {}).get("diagonal_counts") or {})
    if not counts:
        raise KeyError("N24 source summary latest checkpoint eval has no diagonal_counts")
    return {str(depth): float(value) for depth, value in diagonal_counts_to_accuracy(counts).items()}


def checkpoint_reference(payload: dict[str, Any]) -> str:
    for key in ("final_checkpoint_drive_backup", "final_checkpoint", "checkpoint_drive_backup", "checkpoint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Source summary does not expose a checkpoint")


def ensure_permutation_eval_set(
    *,
    eval_id: str,
    n_symbols: int,
    max_depth: int,
    rows_per_depth: int,
    seed: str,
    value_prefix: str,
) -> dict[str, Any]:
    run_dir = ROOT / "outputs" / "stage5" / eval_id
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
            seed,
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
            "--permutation",
        ]
    )
    payload = {
        "kind": "stage5_synthetic_depth_permutation_eval_set",
        "run_id": eval_id,
        "status": "finished",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "value_prefix": value_prefix,
        "seed": seed,
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "test_chain_mcq": path_for_cli(test_chain),
    }
    write_json(summary_path, payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Frozen Synthetic Permutation Eval Set - {eval_id}",
                "",
                f"- Depths: `1..{max_depth}`",
                f"- Rows per depth: `{rows_per_depth}`",
                f"- Test chain set: `{payload['test_chain_mcq']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Stage 5 permutation synthetic eval set {eval_id} [skip ci]")
    return payload


def summarize_parity(*, arbitrary_diag: dict[str, float], permutation_diag: dict[str, float], tolerance: float) -> dict[str, Any]:
    common = sorted(set(arbitrary_diag) & set(permutation_diag), key=int)
    deltas = {
        depth: {
            "arbitrary_accuracy": arbitrary_diag[depth],
            "permutation_accuracy": permutation_diag[depth],
            "delta": permutation_diag[depth] - arbitrary_diag[depth],
            "abs_delta": abs(permutation_diag[depth] - arbitrary_diag[depth]),
        }
        for depth in common
    }
    max_abs_delta = max((item["abs_delta"] for item in deltas.values()), default=None)
    return {
        "tolerance": tolerance,
        "depths_compared": common,
        "max_abs_delta": max_abs_delta,
        "within_tolerance_each_depth": max_abs_delta is not None and max_abs_delta <= tolerance,
        "deltas": deltas,
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    parity = payload.get("parity", {})
    lines = [
        f"# Permutation Zero-Shot Control - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Eval set: `{payload.get('permutation_eval_set', {}).get('run_id')}`",
        f"- Max abs delta vs arbitrary table: `{parity.get('max_abs_delta')}`",
        f"- Within tolerance each depth: `{parity.get('within_tolerance_each_depth')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_PERM_ZERO_SHOT_RUN_ID") or time.strftime(
        "stage5_permutation_zero_shot_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_summary = os.environ.get("STAGE5_PERM_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    source_payload = read_json(source_summary)
    n_symbols = int(os.environ.get("STAGE5_PERM_N_SYMBOLS", "24"))
    max_depth = int(os.environ.get("STAGE5_PERM_MAX_DEPTH", "12"))
    rows_per_depth = int(os.environ.get("STAGE5_PERM_ROWS_PER_DEPTH", "128"))
    value_prefix = os.environ.get("STAGE5_PERM_VALUE_PREFIX", "letter:")
    eval_id = os.environ.get("STAGE5_PERM_EVAL_ID", "stage5_synthetic_depth_permutation_eval_v1_n24_depth12")
    frozen = ensure_permutation_eval_set(
        eval_id=eval_id,
        n_symbols=n_symbols,
        max_depth=max_depth,
        rows_per_depth=rows_per_depth,
        seed=os.environ.get("STAGE5_PERM_SEED", "20260708"),
        value_prefix=value_prefix,
    )
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        os.environ.get("STAGE5_PERM_CHECKPOINT", checkpoint_reference(source_payload)),
        run_dir / "restored" / "n24_checkpoint.pt",
        label="permutation_n24",
    )
    artifact_summary = run_dir / "eval" / "artifact_check.json"
    active_rows = run_dir / "eval" / "active_rows.jsonl"
    active_summary = run_dir / "eval" / "active_summary.json"
    data_jsonl = frozen["test_chain_mcq"]
    dtype = os.environ.get("STAGE5_PERM_DTYPE", "bfloat16")
    payload: dict[str, Any] = {
        "kind": "stage5_permutation_zero_shot",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "permutation_eval_set": frozen,
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            data_jsonl,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(artifact_summary),
            "--max_loops",
            str(max_depth),
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
            data_jsonl,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(active_rows),
            "--output_summary",
            path_for_cli(active_summary),
            "--loop_counts",
            ",".join(str(depth) for depth in range(1, max_depth + 1)),
            "--threshold",
            os.environ.get("STAGE5_PERM_THRESHOLD", "0.71"),
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
    permutation_diag = {str(depth): float(value) for depth, value in active.get("active_diagonal", {}).items()}
    arbitrary_diag = {
        depth: value
        for depth, value in latest_n24_diagonal(source_payload).items()
        if int(depth) <= max_depth
    }
    parity = summarize_parity(
        arbitrary_diag=arbitrary_diag,
        permutation_diag=permutation_diag,
        tolerance=float(os.environ.get("STAGE5_PERM_PARITY_TOLERANCE", "0.05")),
    )
    status = "permutation_zero_shot_parity_pass" if parity["within_tolerance_each_depth"] else "permutation_zero_shot_needs_calibration"
    payload.update(
        {
            "status": status,
            "artifact_check": read_json(artifact_summary),
            "active_eval": {
                "active_rows": path_for_cli(active_rows),
                "active_summary": path_for_cli(active_summary),
                "active_diagonal": permutation_diag,
                "active_total": active.get("active_total", {}),
            },
            "arbitrary_n24_diagonal": arbitrary_diag,
            "parity": parity,
            "decision": (
                "Permutation zero-shot parity holds within tolerance."
                if status == "permutation_zero_shot_parity_pass"
                else "Permutation zero-shot falls outside tolerance; defer calibration until after receipt review."
            ),
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 permutation zero-shot {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": status, "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
