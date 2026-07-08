"""Run the same-reader final-symbol release gate for synthetic-depth checkpoints."""

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


DEFAULT_SOURCE = "outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json"


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


def source_checkpoint(payload: dict[str, Any]) -> str:
    for key in ("final_checkpoint_drive_backup", "final_checkpoint", "checkpoint_drive_backup", "checkpoint"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Source summary does not expose final_checkpoint/checkpoint")


def source_data_jsonl(payload: dict[str, Any]) -> str:
    frozen = payload.get("frozen_eval_set") if isinstance(payload.get("frozen_eval_set"), dict) else {}
    value = frozen.get("test_chain_mcq") or payload.get("data_jsonl")
    if isinstance(value, str) and value.strip():
        return value
    raise KeyError("Source summary does not expose frozen_eval_set.test_chain_mcq")


def max_depth_from_source(payload: dict[str, Any]) -> int:
    frozen = payload.get("frozen_eval_set") if isinstance(payload.get("frozen_eval_set"), dict) else {}
    if frozen.get("max_depth"):
        return int(frozen["max_depth"])
    active = payload.get("frozen_active_eval") if isinstance(payload.get("frozen_active_eval"), dict) else {}
    diag = active.get("active_diagonal") if isinstance(active.get("active_diagonal"), dict) else {}
    if diag:
        return max(int(depth) for depth in diag)
    return int(os.environ.get("STAGE5_SAME_READER_MAX_LOOPS", "14"))


def latest_checkpoint_eval(payload: dict[str, Any]) -> dict[str, Any] | None:
    evals = [item for item in payload.get("checkpoint_evals") or [] if isinstance(item, dict)]
    if not evals:
        return None
    return max(evals, key=lambda item: int(item.get("step") or -1))


def source_active_diagonal(payload: dict[str, Any]) -> dict[str, float]:
    """Return the active-label diagonal that same-reader scoring should match.

    Route-style summaries store it under ``frozen_active_eval``.  N24 summaries
    store per-checkpoint evals and the release receipt must compare against the
    latest checkpoint eval, not an older partial readout.
    """

    active = payload.get("frozen_active_eval") if isinstance(payload.get("frozen_active_eval"), dict) else {}
    diagonal = active.get("active_diagonal") if isinstance(active.get("active_diagonal"), dict) else {}
    if diagonal:
        return {str(depth): float(value) for depth, value in diagonal.items()}

    latest = latest_checkpoint_eval(payload)
    score = latest.get("score") if isinstance(latest, dict) else {}
    counts = score.get("diagonal_counts") if isinstance(score, dict) else {}
    if isinstance(counts, dict) and counts:
        return {str(depth): float(value) for depth, value in diagonal_counts_to_accuracy(counts).items()}
    return {}


def same_reader_diagonal(same_reader: dict[str, Any]) -> dict[str, float]:
    by_depth = same_reader.get("by_depth") if isinstance(same_reader.get("by_depth"), dict) else {}
    return {
        str(depth): float(item.get("same_reader_accuracy"))
        for depth, item in by_depth.items()
        if isinstance(item, dict) and item.get("same_reader_accuracy") is not None
    }


def identity_check_against_active(
    *,
    source_payload: dict[str, Any],
    same_reader: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    expected = source_active_diagonal(source_payload)
    observed = same_reader_diagonal(same_reader)
    common = sorted(set(expected) & set(observed), key=int)
    deltas = {
        depth: {
            "active_accuracy": expected[depth],
            "same_reader_accuracy": observed[depth],
            "delta": observed[depth] - expected[depth],
            "abs_delta": abs(observed[depth] - expected[depth]),
        }
        for depth in common
    }
    missing_expected = sorted(set(expected) - set(observed), key=int)
    missing_observed = sorted(set(observed) - set(expected), key=int)
    max_abs_delta = max((item["abs_delta"] for item in deltas.values()), default=None)
    pass_check = bool(common) and not missing_expected and (max_abs_delta is not None and max_abs_delta <= tolerance)
    return {
        "kind": "same_reader_active_identity_check",
        "expected_source": "source active-label diagonal",
        "tolerance": tolerance,
        "depths_compared": common,
        "missing_expected_depths": missing_expected,
        "extra_same_reader_depths": missing_observed,
        "max_abs_delta": max_abs_delta,
        "pass": pass_check,
        "deltas": deltas,
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("same_reader_final", {})
    lines = [
        f"# Same-Reader Final-Symbol Gate - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Data: `{payload['data_jsonl']}`",
        f"- Same-reader total: `{summary.get('same_reader_total')}`",
        f"- Mapped-final total: `{summary.get('mapped_final_total')}`",
        f"- Suspended reader policy: `{summary.get('metric_policy', {}).get('suspended_reader')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SAME_READER_RUN_ID") or time.strftime(
        "stage5_same_reader_final_symbol_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_summary = os.environ.get("STAGE5_SAME_READER_SOURCE_SUMMARY", DEFAULT_SOURCE)
    source_payload = read_json(source_summary)
    checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        os.environ.get("STAGE5_SAME_READER_CHECKPOINT", source_checkpoint(source_payload)),
        run_dir / "restored" / "source_checkpoint.pt",
        label="same_reader_source",
    )
    data_jsonl = os.environ.get("STAGE5_SAME_READER_DATA_JSONL", source_data_jsonl(source_payload))
    max_loops = int(os.environ.get("STAGE5_SAME_READER_MAX_LOOPS", str(max_depth_from_source(source_payload))))
    dtype = os.environ.get("STAGE5_SAME_READER_DTYPE", "bfloat16")
    value_prefix = os.environ.get("STAGE5_SAME_READER_VALUE_PREFIX", "letter:")
    rows_path = run_dir / "eval" / "same_reader_final_rows.jsonl"
    summary_path = run_dir / "eval" / "same_reader_final_summary.json"

    payload: dict[str, Any] = {
        "kind": "stage5_same_reader_final_symbol_gate",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "data_jsonl": data_jsonl,
        "max_loops": max_loops,
        "metric_policy": "Use full-symbol argmax at loop == depth; MCQ option-text tables are diagnostic only.",
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_final_symbol.py",
            "--data_jsonl",
            data_jsonl,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--max_loops",
            str(max_loops),
            "--threshold",
            os.environ.get("STAGE5_SAME_READER_THRESHOLD", "0.71"),
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
    same_reader = read_json(summary_path)
    identity_check = None
    if os.environ.get("STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        identity_check = identity_check_against_active(
            source_payload=source_payload,
            same_reader=same_reader,
            tolerance=float(os.environ.get("STAGE5_SAME_READER_IDENTITY_TOLERANCE", "0.000001")),
        )
        if not identity_check["pass"]:
            payload.update({"status": "identity_check_failed", "same_reader_final": same_reader, "identity_check": identity_check})
            write_json(run_dir / "summary.json", payload)
            write_markdown(run_dir, payload)
            raise RuntimeError(f"Same-reader identity check failed: {identity_check}")
    payload.update(
        {
            "status": "finished",
            "same_reader_final": same_reader,
            "identity_check": identity_check,
            "rows_path": path_for_cli(rows_path),
            "summary_path": path_for_cli(summary_path),
            "release_gate": {
                "same_reader_decoder_wired": True,
                "retire_suspended_mcq_final_tables": True,
            },
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 same-reader final-symbol gate {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
