"""Publish a fast-vs-slow active-label scorer equivalence receipt."""

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
    restore_checkpoint,
    write_json,
)


DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json"
DEFAULT_DATA_JSONL = "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"


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
    eq = payload.get("equivalence") or {}
    lines = [
        f"# Synthetic Active-Label Scorer Equivalence - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Data: `{payload['data_jsonl']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Records checked: `{eq.get('records')}`",
        f"- Mismatches: `{len(eq.get('mismatches') or [])}`",
        f"- Pass: `{eq.get('pass')}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SCORER_EQUIV_RUN_ID") or time.strftime(
        "stage5_scorer_equivalence_%Y%m%d_%H%M%S"
    )
    source_summary = os.environ.get("STAGE5_SCORER_EQUIV_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    data_jsonl = os.environ.get("STAGE5_SCORER_EQUIV_DATA_JSONL", DEFAULT_DATA_JSONL)
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(source_summary)
    checkpoint = restore_checkpoint(
        [
            source.get("final_checkpoint_drive_backup"),
            source.get("final_checkpoint"),
            source.get("checkpoint_drive_backup"),
            source.get("checkpoint"),
        ],
        run_dir / "restored" / "source_checkpoint.pt",
        label="scorer_equivalence_source",
    )
    equivalence_path = run_dir / "eval" / "scorer_equivalence_summary.json"
    run(
        [
            sys.executable,
            "eval/check_synthetic_active_label_scorer_equivalence.py",
            "--data_jsonl",
            data_jsonl,
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(equivalence_path),
            "--max_rows",
            os.environ.get("STAGE5_SCORER_EQUIV_MAX_ROWS", "2"),
            "--loop_counts",
            os.environ.get("STAGE5_SCORER_EQUIV_LOOP_COUNTS", "1,2,12,22"),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            "letter:",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            os.environ.get("STAGE5_SCORER_EQUIV_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    equivalence = read_json(equivalence_path)
    payload: dict[str, Any] = {
        "kind": "stage5_scorer_equivalence_receipt",
        "run_id": run_id,
        "status": "passed" if equivalence.get("pass") else "failed",
        "source_summary": source_summary,
        "data_jsonl": data_jsonl,
        "checkpoint": path_for_cli(checkpoint),
        "equivalence_summary": path_for_cli(equivalence_path),
        "equivalence": equivalence,
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    if os.environ.get("STAGE5_SCORER_EQUIV_PUBLISH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
        publish_run(run_dir, message=f"Record Stage 5 scorer equivalence receipt {run_id} [skip ci]")
    else:
        print("Skipping publish because STAGE5_SCORER_EQUIV_PUBLISH=0", flush=True)
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0 if equivalence.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
