"""Run the sealed P3.1 source, reference, verified-label, and sentinel pass."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p31_completion_20260810"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
SOURCE_MANIFEST = ROOT / "training/paper2_phase3_p31_source_manifest.json"
TIER1 = ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=600)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("phase3_p31_completion_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_p31_completion_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(value: str, **details: object) -> None:
    write_json(
        RECEIPT_DIR / "status.json",
        {
            "kind": "paper2_phase3_p31_completion_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            "p33_training_authorized": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"phase3_p31_completion_status status={value} details={details}", flush=True)


def main() -> int:
    for path in (SOURCE_MANIFEST, TIER1):
        if not path.exists():
            raise FileNotFoundError(f"missing P3.1 source input: {path}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    source_summary = RECEIPT_DIR / "source_summary.json"
    rows = PRIVATE_DIR / "p31_partitioned_rows.jsonl"
    status("materializing_score_blind_sources")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase3_p31_sources",
            "--source_manifest",
            str(SOURCE_MANIFEST),
            "--tier1_path",
            str(TIER1),
            "--private_rows",
            str(rows),
            "--output_summary",
            str(source_summary),
            "--mmlu_slice_size",
            "512",
        ]
    )
    status("sealing_confirm_before_model_load")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.seal_paper2_phase3_p31",
            "--source_summary",
            str(source_summary),
            "--source_rows",
            str(rows),
            "--source_manifest",
            str(SOURCE_MANIFEST),
            "--output_dir",
            str(RECEIPT_DIR / "confirm_seals"),
        ]
    )
    confirm_seal_ledger = RECEIPT_DIR / "confirm_seals/confirm_seal_ledger.json"
    if not confirm_seal_ledger.is_file():
        raise RuntimeError("CONFIRM seal ledger was not written before model scoring")
    status("scoring_dev_and_verified_train_confirm_inaccessible")
    score_dir = PRIVATE_DIR / "model_scores"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p31_references",
            "--rows_jsonl",
            str(rows),
            "--output_dir",
            str(score_dir),
            "--model_key",
            "base",
            "--model_key",
            "teacher_14b",
            "--mcq_candidate_batch_size",
            os.environ.get("STAGE5_P31_MCQ_CANDIDATE_BATCH_SIZE", "32"),
            "--generation_batch_size",
            os.environ.get("STAGE5_P31_GENERATION_BATCH_SIZE", "4"),
            "--confirm_seal_ledger",
            str(confirm_seal_ledger),
        ]
    )
    status("assembling_references_verified_labels_and_sentinel")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase3_p31_completion",
            "--rows_jsonl",
            str(rows),
            "--source_summary",
            str(source_summary),
            "--source_manifest",
            str(SOURCE_MANIFEST),
            "--base_scores",
            str(score_dir / "base_scores.jsonl"),
            "--teacher_scores",
            str(score_dir / "teacher_14b_scores.jsonl"),
            "--model_score_receipts",
            str(score_dir / "model_score_receipts.json"),
            "--private_dir",
            str(PRIVATE_DIR),
            "--receipt_dir",
            str(RECEIPT_DIR),
        ]
    )
    summary = json.loads((RECEIPT_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["optimizer_steps"] != 0 or summary["p33_training_authorized"]:
        raise RuntimeError("P3.1 completion crossed the no-training boundary")
    status("complete", summary=str(RECEIPT_DIR / "summary.json"))
    print(json.dumps({"status": summary["status"], "drive": str(DRIVE_RUN)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
