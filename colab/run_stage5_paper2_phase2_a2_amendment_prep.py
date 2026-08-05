"""Verify and publish the CPU-only A2 amendment-preparation package."""

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
RUN_ID = "stage5_paper2_phase2_a2_amendment_prep_20260805"
CALIBRATION_ID = "stage5_paper2_phase2_a2_calibration_20260805"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
CALIBRATION_SUMMARY = ROOT / "outputs/stage5" / CALIBRATION_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_PRIVATE = DRIVE_ROOT / CALIBRATION_ID / "private/calibration"
DRIVE_RUN = DRIVE_ROOT / RUN_ID


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=500)
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
        print("a2_amendment_prep_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_amendment_prep_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_a2_amendment_prep_status",
                "status": status,
                "updated_at_unix": time.time(),
                "optimizer_updates": 0,
                "a2_training_launched": False,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"a2_amendment_prep_status status={status} details={details}", flush=True)


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record A2 amendment preparation [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [
        CALIBRATION_SUMMARY,
        DRIVE_PRIVATE / "seed_0_batch_rows.json",
        DRIVE_PRIVATE / "seed_1_batch_rows.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing A2 calibration reconciliation inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("verifying_public_private_receipts")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "training.run_paper2_phase2_a2_amendment_prep",
            "--calibration_summary",
            str(CALIBRATION_SUMMARY),
            "--private_dir",
            str(DRIVE_PRIVATE),
            "--output_dir",
            str(RUN_DIR),
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["optimizer_updates"] != 0 or summary["a2_training_launched"] is not False:
        raise RuntimeError("A2 amendment preparation crossed its no-training boundary")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, receipt_dir / path.name)
    write_status("publishing", pathology_verdict=summary["pathology_verdict"])
    commit = publish()
    write_status(
        "complete",
        publish_commit=commit,
        pathology_verdict=summary["pathology_verdict"],
        strategy_lock_required=True,
    )
    print(json.dumps({"summary": summary, "publish_commit": commit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            try:
                write_status(
                    "failed",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
            except Exception as status_error:
                print(f"status_write_failed={status_error!r}", flush=True)
        raise
