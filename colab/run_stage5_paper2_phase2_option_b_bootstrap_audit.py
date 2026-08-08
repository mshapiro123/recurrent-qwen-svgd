"""Run and publish the read-only Option B document-bootstrap audit."""

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
RUN_ID = "stage5_paper2_phase2_option_b_bootstrap_audit_20260808"
SOURCE_ID = "stage5_paper2_phase2_option_b_20260807"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
SOURCE_SUMMARY = ROOT / "outputs/stage5" / SOURCE_ID / "summary.json"
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
SOURCE_PRIVATE = DRIVE_ROOT / SOURCE_ID / "private/option_b"
STAGE0A_MANIFEST = DRIVE_ROOT / STAGE0A_ID / "private/stage0a/sample_manifest.jsonl"
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
        print("option_b_bootstrap_audit_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("option_b_bootstrap_audit_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_option_b_bootstrap_audit_status",
                "status": status,
                "updated_at_unix": time.time(),
                "mode": "read_only_cpu_post_processing",
                "optimizer_updates": 0,
                "model_loaded": False,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"option_b_bootstrap_audit_status status={status} details={details}", flush=True)


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Option B document-bootstrap audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [SOURCE_SUMMARY, STAGE0A_SUMMARY, STAGE0A_MANIFEST, SOURCE_PRIVATE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Option B bootstrap-audit inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("auditing_saved_rows")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_option_b_bootstrap_audit",
            "--source_summary",
            str(SOURCE_SUMMARY),
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--manifest",
            str(STAGE0A_MANIFEST),
            "--private_root",
            str(SOURCE_PRIVATE),
            "--output_summary",
            str(RUN_DIR / "summary.json"),
            "--output_markdown",
            str(RUN_DIR / "receipt.md"),
            "--bootstrap_replicates",
            "10000",
            "--bootstrap_seed",
            "20260808",
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["optimizer_updates"] != 0 or summary["model_loaded"] is not False:
        raise RuntimeError("Option B bootstrap audit crossed its read-only boundary")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "summary.json", receipt_dir / "summary.json")
    shutil.copy2(RUN_DIR / "receipt.md", receipt_dir / "receipt.md")
    write_status("publishing", corrected_scripted_reading=summary["corrected_scripted_reading"])
    commit = publish()
    write_status(
        "complete",
        publish_commit=commit,
        corrected_scripted_reading=summary["corrected_scripted_reading"],
    )
    print(
        json.dumps(
            {
                "publish_commit": commit,
                "corrected_scripted_reading": summary["corrected_scripted_reading"],
                "drive_receipt": str(receipt_dir / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        DRIVE_RUN.mkdir(parents=True, exist_ok=True)
        write_status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
