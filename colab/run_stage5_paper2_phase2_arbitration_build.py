"""Run and publish CPU-only Phase-2 canonicalizer arbitration and student build."""

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
RUN_ID = "stage5_paper2_phase2_arbitration_build_20260804"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
EXP0A_ID = "stage5_paper2_phase2_exp0a_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
EXP0A_SUMMARY = ROOT / "outputs/stage5" / EXP0A_ID / "summary.json"
STAGE0A_PRIVATE = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{STAGE0A_ID}/private/stage0a"
)
DRIVE_RUN = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}"
)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_arbitration_build_status",
                "status": status,
                "updated_at_unix": time.time(),
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"phase2_status status={status} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=240)
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
        print("child_process_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("child_process_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 arbitration and build receipts [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    # Safety marker: CPU high RAM cached canonicalizer arbitration and loss-free student build
    required = [
        EXP0A_SUMMARY,
        STAGE0A_PRIVATE / "sample_manifest.jsonl",
        STAGE0A_PRIVATE / "model_cache/teacher_14b/summary.json",
    ]
    for path in required:
        print(f"phase2_preflight path={path} exists={path.exists()}", flush=True)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase-2 arbitration inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    private = DRIVE_RUN / "private"
    arbitration = RUN_DIR / "canonicalizer_arbitration_summary.json"
    build = RUN_DIR / "student_build_summary.json"
    # Fail fast on the checkpoint-integrated identity battery before the six
    # expensive cached SVD fits, and preserve its receipt immediately.
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    drive_build = receipt_dir / build.name
    if drive_build.is_file():
        payload = json.loads(drive_build.read_text(encoding="utf-8"))
        if payload.get("status") == "complete_build_only_no_losses":
            shutil.copy2(drive_build, build)
            print(f"phase2_build_resume path={drive_build}", flush=True)
        else:
            raise RuntimeError(f"invalid existing build receipt: {drive_build}")
    else:
        write_status("student_build_start")
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_dc2_student_build",
                "--output_summary",
                str(build),
                "--model_name",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ]
        )
        shutil.copy2(build, drive_build)
    write_status("arbitration_start", completed_fit_artifacts=len(list((private / "canonicalizer").glob("*.pt"))))
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_canonicalizer_arbitration",
            "--stage0a_private",
            str(STAGE0A_PRIVATE),
            "--exp0a_summary",
            str(EXP0A_SUMMARY),
            "--output_private",
            str(private / "canonicalizer"),
            "--output_summary",
            str(arbitration),
        ]
    )
    arbitration_payload = json.loads(arbitration.read_text(encoding="utf-8"))
    build_payload = json.loads(build.read_text(encoding="utf-8"))
    if arbitration_payload.get("training_started") or build_payload.get("training_started"):
        raise RuntimeError("CPU-only target unexpectedly trained parameters")
    summary = {
        "kind": "paper2_phase2_arbitration_build_bundle",
        "status": "complete_cpu_only_no_training",
        "canonicalizer_decision": arbitration_payload["canonicalizer_decision"],
        "student_build_status": build_payload["status"],
        "receipts": {
            "canonicalizer_arbitration": arbitration.relative_to(ROOT).as_posix(),
            "student_build": build.relative_to(ROOT).as_posix(),
        },
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
    }
    summary_path = RUN_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (arbitration, build, summary_path):
        shutil.copy2(path, receipt_dir / path.name)
    write_status("publishing")
    commit = publish()
    write_status("complete", publish_commit=commit)
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
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
