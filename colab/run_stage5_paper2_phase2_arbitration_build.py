"""Run and publish CPU-only Phase-2 canonicalizer arbitration and student build."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 arbitration and build receipts [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    # Safety marker: CPU high RAM cached canonicalizer arbitration and loss-free student build
    if not EXP0A_SUMMARY.is_file():
        raise FileNotFoundError(EXP0A_SUMMARY)
    if not STAGE0A_PRIVATE.is_dir():
        raise FileNotFoundError(STAGE0A_PRIVATE)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    private = DRIVE_RUN / "private"
    arbitration = RUN_DIR / "canonicalizer_arbitration_summary.json"
    build = RUN_DIR / "student_build_summary.json"
    # Fail fast on the checkpoint-integrated identity battery before the six
    # expensive cached SVD fits, and preserve its receipt immediately.
    run(
        [
            sys.executable,
            "-m",
            "eval.eval_paper2_dc2_student_build",
            "--output_summary",
            str(build),
            "--model_name",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]
    )
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build, receipt_dir / build.name)
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
    commit = publish()
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
