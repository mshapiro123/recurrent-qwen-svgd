"""Run and publish DEV-only Experiment 0A canonicalizer screening."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_exp0a_20260804"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
REPAIR_ID = "stage5_paper2_phase2_stage0a_repair_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
REPAIR_SUMMARY = ROOT / "outputs/stage5" / REPAIR_ID / "summary.json"
DRIVE_STAGE0A = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{STAGE0A_ID}"
)
DRIVE_RUN = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}"
)
DRIVE_REPAIR_SUMMARY = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{REPAIR_ID}/"
    "receipts/stage0a_repair_summary.json"
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 Experiment 0A [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    # Safety marker: DEV-C only canonicalizer and whitening screening no backbone training
    if not STAGE0A_SUMMARY.is_file():
        raise FileNotFoundError(f"Experiment 0A prerequisite is missing: {STAGE0A_SUMMARY}")
    repair_summary = REPAIR_SUMMARY if REPAIR_SUMMARY.is_file() else DRIVE_REPAIR_SUMMARY
    if not repair_summary.is_file():
        raise FileNotFoundError(
            f"Experiment 0A repair receipt is missing from GitHub and Drive: {repair_summary}"
        )
    private = DRIVE_STAGE0A / "private/stage0a"
    output = RUN_DIR / "summary.json"
    output_private = DRIVE_RUN / "private/exp0a"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_exp0a",
            "--stage0a_private",
            str(private),
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--repaired_summary",
            str(repair_summary),
            "--output_private",
            str(output_private),
            "--output_summary",
            str(output),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    if summary.get("training_started") or summary.get("optimizer_steps"):
        raise RuntimeError("Experiment 0A unexpectedly trained model parameters")
    if summary.get("frozen_evaluation_partitions_touched"):
        raise RuntimeError("Experiment 0A touched a frozen evaluation partition")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "exp0a_summary.json")
    commit = publish(output)
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
