"""Repair and publish Stage 0A derived metrics from cached Drive shards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_stage0a_repair_20260804"
SOURCE_RUN_ID = "stage5_paper2_phase2_stage0a_20260803"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
SOURCE_SUMMARY = ROOT / "outputs/stage5" / SOURCE_RUN_ID / "summary.json"
DRIVE_SOURCE = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{SOURCE_RUN_ID}"
)
DRIVE_RUN = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}"
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 Stage 0A metric repair [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    # Safety marker: DEV-C CPU-only cached lattice repair no model inference no training
    if not SOURCE_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing landed Stage 0A summary: {SOURCE_SUMMARY}")
    private = DRIVE_SOURCE / "private/stage0a"
    if not private.is_dir():
        raise FileNotFoundError(f"Missing durable Stage 0A private cache: {private}")
    output = RUN_DIR / "summary.json"
    repaired_private = DRIVE_RUN / "private/repaired_metrics_v2"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.repair_paper2_phase2_stage0a",
            "--private_dir",
            str(private),
            "--original_summary",
            str(SOURCE_SUMMARY),
            "--repaired_private_dir",
            str(repaired_private),
            "--output_summary",
            str(output),
        ]
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    if summary.get("training_started") or summary.get("optimizer_steps"):
        raise RuntimeError("Stage 0A repair violated its no-training contract")
    if summary.get("frozen_evaluation_partitions_touched"):
        raise RuntimeError("Stage 0A repair touched a frozen evaluation partition")
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "stage0a_repair_summary.json")
    commit = publish(output)
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

