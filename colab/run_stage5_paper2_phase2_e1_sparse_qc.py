"""Run and publish score-blind sparse-support QC for frozen E1 EVAL-D."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_e1_eval_d_20260808"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
SOURCE_SUMMARY = RUN_DIR / "cache/e1_eval_d_lattice_summary.json"
FREEZE_RECEIPT = RUN_DIR / "receipts/e1_eval_d_freeze_summary.json"
QC_RECEIPT = RUN_DIR / "receipts/e1_sparse_support_qc.json"
READINESS_V2 = RUN_DIR / "receipts/e1_readiness_v2.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
PRIVATE_DIR = DRIVE_ROOT / RUN_ID / "private/e1_eval_d"
DRIVE_RECEIPTS = DRIVE_ROOT / RUN_ID / "receipts"


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    if not SOURCE_SUMMARY.is_file():
        raise FileNotFoundError(f"missing frozen E1 public cache summary: {SOURCE_SUMMARY}")
    if not FREEZE_RECEIPT.is_file():
        raise FileNotFoundError(f"missing frozen E1 receipt: {FREEZE_RECEIPT}")
    if not PRIVATE_DIR.is_dir():
        raise FileNotFoundError(f"missing frozen E1 private cache: {PRIVATE_DIR}")

    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.audit_paper2_phase2_e1_sparse_support",
            "--source_summary",
            str(SOURCE_SUMMARY),
            "--private_dir",
            str(PRIVATE_DIR),
            "--output_summary",
            str(QC_RECEIPT),
        ]
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.check_paper2_phase2_e1_readiness",
            "--eval_d_freeze",
            str(FREEZE_RECEIPT),
            "--sparse_support_qc",
            str(QC_RECEIPT),
            "--output",
            str(READINESS_V2),
        ]
    )
    qc = json.loads(QC_RECEIPT.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_V2.read_text(encoding="utf-8"))
    if qc.get("read_once_scoring_spent") is not False:
        raise RuntimeError("sparse-support QC spent the E1 read-once evaluation")
    if readiness.get("ready_to_lock") is not True or readiness.get("blockers") != []:
        raise RuntimeError(f"E1 remains blocked after sparse QC: {readiness.get('blockers')}")

    DRIVE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    for path in (QC_RECEIPT, READINESS_V2):
        shutil.copy2(path, DRIVE_RECEIPTS / path.name)

    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(
        [
            "git",
            "add",
            "-f",
            "--",
            QC_RECEIPT.relative_to(ROOT).as_posix(),
            READINESS_V2.relative_to(ROOT).as_posix(),
        ]
    )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record score-blind E1 sparse-support QC [skip ci]"])
        run(["git", "push", "origin", "main"])
    print("E1 sparse-support QC landed; read-once scoring remains unspent.", flush=True)


if __name__ == "__main__":
    main()
