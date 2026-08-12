"""Stage and run the read-only P3.3 zero-collateral verification."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p33_verification_20260812"
P33_ID = "stage5_paper2_phase3_p33_20260811"
PREFLIGHT_ID = "stage5_paper2_phase3_retention_preflight_20260811"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
ORACLE_ID = "stage5_paper2_phase3_oracle_forecast_20260810"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def rsync(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "--archive",
            "--delete",
            "--partial",
            "--info=progress2",
            str(source) + (os.sep if source.is_dir() else ""),
            str(destination) + (os.sep if source.is_dir() else ""),
        ]
    )


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 50 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.3 verification requires at least 50 GiB local scratch")


def main() -> int:
    run_dir = ROOT / "outputs/stage5" / RUN_ID
    drive_run = DRIVE_STAGE5 / RUN_ID
    receipts = drive_run / "receipts"
    private = drive_run / "private"
    status_path = receipts / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_phase3_p33_verification_colab_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                **details,
            },
        )
        print(f"p33_verification_status status={value} details={details}", flush=True)

    try:
        status("staging")
        scratch = scratch_root()
        old = scratch / "old"
        new = scratch / "new"
        preflight = DRIVE_STAGE5 / PREFLIGHT_ID / "private/p33_prep"
        rsync(
            DRIVE_STAGE5 / OLD_ID / "private/stage0a/sample_manifest.jsonl",
            old / "sample_manifest.jsonl",
        )
        rsync(
            DRIVE_STAGE5 / OLD_ID / "private/stage0a/model_cache/student_0p5b",
            old / "model_cache/student_0p5b",
        )
        rsync(
            DRIVE_STAGE5 / NEW_ID / "private/full/sample_manifest.jsonl",
            new / "sample_manifest.jsonl",
        )
        rsync(
            DRIVE_STAGE5 / NEW_ID / "private/full/model_cache/student_0p5b",
            new / "model_cache/student_0p5b",
        )
        direction_cache = scratch / "agreement_oracle_directions.pt"
        rsync(
            DRIVE_STAGE5 / ORACLE_ID / "private/oracle_cache/agreement_oracle_directions.pt",
            direction_cache,
        )
        checkpoints = []
        source_checkpoints = []
        for seed in (0, 1):
            destination = scratch / f"seed_{seed}_p33_step_1000.pt"
            rsync(
                DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt",
                destination,
            )
            checkpoints.append(destination)
            source_destination = scratch / f"seed_{seed}_phase3_migrated.pt"
            rsync(
                DRIVE_STAGE5
                / "stage5_paper2_phase3_p31_p32_receipts_20260810"
                / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
                source_destination,
            )
            source_checkpoints.append(source_destination)
        status("evaluating")
        command = [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p33_verification",
            "--old_summary",
            str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
            "--old_private",
            str(old),
            "--new_summary",
            str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
            "--new_private",
            str(new),
            "--positive_audit",
            str(preflight / "p33_audit_slice.jsonl"),
            "--negative_audit",
            str(preflight / "p33_negative_audit_slice.jsonl"),
            "--retention_panel",
            str(preflight / "p33_retention_panel.jsonl"),
            "--direction_cache",
            str(direction_cache),
            "--checkpoint",
            str(checkpoints[0]),
            "--checkpoint",
            str(checkpoints[1]),
            "--source_checkpoint",
            str(source_checkpoints[0]),
            "--source_checkpoint",
            str(source_checkpoints[1]),
            "--output_dir",
            str(run_dir),
            "--device",
            "cuda",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("P3.3 verification exited without a summary receipt")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for source in run_dir.glob("*.json"):
            shutil.copy2(source, receipts / source.name)
        for source in run_dir.glob("*_rows.jsonl"):
            shutil.copy2(source, private / source.name)
        status(
            "complete" if completed.returncode == 0 else "failed_positive_control",
            evaluation_status=summary["status"],
            exit_code=completed.returncode,
        )
        return completed.returncode
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
