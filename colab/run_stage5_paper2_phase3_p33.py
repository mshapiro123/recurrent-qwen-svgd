"""Stage and run one resumable, lock-bound P3.3 seed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p33_20260811"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
PREFLIGHT_ID = "stage5_paper2_phase3_retention_preflight_20260811"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
MIGRATION_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
ORACLE_ID = "stage5_paper2_phase3_oracle_forecast_20260810"
CANONICALIZER_ID = "stage5_paper2_phase2_arbitration_build_20260804"


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
    run([
        "rsync", "--archive", "--delete", "--partial", "--info=progress2",
        str(source) + (os.sep if source.is_dir() else ""),
        str(destination) + (os.sep if source.is_dir() else ""),
    ])


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.3 requires at least 80 GiB local scratch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    seed = args.seed
    run_dir = ROOT / "outputs/stage5" / RUN_ID / f"seed_{seed}"
    drive_run = DRIVE_STAGE5 / RUN_ID
    private = drive_run / "private" / f"seed_{seed}"
    receipts = drive_run / "receipts" / f"seed_{seed}"
    status_path = receipts / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(status_path, {
            "kind": "paper2_phase3_p33_colab_status_v1", "seed": seed,
            "status": value, "updated_at_unix": time.time(), **details,
        })
        print(f"p33_status seed={seed} status={value} details={details}", flush=True)

    try:
        status("staging")
        scratch = scratch_root()
        old = scratch / "old"
        new = scratch / "new"
        preflight = DRIVE_STAGE5 / PREFLIGHT_ID
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/sample_manifest.jsonl", old / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/model_cache/student_0p5b", old / "model_cache/student_0p5b")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/sample_manifest.jsonl", new / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/model_cache/student_0p5b", new / "model_cache/student_0p5b")
        canonicalizer = scratch / "canonicalizer.pt"
        direction_cache = scratch / "agreement_oracle_directions.pt"
        migrated = scratch / f"seed_{seed}_migrated.pt"
        rsync(
            DRIVE_STAGE5 / CANONICALIZER_ID / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt",
            canonicalizer,
        )
        rsync(
            DRIVE_STAGE5 / ORACLE_ID / "private/oracle_cache/agreement_oracle_directions.pt",
            direction_cache,
        )
        rsync(
            DRIVE_STAGE5 / MIGRATION_ID / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
            migrated,
        )
        status("training")
        command = [
            sys.executable, "-u", "-m", "training.run_paper2_phase3_p33",
            "--seed", str(seed),
            "--old_summary", str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
            "--old_private", str(old),
            "--new_summary", str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
            "--new_private", str(new),
            "--canonicalizer", str(canonicalizer),
            "--old_cache", str(scratch / "old_cache.pt"),
            "--new_cache", str(scratch / "new_cache.pt"),
            "--staged_labels", str(preflight / "private/p33_prep/p33_staged_labels.jsonl"),
            "--positive_audit", str(preflight / "private/p33_prep/p33_audit_slice.jsonl"),
            "--negative_audit", str(preflight / "private/p33_prep/p33_negative_audit_slice.jsonl"),
            "--retention_panel", str(preflight / "private/p33_prep/p33_retention_panel.jsonl"),
            "--preflight_summary", str(preflight / "receipts/summary.json"),
            "--guardrail_calibration", str(preflight / "receipts/p33_retention_guardrail_recalibration.json"),
            "--direction_cache", str(direction_cache),
            "--migrated_checkpoint", str(migrated),
            "--output_dir", str(run_dir),
            "--private_dir", str(private),
            "--device", "cuda",
        ]
        run(command)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        write_json(receipts / "summary.json", summary)
        status("complete", step=summary["step"], run_status=summary["status"])
        return 0
    except Exception as error:
        status("failed", exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
