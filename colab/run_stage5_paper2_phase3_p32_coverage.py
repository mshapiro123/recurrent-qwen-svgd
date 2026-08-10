"""Run the score-blind P3.2 agreement-stratum coverage pass."""

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
RUN_ID = "stage5_paper2_phase3_p32_coverage_20260810"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
OLD_SUMMARY = ROOT / "outputs/stage5" / OLD_ID / "summary.json"
OLD_PRIVATE = DRIVE_ROOT / OLD_ID / "private/stage0a"
NEW_SUMMARY = DRIVE_ROOT / NEW_ID / "receipts/full_cache_summary.json"
NEW_PRIVATE = DRIVE_ROOT / NEW_ID / "private/full"


def scratch_root() -> Path:
    candidates = [Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")]
    required_free_gib = 100.0
    for candidate in candidates:
        if not candidate.exists():
            continue
        free_gib = shutil.disk_usage(candidate).free / (1024**3)
        print(f"phase3_coverage_scratch_candidate path={candidate} free_gib={free_gib:.1f}", flush=True)
        if free_gib >= required_free_gib:
            return candidate / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError(
        f"Phase 3 coverage needs {required_free_gib:.0f} GiB of local scratch"
    )


def rsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_arg = str(source) + os.sep if source.is_dir() else str(source)
    destination_arg = str(destination) + os.sep if source.is_dir() else str(destination)
    command = [
        "rsync",
        "--archive",
        "--delete",
        "--partial",
        "--info=progress2",
        source_arg,
        destination_arg,
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stage_sources() -> tuple[Path, Path]:
    scratch = scratch_root()
    staged_old = scratch / "old"
    staged_new = scratch / "new"
    for source, destination in (
        (OLD_PRIVATE / "sample_manifest.jsonl", staged_old / "sample_manifest.jsonl"),
        (OLD_PRIVATE / "lattice/", staged_old / "lattice/"),
        (OLD_PRIVATE / "model_cache/teacher_14b/", staged_old / "model_cache/teacher_14b/"),
        (NEW_PRIVATE / "sample_manifest.jsonl", staged_new / "sample_manifest.jsonl"),
        (NEW_PRIVATE / "lattice/", staged_new / "lattice/"),
        (NEW_PRIVATE / "model_cache/teacher_14b/", staged_new / "model_cache/teacher_14b/"),
    ):
        rsync(source, destination)
    return staged_old, staged_new


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_status(status: str, **details: object) -> None:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        RECEIPT_DIR / "status.json",
        {
            "kind": "paper2_phase3_p32_coverage_status_v1",
            "status": status,
            "updated_at_unix": time.time(),
            "p33_training_authorized": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"phase3_p32_coverage_status status={status} details={details}", flush=True)


def main() -> int:
    required = [OLD_SUMMARY, OLD_PRIVATE, NEW_SUMMARY, NEW_PRIVATE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing P3.2 coverage source: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_drive_sources_to_local_scratch")
    staged_old, staged_new = stage_sources()
    write_status("reading_cached_agreement_lattice")
    output = RUN_DIR / "summary.json"
    index = PRIVATE_DIR / "agreement_coverage_index.jsonl"
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.eval_paper2_phase3_p32_coverage",
        "--old_summary",
        str(OLD_SUMMARY),
        "--old_private",
        str(staged_old),
        "--new_summary",
        str(NEW_SUMMARY),
        "--new_private",
        str(staged_new),
        "--output_index",
        str(index),
        "--output_summary",
        str(output),
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["optimizer_steps"] != 0 or result["p33_training_authorized"]:
        raise RuntimeError("P3.2 coverage crossed the no-training boundary")
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, RECEIPT_DIR / output.name)
    write_status("complete", summary=str(RECEIPT_DIR / output.name), index=str(index))
    print(json.dumps({"status": result["status"], "drive": str(DRIVE_RUN)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
