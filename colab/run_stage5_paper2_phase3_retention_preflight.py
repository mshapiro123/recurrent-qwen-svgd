"""Build and calibrate the exact P3.3 token-retention guardrail, without training."""

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
RUN_ID = "stage5_paper2_phase3_retention_preflight_20260811"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
P32_ID = "stage5_paper2_phase3_p32_coverage_20260810"
MIGRATION_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
EMPIRICAL_ID = "stage5_paper2_phase3_empirical_calibration_20260810"
OLD_SUMMARY = ROOT / "outputs/stage5" / OLD_ID / "summary.json"
OLD_PRIVATE = DRIVE_ROOT / OLD_ID / "private/stage0a"
NEW_SUMMARY = DRIVE_ROOT / NEW_ID / "receipts/full_cache_summary.json"
NEW_PRIVATE = DRIVE_ROOT / NEW_ID / "private/full"
P32_INDEX = DRIVE_ROOT / P32_ID / "private/agreement_coverage_index.jsonl"
MIGRATED = DRIVE_ROOT / MIGRATION_ID / "private/migrated_checkpoints"
PRIOR_EMPIRICAL = DRIVE_ROOT / EMPIRICAL_ID / "receipts/summary.json"
CANONICALIZER = (
    DRIVE_ROOT
    / "stage5_paper2_phase2_arbitration_build_20260804/private/canonicalizer/"
    "learned_mixture_rrr_seed_20260814.pt"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(value: str, **details: object) -> None:
    write_json(
        RECEIPT_DIR / "status.json",
        {
            "kind": "paper2_phase3_retention_preflight_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            "task_level_capability_scoring": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"p33_retention_preflight status={value} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def scratch_root() -> Path:
    for candidate in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if not candidate.exists():
            continue
        free_gib = shutil.disk_usage(candidate).free / (1024**3)
        print(f"p33_retention_scratch_candidate path={candidate} free_gib={free_gib:.1f}", flush=True)
        if free_gib >= 50.0:
            return candidate / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.3 retention preflight needs 50 GiB of local scratch")


def rsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_arg = str(source) + os.sep if source.is_dir() else str(source)
    destination_arg = str(destination) + os.sep if source.is_dir() else str(destination)
    run(
        [
            "rsync",
            "--archive",
            "--delete",
            "--partial",
            "--info=progress2",
            source_arg,
            destination_arg,
        ]
    )


def stage_sources(scratch: Path) -> dict[str, Path]:
    staged_old = scratch / "old"
    staged_new = scratch / "new"
    staged_checkpoints = scratch / "migrated_checkpoints"
    for source, destination in (
        (OLD_PRIVATE / "sample_manifest.jsonl", staged_old / "sample_manifest.jsonl"),
        (OLD_PRIVATE / "model_cache/student_0p5b/", staged_old / "model_cache/student_0p5b/"),
        (NEW_PRIVATE / "sample_manifest.jsonl", staged_new / "sample_manifest.jsonl"),
        (NEW_PRIVATE / "model_cache/student_0p5b/", staged_new / "model_cache/student_0p5b/"),
        (
            MIGRATED / "seed_0_full_a2_phase3_migrated.pt",
            staged_checkpoints / "seed_0_full_a2_phase3_migrated.pt",
        ),
        (
            MIGRATED / "seed_1_full_a2_phase3_migrated.pt",
            staged_checkpoints / "seed_1_full_a2_phase3_migrated.pt",
        ),
    ):
        rsync(source, destination)
    return {
        "old_private": staged_old,
        "new_private": staged_new,
        "checkpoints": staged_checkpoints,
    }


def main() -> int:
    required = [
        OLD_SUMMARY,
        OLD_PRIVATE,
        NEW_SUMMARY,
        NEW_PRIVATE,
        P32_INDEX,
        PRIOR_EMPIRICAL,
        CANONICALIZER,
        MIGRATED / "seed_0_full_a2_phase3_migrated.pt",
        MIGRATED / "seed_1_full_a2_phase3_migrated.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing P3.3 retention-preflight source: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    p33_dir = PRIVATE_DIR / "p33_prep"
    status("drawing_exact_retention_panel")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase3_p33",
            "--coverage_index",
            str(P32_INDEX),
            "--output_dir",
            str(p33_dir),
            "--canonical_projection",
            str(CANONICALIZER),
        ]
    )
    p33 = json.loads((p33_dir / "summary.json").read_text(encoding="utf-8"))
    if p33["retention_panel_rows"] != 1024:
        raise RuntimeError("P3.3 retention panel is not exactly 1,024 positions")

    status("staging_score_blind_sources")
    scratch = scratch_root()
    scratch.mkdir(parents=True, exist_ok=True)
    staged = stage_sources(scratch)
    step0_rows = PRIVATE_DIR / "p33_retention_step0_rows.jsonl"
    step0_summary = RECEIPT_DIR / "p33_retention_step0_summary.json"
    status("scoring_step0_token_retention_no_update")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_retention_step0",
            "--retention_panel",
            str(p33_dir / "p33_retention_panel.jsonl"),
            "--old_summary",
            str(OLD_SUMMARY),
            "--old_private",
            str(staged["old_private"]),
            "--new_summary",
            str(NEW_SUMMARY),
            "--new_private",
            str(staged["new_private"]),
            "--migrated_checkpoint",
            str(staged["checkpoints"] / "seed_0_full_a2_phase3_migrated.pt"),
            "--migrated_checkpoint",
            str(staged["checkpoints"] / "seed_1_full_a2_phase3_migrated.pt"),
            "--output_rows",
            str(step0_rows),
            "--output_summary",
            str(step0_summary),
            "--device",
            "cuda",
            "--batch_size",
            "64",
        ]
    )

    calibration_path = RECEIPT_DIR / "p33_retention_guardrail_recalibration.json"
    status("calibrating_init_relative_retention_rules")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_retention_guardrail",
            "--step0_rows",
            str(step0_rows),
            "--prior_empirical_summary",
            str(PRIOR_EMPIRICAL),
            "--panel_sha256",
            str(p33["retention_panel_sha256"]),
            "--output_summary",
            str(calibration_path),
            "--campaigns",
            "100000",
            "--looks",
            "20",
        ]
    )
    step0 = json.loads(step0_summary.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    final = {
        "kind": "paper2_phase3_p33_retention_preflight_summary_v1",
        "status": "complete_e2_assertions_final_training_may_follow_in_separate_target",
        "p33_prep": p33,
        "step0_retention": step0,
        "retention_guardrail": calibration,
        "assertions": {
            "panel_1024": p33["retention_panel_rows"] == 1024,
            "panel_horizon_balanced": p33["retention_panel_by_horizon"]
            == {"1": 256, "2": 256, "3": 256, "4": 256},
            "all_cohorts_disjoint": p33["audit_cohorts_disjoint"],
            "retention_excluded_from_training": p33[
                "retention_panel_excluded_from_training"
            ],
            "step0_exact_estimator_complete": all(step0["assertions"].values()),
            "calibration_complete": all(calibration["assertions"].values()),
            "twenty_looks": calibration["looks"] == 20,
            "task_scoring_absent": True,
            "optimizer_absent": True,
            "training_steps_zero": True,
        },
        "task_level_capability_scoring": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    failed = [name for name, passed in final["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.3 retention preflight assertions failed: {failed}")
    write_json(RECEIPT_DIR / "summary.json", final)
    status("complete", summary=str(RECEIPT_DIR / "summary.json"))
    print(json.dumps({"status": final["status"], "drive": str(DRIVE_RUN)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
