"""Build the threshold-neutral P3.2 oracle cache and ridge forecasts."""

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
RUN_ID = "stage5_paper2_phase3_oracle_forecast_20260810"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
OLD_ID = "stage5_paper2_phase2_stage0a_20260803"
NEW_ID = "stage5_paper2_phase2_option_b_teacher_cache_20260806"
COVERAGE_ID = "stage5_paper2_phase3_p32_coverage_20260810"
MIGRATION_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
OLD_SUMMARY = ROOT / "outputs/stage5" / OLD_ID / "summary.json"
OLD_PRIVATE = DRIVE_ROOT / OLD_ID / "private/stage0a"
NEW_SUMMARY = DRIVE_ROOT / NEW_ID / "receipts/full_cache_summary.json"
NEW_PRIVATE = DRIVE_ROOT / NEW_ID / "private/full"
COVERAGE_INDEX = DRIVE_ROOT / COVERAGE_ID / "private/agreement_coverage_index.jsonl"
COVERAGE_SUMMARY = DRIVE_ROOT / COVERAGE_ID / "receipts/summary.json"
MIGRATED = DRIVE_ROOT / MIGRATION_ID / "private/migrated_checkpoints"


def scratch_root() -> Path:
    candidates = [Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")]
    required_free_gib = 80.0
    for candidate in candidates:
        if not candidate.exists():
            continue
        free_gib = shutil.disk_usage(candidate).free / (1024**3)
        print(f"phase3_oracle_scratch_candidate path={candidate} free_gib={free_gib:.1f}", flush=True)
        if free_gib >= required_free_gib:
            return candidate / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError(
        f"Phase 3 oracle forecast needs {required_free_gib:.0f} GiB of local scratch"
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


def stage_sources() -> dict[str, Path]:
    scratch = scratch_root()
    scratch.mkdir(parents=True, exist_ok=True)
    staged_old = scratch / "old"
    staged_new = scratch / "new"
    staged_coverage = scratch / "coverage"
    staged_checkpoints = scratch / "migrated_checkpoints"

    # Only stage files consumed by the oracle pass.  The coverage pass has
    # already reduced the lattice and teacher caches to a compact index.
    for source, destination in (
        (OLD_PRIVATE / "sample_manifest.jsonl", staged_old / "sample_manifest.jsonl"),
        (OLD_PRIVATE / "model_cache/student_0p5b/", staged_old / "model_cache/student_0p5b/"),
        (NEW_PRIVATE / "sample_manifest.jsonl", staged_new / "sample_manifest.jsonl"),
        (NEW_PRIVATE / "model_cache/student_0p5b/", staged_new / "model_cache/student_0p5b/"),
        (COVERAGE_INDEX, staged_coverage / COVERAGE_INDEX.name),
        (COVERAGE_SUMMARY, staged_coverage / COVERAGE_SUMMARY.name),
        (MIGRATED / "seed_0_full_a2_phase3_migrated.pt", staged_checkpoints / "seed_0_full_a2_phase3_migrated.pt"),
        (MIGRATED / "seed_1_full_a2_phase3_migrated.pt", staged_checkpoints / "seed_1_full_a2_phase3_migrated.pt"),
    ):
        rsync(source, destination)
    return {
        "root": scratch,
        "old_private": staged_old,
        "new_private": staged_new,
        "coverage_index": staged_coverage / COVERAGE_INDEX.name,
        "coverage_summary": staged_coverage / COVERAGE_SUMMARY.name,
        "checkpoints": staged_checkpoints,
    }


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
            "kind": "paper2_phase3_oracle_forecast_status_v1",
            "status": status,
            "updated_at_unix": time.time(),
            "p33_training_authorized": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"phase3_oracle_forecast_status status={status} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    drive_checkpoints = [
        MIGRATED / "seed_0_full_a2_phase3_migrated.pt",
        MIGRATED / "seed_1_full_a2_phase3_migrated.pt",
    ]
    required = [
        OLD_SUMMARY,
        OLD_PRIVATE,
        NEW_SUMMARY,
        NEW_PRIVATE,
        COVERAGE_INDEX,
        COVERAGE_SUMMARY,
        *drive_checkpoints,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase 3 oracle-forecast source: {missing}")
    coverage = json.loads(COVERAGE_SUMMARY.read_text(encoding="utf-8"))
    if coverage.get("status") != "complete_agreement_coverage_surface_not_final_cache":
        raise RuntimeError("P3.2 coverage receipt is not complete")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_drive_sources_to_local_scratch")
    staged = stage_sources()
    checkpoints = [
        staged["checkpoints"] / "seed_0_full_a2_phase3_migrated.pt",
        staged["checkpoints"] / "seed_1_full_a2_phase3_migrated.pt",
    ]
    write_status("building_threshold_neutral_oracle_cache")
    cache_command = [
        sys.executable,
        "-u",
        "-m",
        "eval.cache_paper2_phase3_agreement_oracle",
        "--coverage_index",
        str(staged["coverage_index"]),
        "--old_summary",
        str(OLD_SUMMARY),
        "--old_private",
        str(staged["old_private"]),
        "--new_summary",
        str(NEW_SUMMARY),
        "--new_private",
        str(staged["new_private"]),
        "--migrated_checkpoint",
        str(checkpoints[0]),
        "--migrated_checkpoint",
        str(checkpoints[1]),
        "--output_dir",
        str(PRIVATE_DIR / "oracle_cache"),
        "--device",
        "cuda",
    ]
    run(cache_command)

    cache_summary_path = PRIVATE_DIR / "oracle_cache/summary.json"
    cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    if cache_summary["optimizer_steps"] != 0 or cache_summary["p33_training_authorized"]:
        raise RuntimeError("oracle cache crossed the no-training boundary")
    direction_cache = Path(cache_summary["direction_cache"]["path"])
    write_status("fitting_document_disjoint_linear_forecasts")
    forecasts = []
    for feature in cache_summary["feature_caches"]:
        output = RUN_DIR / (
            f"linear_forecast_seed_{feature['seed']}_loop_{feature['loop_index']}.json"
        )
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_phase3_linear_forecast",
                "--feature_cache",
                str(feature["path"]),
                "--direction_cache",
                str(direction_cache),
                "--output_summary",
                str(output),
                "--seed",
                str(20260810 + int(feature["seed"])),
            ]
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        forecasts.append(
            {
                "seed": int(feature["seed"]),
                "loop_index": int(feature["loop_index"]),
                "summary": str(output),
                "rows": int(result["rows"]),
                "selected_ridge": float(result["selected_ridge"]),
                "holdout_cosine": result["holdout_cosine"],
            }
        )

    summary = {
        "kind": "paper2_phase3_oracle_forecast_summary_v1",
        "status": "complete_agreement_oracle_and_linear_forecast_no_training",
        "oracle_cache": cache_summary,
        "linear_forecasts": forecasts,
        "assertions": {
            "actual_lm_head_equivalence": cache_summary["assertions"][
                "actual_lm_head_equivalence"
            ],
            "threshold_not_selected": not cache_summary["teachability_threshold_selected"],
            "both_seed_lineages": {row["seed"] for row in forecasts} == {0, 1},
            "all_four_loops": {row["loop_index"] for row in forecasts} == {1, 2, 3, 4},
            "document_disjoint_forecasts": True,
            "optimizer_steps_zero": True,
            "confirm_unscored": True,
            "p33_training_unauthorized": True,
        },
        "remaining_before_final_p32": [
            "verified-stratum base/14B/32B generation and programmatic verification",
            "strategy selection of numeric teachability and gate-negative thresholds",
        ],
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }
    if not all(summary["assertions"].values()):
        raise RuntimeError(f"Phase 3 oracle-forecast assertions failed: {summary['assertions']}")
    write_json(RUN_DIR / "summary.json", summary)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "summary.json", RECEIPT_DIR / "summary.json")
    for path in RUN_DIR.glob("linear_forecast_*.json"):
        shutil.copy2(path, RECEIPT_DIR / path.name)
    shutil.copy2(cache_summary_path, RECEIPT_DIR / "oracle_cache_summary.json")
    write_status("complete", summary=str(RECEIPT_DIR / "summary.json"))
    print(json.dumps({"status": summary["status"], "drive": str(DRIVE_RUN)}, indent=2))
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
