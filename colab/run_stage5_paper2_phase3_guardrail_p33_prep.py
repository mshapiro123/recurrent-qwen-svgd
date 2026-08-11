"""Run the CPU-only three-tier calibration and P3.3 build-only staging."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_guardrail_p33_prep_20260810"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_ROOT / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
P31 = DRIVE_ROOT / "stage5_paper2_phase3_p31_completion_20260810"
P31_REFERENCE_ROWS = P31 / "private/p31_merged_dev_verified_scores.jsonl"
EMPIRICAL = (
    DRIVE_ROOT
    / "stage5_paper2_phase3_empirical_calibration_20260810/receipts/summary.json"
)
P32 = DRIVE_ROOT / "stage5_paper2_phase3_p32_coverage_20260810"
P32_INDEX = P32 / "private/agreement_coverage_index.jsonl"
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
            "kind": "paper2_phase3_guardrail_p33_prep_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            "p33_training_authorized": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"phase3_guardrail_p33_prep_status status={value} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def coverage_has_confidence() -> bool:
    if not P32_INDEX.is_file():
        return False
    with P32_INDEX.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                return {
                    "student_top1_probability",
                    "teacher_14b_top1_probability",
                    "teacher_js_divergence",
                }.issubset(row)
    return False


def main() -> int:
    for path in (P31_REFERENCE_ROWS, EMPIRICAL, CANONICALIZER):
        if not path.exists():
            raise FileNotFoundError(f"missing Phase 3 preparation input: {path}")
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    if not coverage_has_confidence():
        status("refreshing_p32_index_with_rank_confidence_fields")
        run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_p32_coverage.py"])
    if not coverage_has_confidence():
        raise RuntimeError("P3.2 coverage refresh lacks rank-confidence fields")

    superseded_guardrail = RECEIPT_DIR / "guardrail_recalibration.json"
    status("staging_p33_labels_audit_and_observatory")
    p33_dir = PRIVATE_DIR / "p33_prep"
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
    p33_payload = json.loads((p33_dir / "summary.json").read_text(encoding="utf-8"))
    retention_recalibration = {
        "status": "pending_step0_augmented_predictions_before_optimizer_construction",
        "estimator": p33_payload["retention_panel_estimand"],
        "panel_rows": p33_payload["retention_panel_rows"],
        "panel_sha256": p33_payload["retention_panel_sha256"],
        "looks": 20,
        "threshold_reference": "init_relative",
        "requested_sustained_drop_power_points": [-0.5, -1.0, -2.0],
        "tier_s": {
            "familywise_false_stop_max": 0.0001,
            "power_minimum": 0.99,
            "consecutive_looks": 2,
            "delta_cat": "search_before_optimizer_construction",
        },
        "tier_w": {
            "null_warning_rate": "same_class_as_prior_calibration",
            "threshold": "search_before_optimizer_construction",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    final = {
        "kind": "paper2_phase3_guardrail_p33_prep_summary_v1",
        "status": "complete_e2_panel_build_recalibration_pending_training_unauthorized",
        "superseded_task_guardrail": (
            json.loads(superseded_guardrail.read_text(encoding="utf-8"))
            if superseded_guardrail.is_file()
            else None
        ),
        "retention_recalibration": retention_recalibration,
        "p33_prep": p33_payload,
        "assertions": {
            "audit_slice_4096": p33_payload["audit_rows"] == 4096,
            "negative_audit_slice_12288": p33_payload["negative_audit_rows"] == 12288,
            "retention_panel_1024": p33_payload["retention_panel_rows"] == 1024,
            "retention_panel_horizon_balanced": p33_payload[
                "retention_panel_by_horizon"
            ] == {"1": 256, "2": 256, "3": 256, "4": 256},
            "audit_cohorts_disjoint": p33_payload["audit_cohorts_disjoint"],
            "negative_audit_excluded_from_training": p33_payload[
                "negative_audit_excluded_from_training"
            ],
            "retention_panel_excluded_from_training": p33_payload[
                "retention_panel_excluded_from_training"
            ],
            "negative_positive_ratio_3": p33_payload["negative_to_positive_ratio"] == 3.0,
            "position_zero_ignored": set(p33_payload["position_zero_labels"]) <= {"-1"},
            "optimizer_absent": True,
            "training_steps_zero": True,
            "p33_training_unauthorized": True,
        },
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }
    failed = [name for name, passed in final["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 3 prep assertions failed: {failed}")
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
