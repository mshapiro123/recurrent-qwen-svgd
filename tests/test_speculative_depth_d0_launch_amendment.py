from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from eval.eval_speculative_depth_router_feasibility import summarize_teacher_demand
from training.speculative_depth_d0_spec import (
    deterministic_argmax_fp32,
    registered_depth_target,
    severity_quartile,
    summarize_registered_target_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def test_graded_target_uses_frozen_quartile_table_not_descriptive_fit() -> None:
    floor = {
        "calibration": {
            "branch": "graded_floor_curve",
            "targets": {"q1": 2, "q2": 2, "q3": 2, "q4": 2},
        },
        "severity_quartile_boundaries": [0.1, 0.2, 0.3],
    }

    assert registered_depth_target(floor=floor, accepted=True, kl=99.0) == 1
    assert [
        registered_depth_target(floor=floor, accepted=False, kl=value)
        for value in (0.05, 0.15, 0.25, 0.35)
    ] == [2, 2, 2, 2]


def test_quartile_boundary_equality_stays_in_lower_bin() -> None:
    boundaries = [0.1, 0.2, 0.3]
    assert severity_quartile(0.1, boundaries) == "q1"
    assert severity_quartile(0.100001, boundaries) == "q2"
    assert severity_quartile(0.2, boundaries) == "q2"
    assert severity_quartile(0.3, boundaries) == "q3"
    assert severity_quartile(0.300001, boundaries) == "q4"


def test_target_policy_receipt_proves_binary_depth_one_two_schedule() -> None:
    floor = {
        "calibration": {
            "branch": "graded_floor_curve",
            "targets": {"q1": 2, "q2": 2, "q3": 2, "q4": 2},
        },
        "severity_quartile_boundaries": [0.1, 0.2, 0.3],
    }
    receipt = summarize_registered_target_schedule(
        floor=floor,
        scheduled_positions=[
            {"accepted": True, "kl": 0.0},
            {"accepted": False, "kl": 0.05},
            {"accepted": False, "kl": 0.35},
        ],
    )

    assert receipt["status"] == "verified_before_training"
    assert receipt["target_depth_counts"] == {"1": 1, "2": 2}
    assert receipt["rejected_severity_counts"] == {"q1": 1, "q2": 0, "q3": 0, "q4": 1}
    assert receipt["descriptive_mapping_used_for_targets"] is False


def test_deterministic_argmax_casts_fp32_selects_lowest_id_and_flags_ties() -> None:
    logits = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [0.0, 1.0, 2.0]],
        dtype=torch.bfloat16,
    )

    selected, tied = deterministic_argmax_fp32(logits, dim=-1)

    assert selected.tolist() == [0, 1, 2]
    assert tied.tolist() == [True, True, False]


def test_teacher_demand_uses_each_teachers_own_loop1_rejections() -> None:
    rows = [
        {
            "predictions": [1, 2, 3],
            "teacher_7b": 2,
            "teacher_14b": 1,
        },
        {
            "predictions": [4, 5, 6],
            "teacher_7b": 5,
            "teacher_14b": 6,
        },
        {
            "predictions": [7, 8, 9],
            "teacher_7b": 9,
            "teacher_14b": 8,
        },
        {
            "predictions": [10, 11, 12],
            "teacher_7b": 99,
            "teacher_14b": 99,
        },
    ]

    summary = summarize_teacher_demand(rows)

    seven = summary["teacher_7b_own_rejections"]
    fourteen = summary["teacher_14b_own_rejections"]
    overlap = summary["teacher_overlap_on_7b_rejections"]
    assert seven["positions"] == 4
    assert seven["first_correct_depth_counts"] == {"1": 0, "2": 2, "3": 1}
    assert seven["median_first_correct_depth_recoverable"] == 2.0
    assert fourteen["positions"] == 3
    assert fourteen["first_correct_depth_counts"] == {"1": 0, "2": 1, "3": 1}
    assert fourteen["median_first_correct_depth_recoverable"] == 2.5
    assert overlap["fourteen_endorses_loop1"] == 1
    assert overlap["share"] == 0.25


def test_drive_addendum_copy_is_byte_authenticated() -> None:
    addendum = ROOT / "docs/STRATEGY_ADDENDUM_D0_FIGURE_REVIEW_20260727.md"
    payload = addendum.read_bytes()

    assert len(payload) == 2795
    assert hashlib.sha256(payload).hexdigest() == (
        "ff93c5011872e91d64dcdc380169b93beedd3097b6cee2987722d28af06a36b8"
    )


def test_training_path_uses_registered_bins_and_prelaunch_receipts() -> None:
    trainer = (ROOT / "training/run_speculative_depth_d0.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_paper2_d0_train_eval.py").read_text(encoding="utf-8")

    assert "predict_isotonic" not in trainer
    assert trainer.count("registered_depth_target(") >= 2
    assert 'parser.add_argument("--target_policy_receipt", required=True)' in trainer
    assert 'parser.add_argument("--prelaunch_summary", required=True)' in trainer
    assert '"--target_policy_receipt"' in runner
    assert '"--prelaunch_summary"' in runner
    assert "target_policy_receipt_sha256" in trainer
    assert '"assert_ok": "registered_d0_target_policy"' in trainer
    assert '"scheduled_target_depth_counts"' in trainer


def test_trained_floor_records_all_positions_for_teacher_specific_demand() -> None:
    floor = (ROOT / "eval/eval_speculative_depth_d0_floor.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_paper2_d0_train_eval.py").read_text(encoding="utf-8")

    assert '"all_position_rows": all_positions' in floor
    assert 'payload.get("all_position_rows")' in runner
    assert "legacy dual-teacher union is not an admissible substitute" in runner
    assert "summarize_teacher_demand(rows)" in runner
