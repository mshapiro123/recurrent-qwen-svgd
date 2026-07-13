from __future__ import annotations

from pathlib import Path

from colab.run_stage5_inverse_composition_staircase import (
    EXPECTED_TEST_SHA,
    EXPECTED_TRAIN_SHA,
    assess_stage_gate,
    build_matched_data,
    classify_matched_arms,
    latest_checkpoint_stage,
    stage_training_plan,
)


def test_matched_data_regenerates_locked_rows_and_changes_only_rendering(tmp_path) -> None:
    receipt = build_matched_data(tmp_path)

    assert receipt["canonical"]["train"]["row_sha256"] == EXPECTED_TRAIN_SHA
    assert receipt["canonical"]["test"]["row_sha256"] == EXPECTED_TEST_SHA
    assert receipt["matched_identity"]["status"] == "passed"
    assert receipt["matched_identity"]["non_rendering_mismatches"] == []


def test_stage_plan_uses_effective_batch_eight_and_equalized_newest_dose() -> None:
    rows = [{"depth": depth} for depth in range(1, 5) for _ in range(16)]
    plan = stage_training_plan(
        rows,
        cap=4,
        effective_batch_size=8,
        weighted_label_budget=1500.0,
        eval_every=250,
        remaining_phase_steps=4000,
    )

    assert plan["effective_batch_size"] == 8
    assert plan["optimizer_steps"] % 250 == 0
    assert plan["optimizer_steps"] <= 4000
    assert plan["expected_newest_weighted_labels"] <= 1500.0


def test_stage_gate_requires_46_of_64_on_the_newest_diagonal() -> None:
    passed = {"test": {"diagonal_by_depth": {"4": {"correct": 46, "total": 64, "accuracy": 46 / 64}}}}
    failed = {"test": {"diagonal_by_depth": {"4": {"correct": 45, "total": 64, "accuracy": 45 / 64}}}}

    assert assess_stage_gate(passed, cap=4)["passed"] is True
    assert assess_stage_gate(failed, cap=4)["passed"] is False


def test_matched_arm_reading_uses_preregistered_ratio_quadrants() -> None:
    assert classify_matched_arms(experiment_dose=500.0, control_dose=50.0) == "non_native_position_cost"
    assert classify_matched_arms(experiment_dose=100.0, control_dose=100.0) == "exposure_starvation"
    assert classify_matched_arms(experiment_dose=1400.0, control_dose=1300.0) == "composition_hard_both"
    assert classify_matched_arms(experiment_dose=None, control_dose=None) == "composition_hard_both"
    assert classify_matched_arms(experiment_dose=50.0, control_dose=None) == "instrumentation_alarm"


def test_bootstrap_exposes_single_resumable_staircase_target() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_INVERSE_COMPOSITION_STAIRCASE_CELL.py").read_text(encoding="utf-8")

    assert '"inverse_composition_staircase"' in bootstrap
    assert "STAGE5_INVERSE_COMPOSITION_STAIRCASE_CELL_VERSION" in cell
    assert "weighted_per_loop_labels" in cell
    assert "accepted_returncodes={0, 2}" in cell


def test_resume_uses_latest_real_checkpoint_not_envelope_stop() -> None:
    durable = {
        "cap": 3,
        "checkpoint_drive_backup": "/drive/cap3.pt",
        "checkpoint_sha256": "abc",
    }
    envelope_stop = {
        "cap": 4,
        "status": "phase_step_envelope_exhausted",
        "checkpoint_drive_backup": None,
    }

    assert latest_checkpoint_stage([durable, envelope_stop]) is durable
    assert latest_checkpoint_stage([envelope_stop]) is None
