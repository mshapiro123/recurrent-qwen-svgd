from __future__ import annotations

from colab.reentry_recovery_config import (
    int_dict_max_key,
    mode_rows_from_counts,
    repair_assessment_recovery_block_reason,
    target_loop_rows_from_counts,
)


def test_target_loop_rows_preserves_counts_and_sorts_numerically() -> None:
    assert target_loop_rows_from_counts({"3": "8", "1": 24, "2": 16}) == "1=24,2=16,3=8"


def test_target_loop_rows_skips_invalid_and_nonpositive_entries() -> None:
    counts = {
        "1": 12,
        "0": 99,
        "-1": 99,
        "2": 0,
        "3": -4,
        "4": "bad",
        "five": 5,
        "6": "7",
    }

    assert target_loop_rows_from_counts(counts) == "1=12,6=7"


def test_target_loop_rows_blocks_fake_presence_only_ladder() -> None:
    counts = {"1": 48, "2": 16, "4": 8}

    assert target_loop_rows_from_counts(counts) != "1=1,2=1,4=1"
    assert target_loop_rows_from_counts(counts) == "1=48,2=16,4=8"


def test_mode_rows_preserves_mode_counts() -> None:
    assert mode_rows_from_counts({"wide": "4", "direct": 12, "deep_narrow": 8}) == (
        "deep_narrow=8,direct=12,wide=4"
    )


def test_int_dict_max_key_ignores_invalid_and_nonpositive_keys() -> None:
    assert int_dict_max_key({"0": 100, "-1": 100, "2": 1, "4": 1, "bad": 99}, default=3) == 4
    assert int_dict_max_key({"bad": 99, "0": 100}, default=3) == 3


def passing_repair_assessment() -> dict:
    return {
        "recommendation": "run_bounded_recovery_training_with_reentry_repair",
        "status": "bridge_repair_smoke_passed",
        "metrics": {
            "train_metrics_available": True,
            "train_loss": 1.25,
            "depth_supervision_metrics_present": True,
            "loop1_preservation_available": True,
            "loop1_regressed": False,
            "bridge_live": True,
            "bridge_moved": True,
            "use_reentry_adapter": True,
            "adapter_live": True,
            "adapter_moved": True,
        },
    }


def test_repair_assessment_recovery_gate_accepts_current_smoke_evidence() -> None:
    assert repair_assessment_recovery_block_reason(passing_repair_assessment()) is None


def test_repair_assessment_recovery_gate_rejects_stale_recommendation_only_artifact() -> None:
    assessment = {
        "recommendation": "run_bounded_recovery_training_with_reentry_repair",
        "status": "bridge_repair_smoke_passed",
    }

    reason = repair_assessment_recovery_block_reason(assessment)

    assert reason is not None
    assert "missing metrics" in reason


def test_repair_assessment_recovery_gate_rejects_missing_train_metrics() -> None:
    assessment = passing_repair_assessment()
    assessment["metrics"]["train_metrics_available"] = False

    reason = repair_assessment_recovery_block_reason(assessment)

    assert reason is not None
    assert "final training metrics" in reason


def test_repair_assessment_recovery_gate_rejects_nonfinite_train_loss() -> None:
    assessment = passing_repair_assessment()
    assessment["metrics"]["train_loss"] = float("nan")

    reason = repair_assessment_recovery_block_reason(assessment)

    assert reason is not None
    assert "train_loss" in reason


def test_repair_assessment_recovery_gate_rejects_missing_depth_metrics() -> None:
    assessment = passing_repair_assessment()
    assessment["metrics"]["depth_supervision_metrics_present"] = False

    reason = repair_assessment_recovery_block_reason(assessment)

    assert reason is not None
    assert "depth metrics" in reason


def test_repair_assessment_recovery_gate_rejects_unmoved_adapter_when_enabled() -> None:
    assessment = passing_repair_assessment()
    assessment["metrics"]["adapter_moved"] = False

    reason = repair_assessment_recovery_block_reason(assessment)

    assert reason is not None
    assert "adapter moved" in reason
