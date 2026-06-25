import json

from colab.review_stage5_reentry import build_review


def write_assessment(path, *, source_kind, status, recommendation):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kind": "stage5_reentry_assessment",
                "source_kind": source_kind,
                "source_run_id": path.parent.name,
                "status": status,
                "recommendation": recommendation,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_review_recommends_stage2_after_stage1_only(tmp_path) -> None:
    drift = write_assessment(
        tmp_path / "stage5_reentry_drift_x" / "reentry_assessment.json",
        source_kind="reentry_drift_diagnostic",
        status="bridge_dead",
        recommendation="run_reentry_norm_then_repair_smoke",
    )

    review = build_review([drift])

    assert review["latest_stage"] == "stage1_drift"
    assert review["next_target"] == "reentry_norm_diagnostic"


def test_review_recommends_stage3_after_safe_norm(tmp_path) -> None:
    norm = write_assessment(
        tmp_path / "stage5_reentry_norm_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_norm_eval_only",
        status="entry_rms_safe_for_smoke",
        recommendation="run_reentry_repair_smoke",
    )

    review = build_review([norm])

    assert review["latest_stage"] == "stage2_norm"
    assert review["action"] == "run_reentry_repair_smoke"
    assert review["next_target"] == "reentry_repair_smoke"


def test_review_stops_after_norm_regression(tmp_path) -> None:
    norm = write_assessment(
        tmp_path / "stage5_reentry_norm_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_norm_eval_only",
        status="entry_rms_eval_regression",
        recommendation="review_before_trainable_repair",
    )

    review = build_review([norm])

    assert review["action"] == "stop_norm_regression"
    assert review["next_target"] == ""


def test_review_recommends_recovery_after_repair_pass(tmp_path) -> None:
    repair = write_assessment(
        tmp_path / "stage5_reentry_repair_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_repair_smoke",
        status="bridge_repair_smoke_passed",
        recommendation="run_bounded_recovery_training_with_reentry_repair",
    )

    review = build_review([repair])

    assert review["latest_stage"] == "stage3_repair_smoke"
    assert review["action"] == "run_bounded_recovery_training_with_reentry_repair"
    assert review["next_target"] == "reentry_recovery_training"


def test_review_stops_on_stage3_loop1_regression(tmp_path) -> None:
    repair = write_assessment(
        tmp_path / "stage5_reentry_repair_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_repair_smoke",
        status="bridge_live_but_loop1_regressed",
        recommendation="review_or_reduce_repair_lr_before_recovery_training",
    )

    review = build_review([repair])

    assert review["action"] == "stop_loop1_regression"
    assert review["next_target"] == ""


def test_review_stops_on_stage3_missing_loop1_preservation_evidence(tmp_path) -> None:
    repair = write_assessment(
        tmp_path / "stage5_reentry_repair_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_repair_smoke",
        status="loop1_preservation_missing_or_mismatched",
        recommendation="fix_loop1_preservation_eval_before_recovery_training",
    )

    review = build_review([repair])

    assert review["action"] == "stop_loop1_preservation_evidence_missing"
    assert review["next_target"] == ""


def test_review_stops_on_stage3_adapter_not_live(tmp_path) -> None:
    repair = write_assessment(
        tmp_path / "stage5_reentry_repair_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_repair_smoke",
        status="reentry_adapter_not_gradient_live",
        recommendation="fix_reentry_adapter_before_recovery_training",
    )

    review = build_review([repair])

    assert review["action"] == "stop_reentry_adapter_not_live"
    assert review["next_target"] == ""


def test_review_recommends_adapter_smoke_extension_when_live_but_not_moved(tmp_path) -> None:
    repair = write_assessment(
        tmp_path / "stage5_reentry_repair_x" / "reentry_assessment.json",
        source_kind="stage5_reentry_repair_smoke",
        status="reentry_adapter_live_but_not_moved",
        recommendation="extend_reentry_repair_smoke_or_increase_adapter_lr",
    )

    review = build_review([repair])

    assert review["action"] == "extend_reentry_adapter_smoke"
    assert review["next_target"] == "reentry_repair_smoke"
