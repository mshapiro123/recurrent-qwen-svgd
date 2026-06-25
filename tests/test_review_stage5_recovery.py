import json
from pathlib import Path

from colab.review_stage5_recovery import build_review


def write_summary(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_pointer(path: Path, target: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(target), encoding="utf-8")
    return path


def recovery_payload(**overrides):
    payload = {
        "kind": "stage5_reentry_recovery_training",
        "run_id": "stage5_reentry_recovery_test",
        "checkpoint": "outputs/stage5/stage5_reentry_recovery_test/phase1/phase1_step_75.pt",
        "validation_checks": {"status": "validation_sane", "issues": []},
    }
    payload.update(overrides)
    return payload


def test_recovery_review_waits_when_no_stage4_summary(tmp_path: Path) -> None:
    review = build_review(tmp_path / "outputs" / "stage5", pointer=tmp_path / "config" / "stage5_current_source_summary.txt")

    assert review["action"] == "wait_for_reentry_recovery_training"
    assert review["next_target"] == ""
    assert review["launch_env"] == {}


def test_recovery_review_blocks_missing_checkpoint(tmp_path: Path) -> None:
    summary = write_summary(
        tmp_path / "outputs" / "stage5" / "stage4" / "summary.json",
        recovery_payload(checkpoint="", phase1_checkpoint=""),
    )

    review = build_review(tmp_path / "outputs" / "stage5", pointer=write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", summary))

    assert review["action"] == "stop_recovery_checkpoint_missing"
    assert review["next_target"] == ""
    assert review["launch_env"] == {}


def test_recovery_review_blocks_validation_issues(tmp_path: Path) -> None:
    summary = write_summary(
        tmp_path / "outputs" / "stage5" / "stage4" / "summary.json",
        recovery_payload(validation_checks={"status": "validation_needs_review", "issues": ["target_loop_gradient_not_observed"]}),
    )

    review = build_review(tmp_path / "outputs" / "stage5", pointer=write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", summary))

    assert review["action"] == "stop_recovery_validation_needs_review"
    assert review["next_target"] == ""
    assert review["validation_issues"] == ["target_loop_gradient_not_observed"]


def test_recovery_review_routes_sane_recovery_to_benchmark_with_source_override(tmp_path: Path) -> None:
    summary = write_summary(
        tmp_path / "outputs" / "stage5" / "stage4" / "summary.json",
        recovery_payload(),
    )

    review = build_review(tmp_path / "outputs" / "stage5", pointer=write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", summary))

    assert review["action"] == "run_debiased_benchmark_suite"
    assert review["next_target"] == "debiased_benchmark_suite"
    assert review["launch_env"]["STAGE5_CURRENT_A100_TARGET"] == "debiased_benchmark_suite"
    assert review["launch_env"]["STAGE5_CURRENT_A100_SOURCE_SUMMARY"].endswith("outputs/stage5/stage4/summary.json")
    assert review["current_pointer"]["preferred"] is True
