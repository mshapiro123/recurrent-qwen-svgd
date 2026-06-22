from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def arc_mix_payload(*, decision: str, status: str = "proxy_lift", passed: bool = True) -> dict:
    return {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": status,
        "passed": passed,
        "decision": decision,
        "blocked_reason": None if decision == "run_full_balanced_assessment" else "proxy did not pass",
        "next_step": "review",
        "best_arm": {
            "arm": "arc_mix_response_w01_lr2e6",
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/run/phase1/phase1_step_50.pt",
                "arc": {"mean": {"correct": 68, "total": 128}},
                "comparison_to_base": {
                    "mean_margin_delta": 0.02,
                    "max_abs_prediction_count_delta": 4,
                    "calibration_ok": True,
                },
            },
            "phase1_start": {"arc": {"mean": {"correct": 66, "total": 128}}},
            "base_arc": {"mean": {"correct": 68, "total": 128}},
        },
        "arms": [],
    }


def test_arc_mix_result_review_allows_full_assessment_for_clean_decision(tmp_path: Path) -> None:
    import colab.review_stage5_arc_mix_result as module

    source = tmp_path / "summary.json"
    payload = arc_mix_payload(decision="run_full_balanced_assessment")
    source.write_text(json.dumps(payload), encoding="utf-8")

    review = module.build_review(payload, source_summary=source)
    markdown = module.render_markdown(review)

    assert review["full_assessment_justified"] is True
    assert review["best_arm"]["lift_vs_start"] == 2
    assert review["best_arm"]["gap_vs_base"] == 0
    assert "run_stage5_recovery_full_assessment.py" in review["recommended_action"]["command"]
    assert "YES: run exactly one full balanced ARC confirmation" in markdown


def test_arc_mix_result_review_follows_planner_for_legacy_summary_without_decision(
    tmp_path: Path,
) -> None:
    import colab.review_stage5_arc_mix_result as module

    source = tmp_path / "summary.json"
    payload = arc_mix_payload(decision="")
    source.write_text(json.dumps(payload), encoding="utf-8")

    review = module.build_review(payload, source_summary=source)
    markdown = module.render_markdown(review)

    assert review["full_assessment_justified"] is True
    assert review["decision_basis"] == "legacy_planner_action"
    assert "run_stage5_recovery_full_assessment.py" in review["recommended_action"]["command"]
    assert "Legacy summary lacks explicit decision/calibration fields" in markdown


@pytest.mark.parametrize(
    ("decision", "status"),
    [
        ("stop_for_calibration_repair", "proxy_lift_calibration_warning"),
        ("stop_and_revise_objective", "no_proxy_lift"),
    ],
)
def test_arc_mix_result_review_blocks_non_clean_decisions(
    tmp_path: Path, decision: str, status: str
) -> None:
    import colab.review_stage5_arc_mix_result as module

    source = tmp_path / "summary.json"
    payload = arc_mix_payload(decision=decision, status=status, passed=False)
    source.write_text(json.dumps(payload), encoding="utf-8")

    review = module.build_review(payload, source_summary=source)
    markdown = module.render_markdown(review)

    assert review["full_assessment_justified"] is False
    assert review["decision"] == decision
    assert "run_stage5_recovery_full_assessment.py" not in review["recommended_action"]["command"]
    assert "NO: stop A100 work and repair locally" in markdown


def test_arc_mix_result_review_finds_latest_summary(tmp_path: Path, monkeypatch) -> None:
    import colab.review_stage5_arc_mix_result as module

    old = tmp_path / "outputs" / "stage5" / "old" / "summary.json"
    new = tmp_path / "outputs" / "stage5" / "new" / "summary.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text(json.dumps(arc_mix_payload(decision="stop_and_revise_objective")), encoding="utf-8")
    new.write_text(json.dumps(arc_mix_payload(decision="run_full_balanced_assessment")), encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.latest_arc_mix_summary() == new
