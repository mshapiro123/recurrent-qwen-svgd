from __future__ import annotations

from pathlib import Path

from eval import assess_regression_battery as assess
from eval import prepare_arc_mcq
from colab import run_stage5_regression_battery as runner


def paired_payload(*, wins: int, losses: int, ties: int = 900) -> dict:
    n = wins + losses + ties
    base_correct = losses + (ties // 2)
    recurrent_correct = wins + (ties // 2)
    row = {
        "paired_examples": n,
        "base_correct": base_correct,
        "recurrent_correct": recurrent_correct,
        "base_accuracy": base_correct / n,
        "recurrent_accuracy": recurrent_correct / n,
        "accuracy_delta_recurrent_vs_base": (recurrent_correct - base_correct) / n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
    return {
        "status": "completed",
        "source_summary": "outputs/stage5/source/summary.json",
        "checkpoint": "outputs/stage5/source/checkpoint.pt",
        "recurrent_forced_loop_count": 1,
        "paired_comparisons": {
            "arc_easy": {
                "content_question_only": {"mean": row},
                "cyclic_label_aggregated": {"permutation_mean": row},
            },
            "arc_challenge": {
                "content_question_only": {"mean": row},
                "cyclic_label_aggregated": {"permutation_mean": row},
            },
        },
    }


def assessment_status(*, wins: int, losses: int, ties: int = 900) -> str:
    payload = paired_payload(wins=wins, losses=losses, ties=ties)
    out = assess.build_assessment(
        suite_summary=Path(__file__),
        suite_payload=payload,
        required_benchmarks=["arc_easy", "arc_challenge"],
        target_specs=[("content_question_only", "mean")],
        margin=0.03,
        yellow_margin=0.015,
        run_id="test",
    )
    return out["status"]


def test_regression_battery_green_when_confidence_excludes_margin() -> None:
    assert assessment_status(wins=25, losses=20, ties=1955) == "green_noninferior"


def test_regression_battery_yellow_when_point_delta_crosses_half_margin() -> None:
    assert assessment_status(wins=0, losses=20, ties=980) == "yellow_drift_watch"


def test_regression_battery_red_when_upper_ci_below_margin() -> None:
    assert assessment_status(wins=0, losses=220, ties=1780) == "red_regression_established"


def test_regression_battery_reports_pending_non_arc_extensions() -> None:
    payload = paired_payload(wins=25, losses=20, ties=1955)
    out = assess.build_assessment(
        suite_summary=Path(__file__),
        suite_payload=payload,
        required_benchmarks=["arc_easy"],
        target_specs=[("cyclic_label_aggregated", "permutation_mean")],
        margin=0.03,
        yellow_margin=0.015,
        run_id="test",
    )

    assert "hellaswag_500" in out["pending_extensions"]
    assert out["rows"][0]["aggregate"] == "permutation_mean"


def test_prepare_arc_mcq_split_all_loads_train_validation_and_test(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_dataset(_dataset_id, _config, *, split):
        calls.append(split)
        return [
            {
                "id": "shared",
                "question": f"Question {split}",
                "choices": {"text": ["yes", "no"], "label": ["A", "B"]},
                "answerKey": "A",
            }
        ]

    monkeypatch.setattr(prepare_arc_mcq, "load_dataset", fake_load_dataset)

    rows = prepare_arc_mcq.load_split_rows("allenai/ai2_arc", "ARC-Easy", "all")

    assert calls == ["train", "validation", "test"]
    assert [row["id"] for row in rows] == ["train:shared", "validation:shared", "test:shared"]


def test_regression_runner_combined_status_uses_worst_assessment() -> None:
    assert runner.combined_status([{"status": "green_noninferior"}]) == "green_noninferior"
    assert (
        runner.combined_status([{"status": "green_noninferior"}, {"status": "yellow_drift_watch"}])
        == "yellow_drift_watch"
    )
    assert (
        runner.combined_status([{"status": "green_noninferior"}, {"status": "red_regression_established"}])
        == "red_regression_established"
    )


def test_regression_runner_can_resume_existing_and_force_add_outputs() -> None:
    text = Path("colab/run_stage5_regression_battery.py").read_text(encoding="utf-8")

    assert "STAGE5_REGRESSION_RESUME_EXISTING" in text
    assert "resume_existing_benchmark_suite" in text
    assert '["git", "add", "-f"' in text
