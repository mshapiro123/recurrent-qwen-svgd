import json

from eval.analyze_mcq_regressions import paired_rows, summarize


def test_paired_rows_classifies_wins_losses_and_margin_delta() -> None:
    base = {
        "a": {"id": "a", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.2, "B": -1.0}},
        "b": {"id": "b", "prediction": "A", "answer": "B", "hit": False, "scores": {"A": -0.1, "B": -0.4}},
        "c": {"id": "c", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.5, "B": -0.6}},
    }
    candidate = {
        "a": {"id": "a", "prediction": "B", "answer": "A", "hit": False, "scores": {"A": -0.9, "B": -0.1}},
        "b": {"id": "b", "prediction": "B", "answer": "B", "hit": True, "scores": {"A": -0.7, "B": -0.2}},
        "c": {"id": "c", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.3, "B": -0.8}},
    }
    data = {
        "a": {"question": "Which object has the greatest mass?"},
        "b": {"question": "What is the total distance after 12 miles?"},
        "c": {"question": "Why does the shadow move?"},
    }

    rows = paired_rows(base, candidate, data)
    summary = summarize(rows, benchmark="toy")

    assert [row["change"] for row in rows] == ["loss", "win", "tie_correct"]
    assert summary["base_correct"] == 2
    assert summary["candidate_correct"] == 2
    assert summary["changes"] == {"loss": 1, "win": 1, "tie_correct": 1}
    assert summary["features"]["has_number"]["yes"]["delta"] == 1
    assert summary["features"]["asks_why"]["yes"]["delta"] == 0


def test_summary_is_json_serializable() -> None:
    rows = paired_rows(
        {"x": {"id": "x", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.1, "B": -1.0}}},
        {"x": {"id": "x", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.2, "B": -1.0}}},
        {"x": {"question": "Which result is shown in the table?"}},
    )
    json.dumps(summarize(rows, benchmark="toy"))
