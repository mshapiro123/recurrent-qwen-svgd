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
        "a": {"question": "Which object is red?"},
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
    assert summary["prediction_count_deltas"]["candidate_minus_base"] == {"A": -2, "B": 2}
    assert summary["prediction_count_deltas"]["candidate_minus_answer"] == {"A": -1, "B": 1}
    assert summary["prediction_count_deltas"]["max_abs_candidate_minus_base"] == 2
    assert summary["prediction_transition_counts"]["flat"] == {"A->A": 1, "A->B": 2}
    assert summary["prediction_transition_counts"]["changed_predictions"] == 2
    assert rows[0]["routing_bucket"] == "ambiguous_proxy"
    assert rows[1]["routing_bucket"] == "deep_numeric_proxy"
    assert rows[2]["routing_bucket"] == "conceptual_reasoning_proxy"


def test_summary_is_json_serializable() -> None:
    rows = paired_rows(
        {"x": {"id": "x", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.1, "B": -1.0}}},
        {"x": {"id": "x", "prediction": "A", "answer": "A", "hit": True, "scores": {"A": -0.2, "B": -1.0}}},
        {"x": {"question": "Which result is shown in the table?"}},
    )
    json.dumps(summarize(rows, benchmark="toy"))


def test_routing_summary_uses_base_confidence_and_loop_diagnostics() -> None:
    rows = paired_rows(
        {
            "x": {
                "id": "x",
                "prediction": "A",
                "answer": "A",
                "hit": True,
                "scores": {"A": 0.0, "B": -2.0},
            }
        },
        {
            "x": {
                "id": "x",
                "prediction": "B",
                "answer": "A",
                "hit": False,
                "scores": {"A": -1.0, "B": -0.5},
                "loop_diagnostics": {
                    "mean_expected_loops": 2.8,
                    "answer_expected_loops": 2.7,
                    "prediction_expected_loops": 2.9,
                },
            }
        },
        {"x": {"question": "Which material is magnetic?"}},
    )

    summary = summarize(rows, benchmark="toy")
    bucket = summary["routing_buckets"]["base_confident_direct_proxy"]

    assert rows[0]["routing_bucket"] == "base_confident_direct_proxy"
    assert bucket["n"] == 1
    assert bucket["delta"] == -1
    assert bucket["mean_candidate_expected_loops"] == 2.8
    assert bucket["mean_candidate_answer_expected_loops"] == 2.7
