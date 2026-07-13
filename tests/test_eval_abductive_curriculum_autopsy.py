from __future__ import annotations

from eval.eval_abductive_curriculum_autopsy import (
    classify_abductive_prediction,
    continued_reverse_target,
    reverse_target_for_loop,
    summarize_prediction_rows,
)


def sample_row() -> dict:
    return {
        "id": "test_injective_d03_00000",
        "depth": 3,
        "n_symbols": 5,
        "observed_target": "E",
        "selected_start": "A",
        "selected_orbit": ["A", "B", "C", "E"],
        "loop_completions": [" C", " B", " A"],
        "mapping": {"A": "B", "B": "C", "C": "E", "E": "D", "D": "A"},
        "valid_starts": ["A"],
    }


def test_reverse_targets_and_above_diagonal_continuation() -> None:
    row = sample_row()

    assert reverse_target_for_loop(row, 1) == "C"
    assert reverse_target_for_loop(row, 3) == "A"
    assert reverse_target_for_loop(row, 4) is None
    assert continued_reverse_target(row, 4) == "D"
    assert continued_reverse_target(row, 5) == "E"


def test_prediction_confusion_categories_are_mutually_exclusive() -> None:
    row = sample_row()

    assert classify_abductive_prediction(row, "A") == "correct_start"
    assert classify_abductive_prediction(row, "C") == "one_step_preimage"
    assert classify_abductive_prediction(row, "B") == "other_orbit_intermediate"
    assert classify_abductive_prediction(row, "E") == "observed_target"
    assert classify_abductive_prediction(row, "D") == "other_valid_name"
    assert classify_abductive_prediction(row, "not-a-name") == "junk"


def test_autopsy_summary_separates_active_and_above_diagonal_cells() -> None:
    rows = [
        {
            "split": "test",
            "depth": 2,
            "loop": 1,
            "active": True,
            "hit": True,
            "prediction": "B",
            "target": "B",
            "above_diagonal_behavior": None,
            "diagonal_confusion": None,
        },
        {
            "split": "test",
            "depth": 2,
            "loop": 2,
            "active": True,
            "hit": False,
            "prediction": "C",
            "target": "A",
            "above_diagonal_behavior": None,
            "diagonal_confusion": "other_valid_name",
        },
        {
            "split": "test",
            "depth": 2,
            "loop": 3,
            "active": False,
            "hit": False,
            "prediction": "D",
            "target": None,
            "above_diagonal_behavior": "iterate",
            "diagonal_confusion": None,
        },
    ]

    summary = summarize_prediction_rows(rows, split="test")

    assert summary["active_matrix"]["2"]["1"]["accuracy"] == 1.0
    assert summary["active_matrix"]["2"]["2"]["accuracy"] == 0.0
    assert summary["loop1_by_depth"]["2"]["accuracy"] == 1.0
    assert summary["diagonal_confusion"] == {"other_valid_name": 1}
    assert summary["above_diagonal"]["iterate"] == 1
