from __future__ import annotations

import pytest

from eval.eval_synthetic_depth_active_labels import (
    active_target_for_loop,
    candidates_for_row,
    continued_symbol_for_loop,
    parse_int_symbol,
    prompt_for_row,
    single_token_candidate_ids,
    symbol,
    summarize_active_rows,
)


def sample_row() -> dict[str, object]:
    return {
        "id": "row0",
        "question": "Apply f exactly 2 times.",
        "choices": {"A": "3", "B": "5", "C": "7", "D": "9"},
        "answer": "B",
        "target": "5",
        "depth": 2,
        "start": "1",
        "orbit": ["1", "3", "5"],
        "mapping": {"1": "3", "3": "5", "5": "7", "7": "9", "9": "9"},
        "n_symbols": 10,
        "chain_answer_by_loop": {"1": "A", "2": "B"},
    }


def test_active_targets_support_choice_labels_and_full_symbols() -> None:
    row = sample_row()

    assert active_target_for_loop(row, 1, prediction_space="choice_labels", value_prefix="") == "A"
    assert active_target_for_loop(row, 2, prediction_space="choice_labels", value_prefix="") == "B"
    assert active_target_for_loop(row, 3, prediction_space="choice_labels", value_prefix="") is None

    assert active_target_for_loop(row, 1, prediction_space="full_symbols", value_prefix="") == "3"
    assert active_target_for_loop(row, 2, prediction_space="full_symbols", value_prefix="") == "5"
    assert active_target_for_loop(row, 3, prediction_space="full_symbols", value_prefix="") is None


def test_prompt_and_candidates_match_prediction_space() -> None:
    row = sample_row()

    choice_prompt = prompt_for_row(row, prediction_space="choice_labels", prompt_style="with_options")
    symbol_prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")

    assert "A. 3" in choice_prompt
    assert choice_prompt.endswith("Answer:")
    assert "A. 3" not in symbol_prompt
    assert symbol_prompt.endswith("Answer:")
    assert candidates_for_row(row, prediction_space="choice_labels", value_prefix="") == {
        "A": " A",
        "B": " B",
        "C": " C",
        "D": " D",
    }
    assert candidates_for_row(row, prediction_space="full_symbols", value_prefix="")["7"] == " 7"
    assert len(candidates_for_row(row, prediction_space="full_symbols", value_prefix="")) == 10


def test_single_token_candidate_ids_detects_prompt_suffix_tokens() -> None:
    class FakeTokenizer:
        vocab = {"Question\nAnswer:": [10, 11], " A": [101], " B": [102], " multi": [201, 202]}

        def __call__(self, text: str, *, add_special_tokens: bool = True):
            if text == "Question\nAnswer:":
                return {"input_ids": [10, 11]}
            if text == "Question\nAnswer: A":
                return {"input_ids": [10, 11, 101]}
            if text == "Question\nAnswer: B":
                return {"input_ids": [10, 11, 102]}
            if text == "Question\nAnswer: multi":
                return {"input_ids": [10, 11, 201, 202]}
            raise AssertionError(text)

    prompt = "Question\nAnswer:"
    assert single_token_candidate_ids(FakeTokenizer(), prompt, {"A": " A", "B": " B"}) == {"A": 101, "B": 102}
    assert single_token_candidate_ids(FakeTokenizer(), prompt, {"bad": " multi"}) is None


def test_letter_value_prefix_maps_full_symbol_space() -> None:
    row = sample_row() | {
        "start": "B",
        "orbit": ["B", "D", "F"],
        "mapping": {"B": "D", "D": "F", "F": "H"},
        "n_symbols": 16,
    }

    assert symbol(0, prefix="letter:") == "A"
    assert symbol("C", prefix="letter:") == "C"
    assert parse_int_symbol("P", prefix="letter:") == 15
    assert candidates_for_row(row, prediction_space="full_symbols", value_prefix="letter:")["P"] == " P"
    assert active_target_for_loop(row, 2, prediction_space="full_symbols", value_prefix="letter:") == "F"
    assert continued_symbol_for_loop(row, 3, value_prefix="letter:") == "H"


def test_name_value_prefix_maps_full_symbol_space() -> None:
    row = sample_row() | {
        "start": "Sam",
        "orbit": ["Sam", "Max", "Lee"],
        "mapping": {"Sam": "Max", "Max": "Lee", "Lee": "Ana"},
        "n_symbols": 20,
    }

    assert symbol(0, prefix="name:") == "Ben"
    assert symbol("Ada", prefix="name:") == "Ada"
    assert parse_int_symbol("Jon", prefix="name:") == 19
    candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
    assert candidates["Ben"] == " Ben"
    assert candidates["Jon"] == " Jon"
    assert active_target_for_loop(row, 2, prediction_space="full_symbols", value_prefix="name:") == "Lee"
    assert continued_symbol_for_loop(row, 3, value_prefix="name:") == "Ana"


def test_continued_symbol_uses_serialized_mapping_for_above_diagonal_behavior() -> None:
    row = sample_row()

    assert continued_symbol_for_loop(row, 1, value_prefix="") == "3"
    assert continued_symbol_for_loop(row, 2, value_prefix="") == "5"
    assert continued_symbol_for_loop(row, 3, value_prefix="") == "7"


def test_summarize_active_rows_separates_active_cells_from_above_diagonal() -> None:
    rows = [
        {"depth": 1, "forced_loop_count": 1, "active_cell": True, "hit": True},
        {"depth": 1, "forced_loop_count": 1, "active_cell": True, "hit": False},
        {"depth": 2, "forced_loop_count": 1, "active_cell": True, "hit": True},
        {"depth": 2, "forced_loop_count": 2, "active_cell": True, "hit": True},
        {
            "depth": 1,
            "forced_loop_count": 2,
            "active_cell": False,
            "hit": False,
            "above_diagonal_behavior": "iterate",
        },
        {
            "depth": 1,
            "forced_loop_count": 3,
            "active_cell": False,
            "hit": False,
            "above_diagonal_behavior": "hold",
        },
    ]

    summary = summarize_active_rows(rows, threshold=0.71)

    assert summary["active_matrix"]["1"]["1"]["accuracy"] == pytest.approx(0.5)
    assert summary["active_matrix"]["2"]["1"]["accuracy"] == pytest.approx(1.0)
    assert summary["active_diagonal"] == {"1": 0.5, "2": 1.0}
    assert summary["active_diagonal_clears_bar"] is False
    assert summary["active_total"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert summary["above_diagonal"]["n"] == 2
    assert summary["above_diagonal"]["rates"]["iterate"] == pytest.approx(0.5)
    assert summary["above_diagonal"]["rates"]["hold"] == pytest.approx(0.5)
