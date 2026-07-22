from __future__ import annotations

from eval.dense_response_reader import extract_first_completed_symbol


def test_direct_response_stops_at_leading_symbol() -> None:
    candidates = list("ABCD")
    text = " C\n\nStart value: A\nApply f exactly 1 times.\nAnswer: D"

    assert extract_first_completed_symbol(text, candidates) == "C"


def test_scratchpad_response_stops_at_first_answer_marker() -> None:
    candidates = list("ABCD")
    text = " steps: B -> C answer: C answer: D answer: A"

    assert extract_first_completed_symbol(text, candidates) == "C"


def test_reader_retains_bounded_fallback() -> None:
    candidates = list("ABCD")

    assert extract_first_completed_symbol("This is a guess: C", candidates) == "C"
    assert extract_first_completed_symbol("no valid symbol", candidates) is None
