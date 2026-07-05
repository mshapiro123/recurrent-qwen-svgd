from __future__ import annotations

import pytest

from eval.eval_synthetic_depth_splice import (
    classify_prediction,
    lawful_symbol,
    paired_rows,
    summarize_records,
    symbol_at_orbit,
)


class TinyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True, **_: object) -> dict[str, list[int]]:
        ids = [ord(ch) % 31 for ch in text]
        if add_special_tokens:
            ids = [1, *ids, 2]
        return {"input_ids": ids}


def row(name: str, *, depth: int = 8, question: str = "Q") -> dict[str, object]:
    mapping = {chr(ord("A") + idx): chr(ord("A") + ((idx + 1) % 16)) for idx in range(16)}
    return {
        "id": name,
        "question": question,
        "depth": depth,
        "start": "A",
        "target": "I",
        "orbit": list("ABCDEFGHI"),
        "mapping": mapping,
        "n_symbols": 16,
        "choices": {"A": "I", "B": "J", "C": "K", "D": "L"},
        "answer": "A",
    }


def test_symbol_targets_distinguish_lawful_from_shortcut_after_splice() -> None:
    row_a = row("a")

    spliced_symbol = "D"

    assert lawful_symbol(row_a, spliced_symbol, 1, value_prefix="letter:") == "E"
    assert lawful_symbol(row_a, spliced_symbol, 3, value_prefix="letter:") == "G"
    assert symbol_at_orbit(row_a, 3, value_prefix="letter:") == "D"
    assert symbol_at_orbit(row_a, 5, value_prefix="letter:") == "F"


def test_classify_prediction_prefers_lawful_then_shortcut_then_other() -> None:
    assert classify_prediction("E", "E", "C") == "lawful"
    assert classify_prediction("C", "E", "C") == "shortcut"
    assert classify_prediction("P", "E", "C") == "other"


def test_summarize_records_applies_preregistered_verdict_window() -> None:
    records = []
    for idx in range(8):
        records.append({"k0": 2, "j": 1, "classification": "lawful"})
    for idx in range(2):
        records.append({"k0": 2, "j": 2, "classification": "other"})
    for idx in range(12):
        records.append({"k0": 2, "j": 4, "classification": "shortcut"})

    summary = summarize_records(records, lawful_bar=0.75, shortcut_bar=0.5)

    assert summary["verdict"] == "state_driven"
    assert summary["verdict_counts"] == {"lawful": 8, "shortcut": 0, "other": 2, "n": 10}
    assert summary["overall_counts"]["shortcut"] == 12


def test_summarize_records_detects_prompt_position_shortcut() -> None:
    records = [{"k0": 4, "j": 1, "classification": "shortcut"} for _ in range(6)]
    records.extend({"k0": 4, "j": 2, "classification": "lawful"} for _ in range(4))

    summary = summarize_records(records, lawful_bar=0.75, shortcut_bar=0.5)

    assert summary["verdict"] == "prompt_position_shortcut"
    assert summary["shortcut_fraction_j1_to_j3"] == pytest.approx(0.6)


def test_paired_rows_only_uses_compatible_prompt_lengths_at_target_depth() -> None:
    tokenizer = TinyTokenizer()
    rows = [
        row("a", question="same"),
        row("b", question="same"),
        row("c", question="same"),
        row("d", question="same"),
        row("wrong_depth", depth=6, question="same"),
        row("odd_length", question="different length"),
    ]

    pairs = paired_rows(rows, tokenizer, target_depth=8, n_pairs=2, seed=0)

    assert len(pairs) == 2
    flattened = {item["id"] for pair in pairs for item in pair}
    assert "wrong_depth" not in flattened
    assert "odd_length" not in flattened
