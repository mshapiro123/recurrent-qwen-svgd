from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_a_surpass_receipt import (
    build_compute_ledger,
    paired_binary_test,
    row_hash_receipt,
    score_phase_a_rows,
)


def test_paired_binary_test_reports_direction_and_exact_probabilities() -> None:
    result = paired_binary_test(
        [True, True, True, False, True, False],
        [False, False, True, True, False, False],
    )

    assert result["helped"] == 3
    assert result["hurt"] == 1
    assert result["tied"] == 2
    assert 0.0 <= result["one_sided_p"] <= result["two_sided_p"] <= 1.0


def test_score_phase_a_rows_joins_identical_ids_and_separates_primary_from_extension() -> None:
    a_rows = [
        {"id": "d1-a", "depth": 1, "same_reader_final_hit": True},
        {"id": "d1-b", "depth": 1, "same_reader_final_hit": True},
        {"id": "d2-a", "depth": 2, "same_reader_final_hit": True},
        {"id": "d2-b", "depth": 2, "same_reader_final_hit": False},
    ]
    dense_rows = [
        {
            "id": "d1-a",
            "depth": 1,
            "correct": {"B_step4000": False, "C_step4000": True},
        },
        {
            "id": "d1-b",
            "depth": 1,
            "correct": {"B_step4000": False, "C_step4000": False},
        },
        {
            "id": "d2-a",
            "depth": 2,
            "correct": {"B_step4000": False, "C_step4000": False},
        },
        {
            "id": "d2-b",
            "depth": 2,
            "correct": {"B_step4000": False, "C_step4000": False},
        },
    ]

    result = score_phase_a_rows(a_rows, dense_rows, rows_per_depth=2)

    assert result["rows"] == 4
    assert result["comparisons"]["A_vs_B_step4000"]["role"] == "preregistered_primary"
    assert result["comparisons"]["A_vs_C_step4000"]["role"] == "analysis_extension"
    assert result["comparisons"]["A_vs_B_step4000"]["paired"]["helped"] == 3


def test_phase_a_receipt_target_is_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_PHASE_A_SURPASS_RECEIPT_CELL.py").read_text(encoding="utf-8")

    assert '"phase_a_surpass_receipt"' in bootstrap
    assert "STAGE5_PHASE_A_SURPASS_RECEIPT_CELL_VERSION" in cell
    assert "accepted_returncodes={0, 2}" in cell


class _WhitespaceTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def test_compute_ledger_separates_latent_transitions_from_text_decode() -> None:
    a_rows = [{"id": "d1", "depth": 1}, {"id": "d2", "depth": 2}]
    dense = {
        label: [
            {"id": "d1", "depth": 1, "continuation": "one two"},
            {"id": "d2", "depth": 2, "continuation": "one two three"},
        ]
        for label in ("B_step4000", "C_step4000", "D_step4000")
    }
    tokenizer = _WhitespaceTokenizer()

    ledger = build_compute_ledger(
        a_rows,
        dense,
        tokenizers={label: tokenizer for label in dense},
    )

    assert ledger["A"]["by_depth"]["2"]["latent_recurrent_transitions"] == 2
    assert ledger["A"]["by_depth"]["2"]["text_context_growth_tokens"] == 0
    assert ledger["dense"]["C_step4000"]["by_depth"]["2"]["sequential_decode_rounds_mean"] == 3


def test_row_hash_receipt_is_order_invariant() -> None:
    rows = [{"id": "b", "depth": 2}, {"id": "a", "depth": 1}]

    assert row_hash_receipt(rows) == row_hash_receipt(list(reversed(rows)))
