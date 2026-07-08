from __future__ import annotations

import pytest

from colab.run_stage5_same_reader_final_symbol import identity_check_against_active
from eval.eval_synthetic_depth_final_symbol import choice_label_for_symbol, summarize_final_symbol_rows


def test_choice_label_for_symbol_maps_full_symbol_to_option_label() -> None:
    row = {"choices": {"A": "B", "C": "F", "D": "X"}}

    assert choice_label_for_symbol(row, "F") == "C"
    assert choice_label_for_symbol(row, "Z") is None


def test_summarize_final_symbol_rows_reports_raw_and_mapped_metrics() -> None:
    rows = [
        {"depth": 1, "same_reader_final_hit": True, "mapped_final_hit": True},
        {"depth": 1, "same_reader_final_hit": False, "mapped_final_hit": False},
        {"depth": 2, "same_reader_final_hit": True, "mapped_final_hit": False},
    ]

    summary = summarize_final_symbol_rows(rows, threshold=0.71)

    assert summary["kind"] == "synthetic_depth_same_reader_final_symbol"
    assert summary["by_depth"]["1"]["same_reader_accuracy"] == pytest.approx(0.5)
    assert summary["by_depth"]["2"]["same_reader_accuracy"] == pytest.approx(1.0)
    assert summary["same_reader_total"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["mapped_final_total"]["accuracy"] == pytest.approx(1 / 3)
    assert summary["all_depths_clear_threshold"] is False
    assert summary["metric_policy"]["suspended_reader"].startswith("option-text")


def test_same_reader_identity_check_compares_against_latest_checkpoint_eval() -> None:
    source = {
        "checkpoint_evals": [
            {"step": 2000, "score": {"diagonal_counts": {"1": {"correct": 1, "total": 2, "accuracy": 0.5}}}},
            {"step": 6000, "score": {"diagonal_counts": {"1": {"correct": 2, "total": 2, "accuracy": 1.0}}}},
        ]
    }
    same_reader = {"by_depth": {"1": {"same_reader_accuracy": 1.0}}}

    check = identity_check_against_active(source_payload=source, same_reader=same_reader, tolerance=0.0)

    assert check["pass"] is True
    assert check["deltas"]["1"]["active_accuracy"] == 1.0
