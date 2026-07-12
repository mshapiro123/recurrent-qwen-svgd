from __future__ import annotations

from eval.eval_synthetic_diagonal_guardrail import summarize_rows


def test_diagonal_guardrail_summary_reports_per_depth_minimum() -> None:
    rows = [
        {"depth": 1, "hit": True},
        {"depth": 1, "hit": True},
        {"depth": 2, "hit": True},
        {"depth": 2, "hit": False},
    ]

    summary = summarize_rows(rows)

    assert summary["active_diagonal"] == {"1": 1.0, "2": 0.5}
    assert summary["active_diagonal_min"] == 0.5
    assert summary["accuracy"] == 0.75

