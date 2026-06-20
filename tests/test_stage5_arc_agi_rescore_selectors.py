from __future__ import annotations

import json

from colab.run_stage5_arc_agi_rescore_selectors import (
    CandidatePair,
    best_rows_by_label,
    candidate_label,
    find_candidate_pairs,
    original_row,
    requested_strategies,
)


def test_requested_strategies_defaults_and_validates() -> None:
    assert requested_strategies("heuristic,self_consistency") == ["heuristic", "self_consistency"]

    try:
        requested_strategies("heuristic,bogus")
    except ValueError as exc:
        assert "Unknown selector strategies" in str(exc)
    else:
        raise AssertionError("unknown selector strategy should fail")


def test_candidate_label_strips_candidates_suffix(tmp_path) -> None:
    assert candidate_label(tmp_path / "recovered__tta_all_candidates.jsonl") == "recovered__tta_all"


def test_find_candidate_pairs_matches_summary_by_label(tmp_path) -> None:
    candidate_path = tmp_path / "recovered__tta_all_candidates.jsonl"
    summary_path = tmp_path / "recovered__tta_all_summary.json"
    candidate_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    pairs = find_candidate_pairs(tmp_path)

    assert pairs == [CandidatePair("recovered__tta_all", candidate_path, summary_path)]


def test_original_row_preserves_source_strategy_and_metrics(tmp_path) -> None:
    summary_path = tmp_path / "arm_summary.json"
    payload = {
        "selection_strategy": "self_consistency",
        "summary": {
            "examples_with_targets": 2,
            "selected_exact": 1,
            "best_of_k_exact": 2,
            "first_exact": 0,
            "selected_accuracy": 0.5,
            "best_of_k_accuracy": 1.0,
            "tasks_solved_best_of_k": 1,
            "tasks_with_targets": 1,
            "valid_candidate_rate": 1.0,
        },
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    pair = CandidatePair("arm", tmp_path / "arm_candidates.jsonl", summary_path)

    row = original_row(pair)

    assert row is not None
    assert row["selection_strategy"] == "original:self_consistency"
    assert row["selected_exact"] == 1
    assert row["selected_delta_vs_source"] is None


def test_best_rows_by_label_prefers_selected_then_best_then_valid_rate() -> None:
    rows = [
        {"label": "a", "selection_strategy": "heuristic", "selected_exact": 1, "best_of_k_exact": 2, "valid_candidate_rate": 0.9},
        {
            "label": "a",
            "selection_strategy": "self_consistency",
            "selected_exact": 2,
            "best_of_k_exact": 2,
            "valid_candidate_rate": 0.1,
        },
        {
            "label": "b",
            "selection_strategy": "heuristic",
            "selected_exact": 1,
            "best_of_k_exact": 1,
            "valid_candidate_rate": 0.2,
        },
        {
            "label": "b",
            "selection_strategy": "symbolic_priority",
            "selected_exact": 1,
            "best_of_k_exact": 2,
            "valid_candidate_rate": 0.1,
        },
    ]

    best = best_rows_by_label(rows)

    assert best["a"]["selection_strategy"] == "self_consistency"
    assert best["b"]["selection_strategy"] == "symbolic_priority"
