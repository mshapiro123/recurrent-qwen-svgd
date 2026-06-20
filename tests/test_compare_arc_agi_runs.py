from __future__ import annotations

from eval.compare_arc_agi_runs import (
    compare_payloads,
    paired_rows,
    two_sided_sign_p_value,
)


def _example(task_id: str, selected: bool, best: bool, first: bool | None = None) -> dict[str, object]:
    return {
        "task_id": task_id,
        "test_index": 0,
        "has_target": True,
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": selected if first is None else first,
    }


def test_paired_rows_keep_common_target_examples_only() -> None:
    reference = {
        "examples": [
            _example("synthetic_move_recolor_000001", True, True),
            _example("only_reference", True, True),
            {"task_id": "no_target", "test_index": 0, "has_target": False},
        ]
    }
    candidate = {
        "examples": [
            _example("synthetic_move_recolor_000001", False, True),
            _example("only_candidate", True, True),
        ]
    }

    rows = paired_rows(reference, candidate, "selected_exact")

    assert rows == [
        {
            "task_id": "synthetic_move_recolor_000001",
            "test_index": 0,
            "family": "move_recolor",
            "reference": 1,
            "candidate": 0,
            "delta": -1,
        }
    ]


def test_two_sided_sign_p_value_handles_ties_and_extremes() -> None:
    assert two_sided_sign_p_value(0, 0) is None
    assert two_sided_sign_p_value(3, 0) == 0.25
    assert two_sided_sign_p_value(2, 2) == 1.0


def test_compare_payloads_reports_paired_metric_and_family_deltas() -> None:
    reference = {
        "summary": {"selected_exact": 1, "best_of_k_exact": 2},
        "examples": [
            _example("synthetic_move_recolor_000001", True, True),
            _example("synthetic_move_recolor_000002", False, True),
            _example("synthetic_frame_object_000003", False, False),
        ],
    }
    candidate = {
        "summary": {"selected_exact": 2, "best_of_k_exact": 3},
        "examples": [
            _example("synthetic_move_recolor_000001", True, True),
            _example("synthetic_move_recolor_000002", True, True),
            _example("synthetic_frame_object_000003", False, True),
        ],
    }

    payload = compare_payloads(
        reference,
        candidate,
        reference_label="base",
        candidate_label="recovered",
        bootstrap_samples=25,
        seed=123,
    )

    selected = payload["metrics"]["selected_exact"]
    best = payload["metrics"]["best_of_k_exact"]

    assert payload["common_examples"] == 3
    assert selected["reference_exact"] == 1
    assert selected["candidate_exact"] == 2
    assert selected["delta_exact"] == 1
    assert selected["wins"] == 1
    assert selected["losses"] == 0
    assert selected["ties"] == 2
    assert selected["bootstrap_delta_accuracy_ci95"]["low"] is not None
    assert best["delta_exact"] == 1
    assert payload["task_family_metrics"]["selected_exact"]["move_recolor"]["delta_exact"] == 1
    assert payload["task_family_metrics"]["best_of_k_exact"]["frame_object"]["delta_exact"] == 1
