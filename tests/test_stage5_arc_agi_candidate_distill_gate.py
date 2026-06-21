from __future__ import annotations

from colab.run_stage5_arc_agi_candidate_distill_gate import compact, metric_delta


def _summary(selected: int, best: int, valid_rate: float = 1.0) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": selected,
        "valid_candidate_rate": valid_rate,
        "examples_with_targets": 10,
        "tasks_solved_best_of_k": best,
        "tasks_with_targets": 10,
    }


def test_compact_counts_candidate_distill_rows_and_best_checkpoint() -> None:
    payload = {
        "base": _summary(3, 4),
        "phase1_start": _summary(1, 2),
        "phase1_arc_agi_tuned": _summary(5, 6, valid_rate=0.8),
        "best_checkpoint": {
            "step": 200,
            "summary": _summary(7, 8, valid_rate=0.9),
        },
        "candidate_distill_info": [
            {
                "rows": 4,
                "selector_generated_rows": 2,
                "selected_rows": 3,
                "selected_exceeds_best_of_k_rows": 1,
                "program_fit_rows": 1,
            },
            {
                "rows": 6,
                "selector_generated_rows": 4,
                "selected_rows": 5,
                "selected_exceeds_best_of_k_rows": 2,
                "program_fit_rows": 0,
            },
        ],
    }

    row = compact(payload)

    assert row["base_selected"] == 3
    assert row["start_best"] == 2
    assert row["tuned_selected"] == 5
    assert row["best_step"] == 200
    assert row["best_selected"] == 7
    assert row["candidate_distill_rows"] == 10
    assert row["candidate_distill_selector_generated_rows"] == 6
    assert row["candidate_distill_selected_rows"] == 8
    assert row["candidate_distill_selected_exceeds_best_of_k_rows"] == 3
    assert row["candidate_distill_program_fit_rows"] == 1


def test_compact_uses_final_checkpoint_when_no_ladder_best() -> None:
    payload = {
        "base": _summary(3, 4),
        "phase1_start": _summary(1, 2),
        "phase1_arc_agi_tuned": _summary(5, 6, valid_rate=0.8),
    }

    row = compact(payload)

    assert row["best_step"] is None
    assert row["best_selected"] == 5
    assert row["candidate_distill_rows"] == 0
    assert row["candidate_distill_selector_generated_rows"] == 0


def test_metric_delta_compares_candidate_distill_against_baseline() -> None:
    baseline = {
        "tuned_selected": 2,
        "tuned_best": 4,
        "best_selected": 3,
        "best_best": 5,
        "tasks_solved_best": 1,
        "best_valid_rate": 0.7,
    }
    candidate = {
        "tuned_selected": 4,
        "tuned_best": 4,
        "best_selected": 6,
        "best_best": 7,
        "tasks_solved_best": 2,
        "best_valid_rate": 0.9,
    }

    delta = metric_delta(candidate, baseline)

    assert delta == {
        "tuned_selected_delta": 2,
        "tuned_best_delta": 0,
        "best_selected_delta": 3,
        "best_best_delta": 2,
        "tasks_solved_best_delta": 1,
        "best_valid_rate_delta": 0.20000000000000007,
    }
