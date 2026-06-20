from __future__ import annotations

from colab.run_stage5_arc_agi_recovered_benchmark import (
    final_stage_row,
    metric_delta,
    recovered_checkpoint_from_curriculum,
)


def test_final_stage_row_returns_last_stage() -> None:
    curriculum = {
        "stages": [
            {"stage": {"name": "warmup"}},
            {"stage": {"name": "mixed"}},
        ]
    }

    assert final_stage_row(curriculum)["stage"]["name"] == "mixed"


def test_recovered_checkpoint_from_curriculum_prefers_final_stage_selected_checkpoint() -> None:
    curriculum = {
        "final_checkpoint": "final.pt",
        "stages": [
            {
                "selected_checkpoint": {"checkpoint": "stage0.pt"},
            },
            {
                "selected_checkpoint": {"checkpoint": "stage1.pt"},
            },
        ],
    }

    checkpoint = recovered_checkpoint_from_curriculum(curriculum)

    assert checkpoint.name == "stage1.pt"


def test_metric_delta_tracks_core_arc_metrics() -> None:
    delta = metric_delta(
        {
            "first_exact": 2,
            "selected_exact": 3,
            "best_of_k_exact": 4,
            "tasks_solved_best_of_k": 5,
        },
        {
            "first_exact": 1,
            "selected_exact": 3,
            "best_of_k_exact": 2,
            "tasks_solved_best_of_k": 7,
        },
    )

    assert delta == {
        "first_exact_delta": 1,
        "selected_exact_delta": 0,
        "best_of_k_exact_delta": 2,
        "tasks_solved_best_of_k_delta": -2,
    }
