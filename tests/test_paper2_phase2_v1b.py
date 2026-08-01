from __future__ import annotations

import math

import torch

from eval.eval_paper2_phase2_v1b import (
    _load_append_predictions,
    aggregate_intervention_records,
    deterministic_position_sample,
    tube_radius,
)


def test_tube_radius_matches_governing_formula() -> None:
    observed = tube_radius(
        c_value=0.05,
        state_rms=2.0,
        hidden_size=16,
        gamma=0.05,
        rho=0.8,
    )
    assert math.isclose(observed, 0.1)


def test_deterministic_position_sample_is_seeded_and_cohort_specific() -> None:
    rows = [
        {"row_id": "r1", "position": 1},
        {"row_id": "r2", "position": 2},
        {"row_id": "r3", "position": 3},
    ]
    first = deterministic_position_sample(
        rows, sample_size=2, seed=20260731, cohort="oracle_help"
    )
    second = deterministic_position_sample(
        rows, sample_size=2, seed=20260731, cohort="oracle_help"
    )
    reordered = deterministic_position_sample(
        list(reversed(rows)), sample_size=2, seed=20260731, cohort="oracle_help"
    )
    assert first == second
    assert first == reordered


def test_intervention_aggregate_separates_pair_crossing_and_teacher_flip() -> None:
    records = [
        {
            "cohort": "oracle_help",
            "c_value": 0.05,
            "first_order_predicted_pair_cross": True,
            "realized_pair_cross": True,
            "target_correct_before": False,
            "target_correct_after": True,
            "collateral_positions": 3,
            "collateral_helps": 1,
            "collateral_hurts": 0,
            "collateral_prediction_changes": 1,
        },
        {
            "cohort": "oracle_help",
            "c_value": 0.05,
            "first_order_predicted_pair_cross": True,
            "realized_pair_cross": True,
            "target_correct_before": False,
            "target_correct_after": False,
            "collateral_positions": 3,
            "collateral_helps": 0,
            "collateral_hurts": 1,
            "collateral_prediction_changes": 2,
        },
        {
            "cohort": "preserve_control",
            "c_value": 0.05,
            "first_order_predicted_pair_cross": False,
            "realized_pair_cross": False,
            "target_correct_before": True,
            "target_correct_after": True,
            "collateral_positions": 3,
            "collateral_helps": 0,
            "collateral_hurts": 0,
            "collateral_prediction_changes": 0,
        },
    ]
    summary = aggregate_intervention_records(records)
    help_cell = summary["oracle_help"]["0.05"]
    assert help_cell["positions"] == 2
    assert help_cell["first_order_predicted_pair_cross_rate"] == 1.0
    assert help_cell["realized_pair_cross_rate"] == 1.0
    assert help_cell["realized_teacher_flip_rate"] == 0.5
    assert help_cell["pair_cross_prediction_gap"] == 0.0
    assert help_cell["teacher_flip_minus_first_order_gap"] == -0.5
    assert help_cell["collateral_hurt_rate"] == 1 / 6
    assert help_cell["collateral_net"] == 0

    control = summary["preserve_control"]["0.05"]
    assert control["target_preservation_rate"] == 1.0
    assert control["collateral_hurt_rate"] == 0.0


def test_append_loader_consumes_locked_prefix_and_ignores_trailing_cache(tmp_path) -> None:
    for batch_number, indices in ((1, [0, 1]), (2, [2, 3]), (3, [4, 5])):
        torch.save(
            {
                "indices": indices,
                "predictions": torch.tensor(
                    [[[batch_number, batch_number + 10]]] * len(indices)
                ),
            },
            tmp_path / f"batch_{batch_number:06d}.pt",
        )

    loaded = _load_append_predictions(cache_dir=tmp_path, rows=4, batch_size=2)

    assert len(loaded) == 4
    assert loaded[0].tolist() == [[1, 11]]
    assert loaded[3].tolist() == [[2, 12]]
