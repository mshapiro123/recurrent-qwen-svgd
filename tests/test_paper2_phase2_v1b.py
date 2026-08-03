from __future__ import annotations

import math

import pytest
import torch

from eval.eval_paper2_phase2_v1b import (
    _load_append_predictions,
    aggregate_by_first_order_distance_quantile,
    capped_state_rms,
    compare_paired_predictions,
    aggregate_intervention_records,
    deterministic_position_sample,
    parse_c_values,
    tube_radius,
    wilson_lower_bound,
)


def test_capped_state_rms_preserves_raw_contract() -> None:
    assert capped_state_rms(0.4, 0.55) == (0.4, False)
    assert capped_state_rms(58.0, 0.55) == (0.55, True)
    assert capped_state_rms(0.4, None) == (0.4, False)


def test_wilson_lower_bound_matches_locked_v1d_scale() -> None:
    observed = wilson_lower_bound(1997, 2000)
    assert 0.995 < observed < 0.996


def test_paired_comparison_separates_batch_shape_drift_from_causal_change() -> None:
    teacher = torch.tensor([3, 4, 5, 6])
    registered = torch.tensor([1, 4, 8, 6])
    neutral = torch.tensor([2, 4, 8, 6])
    perturbed = torch.tensor([2, 4, 5, 6])

    observed = compare_paired_predictions(
        registered=registered,
        neutral=neutral,
        perturbed=perturbed,
        teacher=teacher,
        position=2,
    )

    assert observed["neutral_vs_registered_prediction_changes"] == 1
    assert observed["neutral_vs_registered_prefix_changes"] == 1
    assert observed["causal_prefix_prediction_changes"] == 0
    assert observed["target_correct_before"] is False
    assert observed["target_correct_after"] is True


def test_paired_comparison_rejects_a_true_prior_position_change() -> None:
    with pytest.raises(
        RuntimeError,
        match="batch-matched neutral",
    ):
        compare_paired_predictions(
            registered=torch.tensor([1, 2, 3]),
            neutral=torch.tensor([1, 2, 3]),
            perturbed=torch.tensor([9, 2, 3]),
            teacher=torch.tensor([1, 2, 4]),
            position=2,
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


def test_parse_c_values_locks_order_and_rejects_duplicates() -> None:
    assert parse_c_values("0.075,0.10,0.15") == (0.075, 0.1, 0.15)
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_c_values("0.10,0.075")
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_c_values("0.10,0.10")


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


def test_first_order_distance_quantiles_cover_full_primary_population() -> None:
    records = []
    for position in range(8):
        for c_value in (0.05, 0.10):
            records.append(
                {
                    "cohort": "oracle_help",
                    "row_index": position,
                    "position": position,
                    "c_value": c_value,
                    "gradient_l2": 1.0,
                    "margin_before": float(position + 1),
                    "margin_after": 0.0,
                    "radius": c_value,
                    "state_rms": 1.0,
                    "first_order_predicted_pair_cross": False,
                    "realized_pair_cross": False,
                    "target_correct_before": False,
                    "target_correct_after": False,
                    "collateral_positions": 1,
                    "collateral_helps": 0,
                    "collateral_hurts": 0,
                    "collateral_prediction_changes": 0,
                    "position": position,
                    "scored_positions": 10,
                }
            )
    observed = aggregate_by_first_order_distance_quantile(records)["oracle_help"]
    assert [observed[f"q{index}"]["unique_positions"] for index in range(1, 5)] == [2, 2, 2, 2]
    assert sum(observed[f"q{index}"]["results"]["0.05"]["positions"] for index in range(1, 5)) == 8


def test_append_loader_consumes_locked_prefix_and_ignores_trailing_cache(tmp_path) -> None:
    rows = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [1, 2]},
        {"input_ids": [1, 2, 3]},
        {"input_ids": [1]},
    ]
    for batch_number, indices in (
        (1, [0, 2]),
        (2, [1]),
        (3, [3]),
        (4, [4]),
    ):
        torch.save(
            {
                "indices": indices,
                "predictions": torch.tensor(
                    [[[batch_number, batch_number + 10]]] * len(indices)
                ),
            },
            tmp_path / f"batch_{batch_number:06d}.pt",
        )

    loaded = _load_append_predictions(cache_dir=tmp_path, rows=rows, batch_size=2)

    assert len(loaded) == 4
    assert loaded[0].tolist() == [[1, 11]]
    assert loaded[3].tolist() == [[3, 13]]
