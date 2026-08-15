from __future__ import annotations

import math

from analysis.build_paper2_phase3_p34_directional_stability import (
    GROUP_ORDER,
    adjacent_row_churn,
    score_curve_smoothing,
    shapley_r2,
)


def _row(item: str, *, look: int, base: bool, augmented: bool, battery: str = "gsm8k"):
    return {
        "item_id": item,
        "look": look,
        "base_correct": base,
        "augmented_correct": augmented,
        "battery": battery,
        "option_scores": {"A": 0.0, "B": -1.0, "C": -2.0, "D": -3.0},
    }


def test_adjacent_row_churn_tracks_identity_not_just_counts():
    prior = [
        _row("a", look=1, base=False, augmented=True),
        _row("b", look=1, base=True, augmented=False),
        _row("c", look=1, base=True, augmented=True, battery="mbpp"),
    ]
    current = [
        _row("a", look=2, base=False, augmented=False),
        _row("b", look=2, base=True, augmented=True),
        _row("c", look=2, base=True, augmented=True, battery="mbpp"),
    ]
    result = adjacent_row_churn(prior, current)
    assert result["outcome_changed_rows"] == 2
    assert math.isclose(result["outcome_changed_fraction"], 2 / 3)
    assert result["fix_set_jaccard"] == 0.0
    assert result["regression_set_jaccard"] == 0.0
    assert result["battery_change_rates"]["gsm8k"]["changed_fraction"] == 1.0
    assert result["battery_change_rates"]["mbpp"]["changed_fraction"] == 0.0


def test_shapley_r2_sums_to_incremental_fit():
    rows = []
    for seed in (0, 1):
        for look in range(2, 12):
            depth = 2.5 + 0.01 * look
            swing = 3.0 * depth + 0.2 * seed
            rows.append(
                {
                    "seed": seed,
                    "look": look,
                    "signed_swing": swing,
                    "absolute_swing": abs(swing),
                    "discordant_fraction": 0.1 + 0.001 * look,
                    "curriculum": {"mean_depth": depth, "code_fraction": 0.4},
                    "scored_gate_ceiling": 0.02,
                    "training_controller_transition_since_prior_score": False,
                    "maximum_absolute_log_weight_update": 0.05,
                    "share_demotions_in_segment": 0,
                }
            )
    result = shapley_r2(rows, "signed_swing")
    total = sum(result["shapley_incremental_r2"].values())
    assert set(result["shapley_incremental_r2"]) == set(GROUP_ORDER)
    assert math.isclose(total, result["incremental_r2"], abs_tol=1e-10)
    assert result["shapley_incremental_r2"]["curriculum"] > 0.0


def test_score_smoothing_is_labeled_as_telemetry_not_weight_ema():
    result = score_curve_smoothing([1, 2, 3, 4, 5])
    assert result["raw_endpoint"] == 5
    assert result["late_3_mean"] == 4.0
    assert "not a weight EMA" in result["scope"]
