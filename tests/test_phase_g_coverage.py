from __future__ import annotations

import pytest

from eval.phase_g_coverage import (
    categorical_entropy,
    exact_coverage,
    exact_valid_preimages,
    iso_compute_depth,
    temperature_for_target_entropy,
)


def constructed_row() -> dict:
    return {
        "id": "constructed",
        "n_symbols": 4,
        "depth": 1,
        "observed_target": "D",
        "symbol_names": ["A", "B", "C", "D"],
        "mapping_values": {"0": 3, "1": 3, "2": 2, "3": 2},
        "valid_starts": ["A", "B"],
        "coverage_denominator": 2,
    }


def test_exact_coverage_recomputes_forward_orbit_and_distinct_validity() -> None:
    row = constructed_row()

    assert exact_valid_preimages(row) == ["A", "B"]
    result = exact_coverage(["A", "A", "B", "D"], row)

    assert result["valid_samples"] == 3
    assert result["unique_valid_count"] == 2
    assert result["coverage"] == 1.0
    assert result["validity_check"] == "independent_forward_orbit_enumeration"


def test_exact_coverage_rejects_corrupt_manifest_denominator() -> None:
    row = constructed_row()
    row["coverage_denominator"] = 3

    with pytest.raises(ValueError, match="wrong coverage denominator"):
        exact_coverage(["A"], row)


def test_entropy_temperature_bisection_recovers_known_temperature() -> None:
    scores = {"A": 3.0, "B": 1.0, "C": -2.0}
    target = categorical_entropy(scores, 1.7)

    result = temperature_for_target_entropy(scores, target)

    assert result["clamped"] is False
    assert float(result["temperature"]) == pytest.approx(1.7, rel=1e-4)
    assert float(result["absolute_error"]) <= 1e-6


def test_iso_compute_depth_matches_total_recurrent_transitions() -> None:
    assert iso_compute_depth(trajectories=20, loops_per_trajectory=16) == 320
