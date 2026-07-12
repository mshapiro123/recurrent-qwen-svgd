"""Exact Phase G coverage and entropy-matching utilities."""

from __future__ import annotations

import math
from typing import Any

from training.synthetic_depth_task import NAME_SYMBOLS


def row_symbol_names(row: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(str(item) for item in (row.get("symbol_names") or []))
    if not names:
        n_symbols = int(row.get("n_symbols", len(NAME_SYMBOLS)))
        names = tuple(NAME_SYMBOLS[:n_symbols])
    if len(names) < int(row.get("n_symbols", len(names))):
        raise ValueError(f"Row {row.get('id')} symbol_names do not cover its universe")
    return names


def exact_valid_preimages(row: dict[str, Any]) -> list[str]:
    """Enumerate validity from the forward map instead of trusting the manifest."""

    names = row_symbol_names(row)
    mapping = {int(left): int(right) for left, right in row["mapping_values"].items()}
    if set(mapping) != set(range(int(row["n_symbols"]))):
        raise ValueError(f"Row {row.get('id')} mapping does not cover its symbol universe")
    target_name = str(row["observed_target"])
    if target_name not in names:
        raise ValueError(f"Row {row.get('id')} target {target_name!r} is outside symbol_names")
    target = names.index(target_name)
    depth = int(row["depth"])
    valid = []
    for start in range(int(row["n_symbols"])):
        current = start
        for _ in range(depth):
            current = mapping[current]
        if current == target:
            valid.append(names[start])
    stored = [str(item) for item in row.get("valid_starts") or []]
    if sorted(stored) != sorted(valid):
        raise ValueError(
            f"Row {row.get('id')} stored preimages differ from independent forward enumeration"
        )
    if int(row.get("coverage_denominator", -1)) != len(valid):
        raise ValueError(f"Row {row.get('id')} stores the wrong coverage denominator")
    return valid


def exact_coverage(samples: list[str], row: dict[str, Any]) -> dict[str, Any]:
    valid_starts = set(exact_valid_preimages(row))
    valid_samples = [sample for sample in samples if sample in valid_starts]
    unique_valid = sorted(set(valid_samples))
    denominator = len(valid_starts)
    return {
        "samples": samples,
        "valid_samples": len(valid_samples),
        "invalid_samples": len(samples) - len(valid_samples),
        "unique_samples": len(set(samples)),
        "unique_valid": unique_valid,
        "unique_valid_count": len(unique_valid),
        "coverage_denominator": denominator,
        "coverage": len(unique_valid) / denominator,
        "full_coverage": set(unique_valid) == valid_starts,
        "validity_check": "independent_forward_orbit_enumeration",
    }


def categorical_probabilities(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = {name: float(score) / float(temperature) for name, score in scores.items()}
    maximum = max(scaled.values())
    weights = {name: math.exp(value - maximum) for name, value in scaled.items()}
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def categorical_entropy(scores: dict[str, float], temperature: float) -> float:
    probabilities = categorical_probabilities(scores, temperature)
    return -sum(probability * math.log(probability) for probability in probabilities.values() if probability > 0)


def temperature_for_target_entropy(
    scores: dict[str, float],
    target_entropy: float,
    *,
    minimum: float = 1e-3,
    maximum: float = 100.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> dict[str, float | bool]:
    """Invert categorical entropy with bounded bisection.

    The target is clamped only when score ties make it unattainable at the
    configured temperature bounds; the clamp is recorded for preregistered
    comparator diagnostics.
    """

    if not scores:
        raise ValueError("scores cannot be empty")
    low_entropy = categorical_entropy(scores, minimum)
    high_entropy = categorical_entropy(scores, maximum)
    target = float(target_entropy)
    if target <= low_entropy:
        return {
            "temperature": float(minimum),
            "achieved_entropy": low_entropy,
            "target_entropy": target,
            "absolute_error": abs(low_entropy - target),
            "clamped": True,
        }
    if target >= high_entropy:
        return {
            "temperature": float(maximum),
            "achieved_entropy": high_entropy,
            "target_entropy": target,
            "absolute_error": abs(high_entropy - target),
            "clamped": True,
        }
    low = float(minimum)
    high = float(maximum)
    midpoint = (low + high) / 2.0
    achieved = categorical_entropy(scores, midpoint)
    for _ in range(int(max_iterations)):
        midpoint = (low + high) / 2.0
        achieved = categorical_entropy(scores, midpoint)
        if abs(achieved - target) <= tolerance:
            break
        if achieved < target:
            low = midpoint
        else:
            high = midpoint
    return {
        "temperature": midpoint,
        "achieved_entropy": achieved,
        "target_entropy": target,
        "absolute_error": abs(achieved - target),
        "clamped": False,
    }


def iso_compute_depth(*, trajectories: int, loops_per_trajectory: int) -> int:
    if trajectories < 1 or loops_per_trajectory < 1:
        raise ValueError("trajectories and loops_per_trajectory must be positive")
    return int(trajectories) * int(loops_per_trajectory)
