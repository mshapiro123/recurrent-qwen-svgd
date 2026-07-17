"""Exact branching-coverage and comparator utilities for Phase G-alpha."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch


def exact_branching_coverage(
    predictions: list[str],
    reachable_symbols: list[str],
) -> dict[str, Any]:
    valid = set(map(str, reachable_symbols))
    if not valid:
        raise ValueError("reachable_symbols cannot be empty")
    unique_valid = sorted(set(map(str, predictions)) & valid)
    return {
        "samples": len(predictions),
        "valid_samples": sum(str(value) in valid for value in predictions),
        "unique_samples": len(set(map(str, predictions))),
        "unique_valid": unique_valid,
        "unique_valid_count": len(unique_valid),
        "coverage_denominator": len(valid),
        "coverage": len(unique_valid) / len(valid),
        "full_coverage": set(unique_valid) == valid,
        "duplicate_rate": 1.0 - len(set(map(str, predictions))) / max(1, len(predictions)),
    }


def categorical_entropy_from_scores(scores: dict[str, float], temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = torch.tensor(list(scores.values()), dtype=torch.float64) / float(temperature)
    probabilities = torch.softmax(values, dim=-1)
    return float(-(probabilities * probabilities.clamp_min(1e-300).log()).sum().item())


def solve_global_temperature(
    score_rows: list[dict[str, float]],
    *,
    target_mean_entropy: float,
    iterations: int = 80,
) -> dict[str, float]:
    if not score_rows:
        raise ValueError("score_rows cannot be empty")
    low, high = 1e-3, 100.0
    for _ in range(iterations):
        middle = math.sqrt(low * high)
        achieved = sum(
            categorical_entropy_from_scores(scores, middle) for scores in score_rows
        ) / len(score_rows)
        if achieved < target_mean_entropy:
            low = middle
        else:
            high = middle
    temperature = math.sqrt(low * high)
    achieved = sum(
        categorical_entropy_from_scores(scores, temperature) for scores in score_rows
    ) / len(score_rows)
    return {
        "temperature": temperature,
        "target_mean_entropy": float(target_mean_entropy),
        "achieved_mean_entropy": achieved,
        "absolute_error": abs(achieved - float(target_mean_entropy)),
    }


def summarize_coverage_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("coverage rows cannot be empty")

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "rows": len(selected),
            "mean_coverage": sum(float(row["coverage"]) for row in selected) / len(selected),
            "valid_sample_rate": (
                sum(int(row["valid_samples"]) for row in selected)
                / max(1, sum(int(row["samples"]) for row in selected))
            ),
            "mean_unique_valid": (
                sum(int(row["unique_valid_count"]) for row in selected) / len(selected)
            ),
            "full_coverage_rate": (
                sum(bool(row["full_coverage"]) for row in selected) / len(selected)
            ),
            "mean_duplicate_rate": (
                sum(float(row["duplicate_rate"]) for row in selected) / len(selected)
            ),
        }

    by_depth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_depth[str(row["depth"])].append(row)
        by_stratum[str(row["reachable_set_stratum"])].append(row)
    return {
        "overall": aggregate(rows),
        "by_depth": {key: aggregate(value) for key, value in sorted(by_depth.items())},
        "by_reachable_set_stratum": {
            key: aggregate(value) for key, value in sorted(by_stratum.items())
        },
    }
