"""Canonical frontier metrics for Stage 5 synthetic-depth experiments."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_BAR = 0.71


def _numeric_depth_items(values: dict[Any, Any]) -> list[tuple[int, float]]:
    items: list[tuple[int, float]] = []
    for raw_depth, raw_value in values.items():
        try:
            depth = int(raw_depth)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            items.append((depth, value))
    return sorted(items)


def bar_crossing_frontier(diag_acc: dict[Any, Any], *, bar: float = DEFAULT_BAR) -> float:
    """Return the canonical interpolated depth frontier.

    ``diag_acc`` maps depth to active-label diagonal accuracy.  The frontier is
    the first linear interpolation where the curve crosses below ``bar``.  If
    every observed depth clears the bar, return the last observed depth.  If the
    curve starts below the bar, or no valid depths are present, return NaN.
    """

    items = _numeric_depth_items(diag_acc)
    if not items:
        return float("nan")
    first_depth, first_value = items[0]
    if first_value < bar:
        return float("nan")
    for (left_depth, left_value), (right_depth, right_value) in zip(items, items[1:]):
        if left_value >= bar > right_value:
            denominator = left_value - right_value
            if denominator <= 0:
                return float(left_depth)
            return float(left_depth) + ((left_value - bar) / denominator) * float(right_depth - left_depth)
    last_depth, last_value = items[-1]
    return float(last_depth) if last_value >= bar else float("nan")


def frontier_in_band(frontier: float, *, target: float, tolerance: float = 1.0) -> bool:
    return math.isfinite(frontier) and abs(float(frontier) - float(target)) <= float(tolerance)


def deepest_passing_selection_frontier(selection: dict[Any, Any]) -> int:
    """Deprecated diagnostic: deepest selected depth whose local gate passed."""

    frontier = 0
    for raw_depth, item in sorted(selection.items(), key=lambda item: int(item[0])):
        if isinstance(item, dict) and bool(item.get("pass")):
            frontier = int(raw_depth)
    return frontier


def selection_accuracies(selection: dict[Any, Any]) -> dict[str, float]:
    """Extract depth accuracies from a route-score selection table."""

    out: dict[str, float] = {}
    for raw_depth, item in sorted(selection.items(), key=lambda item: int(item[0])):
        if not isinstance(item, dict):
            continue
        if "accuracy" in item:
            out[str(int(raw_depth))] = float(item["accuracy"])
        elif int(item.get("total", 0)):
            out[str(int(raw_depth))] = float(item.get("correct", 0)) / float(item["total"])
    return out


def diagonal_counts_to_accuracy(diagonal_counts: dict[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_depth, item in sorted(diagonal_counts.items(), key=lambda item: int(item[0])):
        if isinstance(item, dict) and "accuracy" in item:
            out[str(int(raw_depth))] = float(item["accuracy"])
    return out
