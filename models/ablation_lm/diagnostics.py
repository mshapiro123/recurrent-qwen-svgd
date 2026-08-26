"""Trajectory-grounded calibration gates for experimental modules."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RouterMomentSnapshot:
    """Router logit moments ``(m, s)`` observed at one optimizer step."""

    step: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("router snapshot step must be a non-negative integer")
        values = torch.tensor((self.mean, self.std), dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or self.std < 0:
            raise ValueError("router moments must be finite and std must be non-negative")


@dataclass(frozen=True)
class RouterCalibrationDecision:
    """Whether a rolling trajectory, never step zero, can freeze calibration."""

    ready: bool
    reason: str
    mean_relative_drift: float | None
    std_relative_drift: float | None


def router_calibration_stability(
    snapshots: tuple[RouterMomentSnapshot, ...] | list[RouterMomentSnapshot],
    *,
    window: int = 8,
    minimum_step: int = 16,
    relative_tolerance: float = 0.05,
    absolute_floor: float = 1e-6,
) -> RouterCalibrationDecision:
    """Compare two adjacent windows of router ``(m, s)`` trajectory moments.

    The gate cannot pass from initialization data: it requires two full,
    non-overlapping windows ending no earlier than ``minimum_step``. This is a
    freeze-eligibility receipt, not an automatic freeze operation.
    """

    if type(window) is not int or window < 2:
        raise ValueError("window must be an integer of at least two")
    if type(minimum_step) is not int or minimum_step < 1:
        raise ValueError("minimum_step must be a positive integer")
    if (
        not math.isfinite(relative_tolerance)
        or not math.isfinite(absolute_floor)
        or relative_tolerance <= 0
        or absolute_floor <= 0
    ):
        raise ValueError("stability tolerances must be finite and positive")
    if len(snapshots) < 2 * window:
        return RouterCalibrationDecision(False, "insufficient_trajectory_windows", None, None)
    ordered = tuple(sorted(snapshots, key=lambda item: item.step))
    if len({item.step for item in ordered}) != len(ordered):
        raise ValueError("router calibration snapshots must have unique steps")
    if ordered[-1].step < minimum_step:
        return RouterCalibrationDecision(False, "minimum_nonzero_step_not_reached", None, None)
    previous = ordered[-2 * window : -window]
    current = ordered[-window:]
    window_steps = tuple(item.step for item in (*previous, *current))
    expected_steps = tuple(range(window_steps[0], window_steps[0] + 2 * window))
    if window_steps[0] <= 0 or window_steps != expected_steps:
        return RouterCalibrationDecision(False, "nonadjacent_or_step_zero_windows", None, None)

    def average(items: tuple[RouterMomentSnapshot, ...], field: str) -> float:
        return sum(float(getattr(item, field)) for item in items) / len(items)

    previous_mean = average(previous, "mean")
    current_mean = average(current, "mean")
    previous_std = average(previous, "std")
    current_std = average(current, "std")
    mean_drift = abs(current_mean - previous_mean) / max(abs(previous_mean), absolute_floor)
    std_drift = abs(current_std - previous_std) / max(abs(previous_std), absolute_floor)
    ready = mean_drift <= relative_tolerance and std_drift <= relative_tolerance
    return RouterCalibrationDecision(
        ready,
        "stable_trajectory_windows" if ready else "router_moments_still_drifting",
        mean_drift,
        std_drift,
    )
