"""Pure decision contracts for the locked Phase-2 A2 matrix."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


PRIMARY = ("cumulative_kl", "local_ce")
NON_PRIMARY = ("final_ce", "preserve_kl")


def classify_relative_explosion(
    *,
    prior_norms: Sequence[float],
    current_norm: float,
    previous_consecutive_exceedances: int,
    window: int = 100,
    multiplier: float = 10.0,
    consecutive_to_stop: int = 3,
) -> dict[str, Any]:
    """Classify a trajectory-relative gradient explosion before an update.

    The current observation is excluded from the trailing reference. Saved A2
    checkpoints do not contain historical per-step norms, so the guard remains
    telemetry until a full post-resume window has been observed.
    """

    if window < 1 or consecutive_to_stop < 1 or multiplier <= 1:
        raise ValueError("relative explosion constants are invalid")
    if not math.isfinite(float(current_norm)) or float(current_norm) < 0:
        raise ValueError("current gradient norm must be finite and non-negative")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in prior_norms):
        raise ValueError("prior gradient norms must be finite and non-negative")
    armed = len(prior_norms) >= int(window)
    reference = (
        float(statistics.median(float(value) for value in prior_norms[-window:]))
        if armed
        else None
    )
    threshold = float(multiplier) * reference if reference is not None else None
    exceeds = bool(armed and float(current_norm) > float(threshold))
    consecutive = int(previous_consecutive_exceedances) + 1 if exceeds else 0
    return {
        "armed": armed,
        "history_count": len(prior_norms),
        "window": int(window),
        "reference_median": reference,
        "multiplier": float(multiplier),
        "threshold": threshold,
        "current_norm": float(current_norm),
        "exceeds": exceeds,
        "consecutive_exceedances": consecutive,
        "consecutive_to_stop": int(consecutive_to_stop),
        "stop": armed and consecutive >= int(consecutive_to_stop),
    }


def validate_guardrail_inventory(inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the Guardrail Doctrine fields for every A2 runner rule."""

    required = {
        "name",
        "threshold",
        "estimator",
        "reference_point",
        "cadence",
        "disposition",
        "named_cliff",
    }
    allowed_dispositions = {"log", "warn", "stop", "endpoint_verdict"}
    seen: set[str] = set()
    errors: list[str] = []
    for number, row in enumerate(inventory):
        missing = required.difference(row)
        if missing:
            errors.append(f"row {number} missing {sorted(missing)}")
            continue
        name = str(row["name"])
        if name in seen:
            errors.append(f"duplicate rule {name}")
        seen.add(name)
        disposition = str(row["disposition"])
        if disposition not in allowed_dispositions:
            errors.append(f"{name} has invalid disposition {disposition}")
        if disposition == "stop" and not str(row["named_cliff"]).strip():
            errors.append(f"{name} has stop authority without a named cliff")
    return {
        "valid": not errors,
        "rules": len(inventory),
        "stop_rules": sum(row.get("disposition") == "stop" for row in inventory),
        "telemetry_rules": sum(row.get("disposition") == "log" for row in inventory),
        "errors": errors,
    }


def classify_directional_shares(shares: Mapping[str, float]) -> dict[str, Any]:
    """Classify one matched-estimator audit under the locked severity tiers."""

    required = set(PRIMARY + NON_PRIMARY)
    if set(shares) != required:
        raise ValueError("directional shares must contain the four registered losses")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in shares.values()):
        raise ValueError("directional shares must be finite and non-negative")

    gross = []
    marginal = []
    primary = sum(float(shares[name]) for name in PRIMARY)
    if primary < 0.40:
        gross.append("primary:below_0p40")
    elif primary < 0.50:
        marginal.append("primary:below_0p50")
    for name in NON_PRIMARY:
        value = float(shares[name])
        if value > 0.35:
            gross.append(f"{name}:above_0p35")
        elif value > 0.25:
            marginal.append(f"{name}:above_0p25")
    return {
        "classification": "gross" if gross else "marginal" if marginal else "pass",
        "primary_share": primary,
        "gross_bounds": gross,
        "marginal_bounds": marginal,
    }


def repeated_marginal_bounds(
    previous: Sequence[str], current: Sequence[str]
) -> list[str]:
    """Return marginal bounds missed in two consecutive audits."""

    return sorted(set(previous).intersection(current))


def relative_oracle_headroom(
    *, quality_safe_oracle_delta: float, zero_loop_mean: float
) -> float:
    if not math.isfinite(quality_safe_oracle_delta) or not math.isfinite(zero_loop_mean):
        raise ValueError("headroom inputs must be finite")
    if zero_loop_mean <= 0:
        raise ValueError("zero-loop mean must be positive")
    return float(quality_safe_oracle_delta) / float(zero_loop_mean)


def final_window_slope(history: Sequence[Mapping[str, float]], *, window: int = 100) -> float:
    """Return per-step accepted-length slope across the final registered window."""

    if len(history) < 2:
        raise ValueError("slope requires at least two evaluations")
    ordered = sorted(history, key=lambda row: int(row["step"]))
    final_step = int(ordered[-1]["step"])
    eligible = [row for row in ordered if int(row["step"]) >= final_step - int(window)]
    if len(eligible) < 2:
        raise ValueError("history does not span the requested slope window")
    first, last = eligible[0], eligible[-1]
    elapsed = int(last["step"]) - int(first["step"])
    if elapsed <= 0:
        raise ValueError("slope evaluations must have distinct steps")
    return (float(last["mean_accepted_length"]) - float(first["mean_accepted_length"])) / elapsed


def retention_slope_latest(
    history: Sequence[Mapping[str, float]], *, evaluations: int = 3
) -> float:
    """Return the least-squares retention slope over the latest evaluations."""

    if evaluations < 2 or len(history) < evaluations:
        raise ValueError("retention slope requires the requested evaluation window")
    rows = sorted(history, key=lambda row: int(row["step"]))[-evaluations:]
    steps = [float(row["step"]) for row in rows]
    values = [float(row["retention"]) for row in rows]
    center_step = sum(steps) / len(steps)
    center_value = sum(values) / len(values)
    denominator = sum((step - center_step) ** 2 for step in steps)
    if denominator <= 0:
        raise ValueError("retention evaluations must have distinct steps")
    return sum(
        (step - center_step) * (value - center_value)
        for step, value in zip(steps, values)
    ) / denominator


def classify_inflight_quality(
    *, step_zero_retention: float, retention: float, wilson_lower: float,
    previous_point_failures: int, point_drop: float = 0.003,
    point_failures_to_stop: int = 2, wilson_floor: float = 0.990,
) -> dict[str, Any]:
    """Apply the locked trajectory-grounded quality tripwire."""

    values = (step_zero_retention, retention, wilson_lower, point_drop, wilson_floor)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("quality tripwire inputs must be finite")
    point_floor = float(step_zero_retention) - float(point_drop)
    point_miss = float(retention) < point_floor
    consecutive = int(previous_point_failures) + 1 if point_miss else 0
    wilson_miss = float(wilson_lower) < float(wilson_floor)
    return {
        "point_floor": point_floor,
        "point_miss": point_miss,
        "consecutive_point_misses": consecutive,
        "wilson_floor": float(wilson_floor),
        "wilson_miss": wilson_miss,
        "stop": wilson_miss or consecutive >= int(point_failures_to_stop),
        "stop_reason": (
            "quality_wilson_floor_immediate"
            if wilson_miss
            else "quality_trajectory_two_consecutive_evaluations"
            if consecutive >= int(point_failures_to_stop)
            else None
        ),
    }


def should_extend(
    *, relative_headroom: float, accepted_length_slope: float,
    minimum_headroom: float = 0.02, maximum_slope: float = 0.002,
) -> bool:
    return float(relative_headroom) < float(minimum_headroom) or float(
        accepted_length_slope
    ) > float(maximum_slope)


def paired_verdict(
    *, relative_headroom: float, full_mean: float, control_mean: float,
    quality_noninferior: bool,
) -> dict[str, Any]:
    gates = {
        "oracle_headroom_at_least_0p02": float(relative_headroom) >= 0.02,
        "full_strictly_better_than_control": float(full_mean) > float(control_mean),
        "endpoint_quality_retained": bool(quality_noninferior),
    }
    return {
        "gates": gates,
        "verdict": "positive" if all(gates.values()) else "budget-limited",
    }
