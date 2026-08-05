"""Pure decision contracts for the locked Phase-2 A2 matrix."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


PRIMARY = ("cumulative_kl", "local_ce")
NON_PRIMARY = ("final_ce", "preserve_kl")


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
