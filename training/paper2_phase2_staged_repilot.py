"""Pure contracts for the locked Phase-2 staged re-pilot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


PROTOCOL_LOCK_COMMIT = "c0ace7aa02ebea11fc0809298572736d29b24012"


def solve_static_weights(
    mean_gradient_norms: Mapping[str, float],
    target_shares: Mapping[str, float],
    *,
    anchor: str,
    minimum_norm: float = 1e-12,
) -> dict[str, float]:
    if set(mean_gradient_norms) != set(target_shares):
        raise ValueError("gradient norms and target shares must name the same losses")
    if anchor not in target_shares:
        raise ValueError("anchor loss is absent")
    if not math.isclose(sum(target_shares.values()), 1.0, abs_tol=1e-12):
        raise ValueError("target shares must sum to one")
    for name, value in mean_gradient_norms.items():
        if not math.isfinite(value) or value < minimum_norm:
            raise ValueError(f"calibration gradient for {name} is unusable: {value}")
    anchor_norm = float(mean_gradient_norms[anchor])
    anchor_share = float(target_shares[anchor])
    return {
        name: (float(share) / anchor_share)
        * (anchor_norm / float(mean_gradient_norms[name]))
        for name, share in target_shares.items()
    }


def realized_gradient_shares(
    gradient_norms: Mapping[str, float], weights: Mapping[str, float]
) -> dict[str, float]:
    if set(gradient_norms) != set(weights):
        raise ValueError("gradient norms and weights must name the same losses")
    weighted = {
        name: abs(float(weights[name])) * float(value)
        for name, value in gradient_norms.items()
    }
    denominator = sum(weighted.values())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("realized gradient-share denominator is invalid")
    return {name: value / denominator for name, value in weighted.items()}


def shares_within_absolute_tolerance(
    realized: Mapping[str, float], target: Mapping[str, float], *, tolerance: float
) -> bool:
    return set(realized) == set(target) and all(
        abs(float(realized[name]) - float(target[name])) <= float(tolerance)
        for name in target
    )


def drift_alarm(
    realized: Mapping[str, float], target: Mapping[str, float], *, ratio: float
) -> bool:
    if ratio <= 1:
        raise ValueError("drift ratio must exceed one")
    return any(
        float(realized[name]) > ratio * float(target[name])
        or float(realized[name]) < float(target[name]) / ratio
        for name in target
    )


def trust_tripwire(
    history: Sequence[bool], *, window: int = 100, maximum_exceeding: int = 50
) -> bool:
    return len(history) >= window and sum(bool(value) for value in history[-window:]) > int(
        maximum_exceeding
    )


def a1_gate(
    *,
    initial_probe_kl: float,
    final_probe_kl: float,
    initial_flow_mse: float,
    final_flow_mse: float,
    minimum_probe_improvement: float = 0.10,
) -> bool:
    probe_improvement = float(initial_probe_kl) - float(final_probe_kl)
    return (
        probe_improvement + 1e-12 >= float(minimum_probe_improvement)
        and final_flow_mse < initial_flow_mse
    )


def relative_improvement(previous: float, current: float) -> float:
    if previous <= 0 or not math.isfinite(previous) or not math.isfinite(current):
        raise ValueError("relative improvement requires finite positive losses")
    return (previous - current) / previous


def a1_should_extend(
    *, gate_passed: bool, step_900_flow_loss: float, step_1000_flow_loss: float
) -> bool:
    return (not gate_passed) or relative_improvement(
        step_900_flow_loss, step_1000_flow_loss
    ) > 0.005
