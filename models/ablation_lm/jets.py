"""First-order jet utilities for local recurrent Jacobian diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FirstOrderJet:
    """A primal value and one directional derivative."""

    primal: torch.Tensor
    tangent: torch.Tensor


@dataclass(frozen=True)
class TrajectoryJetMetrics:
    """Basis-invariant first/second finite-difference trajectory telemetry."""

    velocity_rms: torch.Tensor
    acceleration_rms: torch.Tensor
    turning_cosine: torch.Tensor
    wedge_gram: torch.Tensor


def first_order_jet(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    tangent: torch.Tensor,
) -> FirstOrderJet:
    """Evaluate ``(f(x), J_f(x) tangent)`` without materializing ``J_f``."""

    if primal.shape != tangent.shape:
        raise ValueError("primal and tangent must have identical shapes")
    value, directional = torch.func.jvp(function, (primal,), (tangent,))
    return FirstOrderJet(primal=value, tangent=directional)


def estimate_jacobian_spectral_norm(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    *,
    iterations: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """Power-iterate on ``J.T J`` using only JVP/VJP operations."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    generator = torch.Generator(device=primal.device).manual_seed(int(seed))
    direction = torch.randn(
        primal.shape,
        generator=generator,
        device=primal.device,
        dtype=primal.dtype,
    )
    direction = direction / direction.float().norm().clamp_min(1e-12).to(direction.dtype)
    _output, vjp = torch.func.vjp(function, primal)
    sigma = primal.new_zeros((), dtype=torch.float32)
    for _ in range(iterations):
        jet = first_order_jet(function, primal, direction)
        sigma = jet.tangent.float().norm()
        (normal_direction,) = vjp(jet.tangent)
        norm = normal_direction.float().norm().clamp_min(1e-12)
        direction = normal_direction / norm.to(dtype=normal_direction.dtype)
    return sigma


def trajectory_jet_metrics(states: torch.Tensor, *, eps: float = 1e-12) -> TrajectoryJetMetrics:
    """Compute velocity, acceleration, turning, and wedge/Gram invariants.

    ``states`` uses ``[visits, ..., width]`` and requires at least three visits.
    Reductions preserve all axes between visit and width, making it possible to
    retain token-level telemetry or reduce it later under an explicit mask.
    """

    if states.ndim < 2 or states.shape[0] < 3:
        raise ValueError("trajectory jets require [at least 3 visits, ..., width]")
    velocity = states[1:].float() - states[:-1].float()
    acceleration = velocity[1:] - velocity[:-1]
    aligned_velocity = velocity[1:]
    velocity_sq = aligned_velocity.square().sum(dim=-1)
    acceleration_sq = acceleration.square().sum(dim=-1)
    inner = (aligned_velocity * acceleration).sum(dim=-1)
    denominator = (velocity_sq * acceleration_sq).clamp_min(eps).sqrt()
    turning = inner / denominator
    wedge = (velocity_sq * acceleration_sq - inner.square()).clamp_min(0.0)
    return TrajectoryJetMetrics(
        velocity_rms=aligned_velocity.square().mean(dim=-1).sqrt(),
        acceleration_rms=acceleration.square().mean(dim=-1).sqrt(),
        turning_cosine=turning,
        wedge_gram=wedge,
    )
