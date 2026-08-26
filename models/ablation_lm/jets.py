"""First-order jet utilities for local recurrent Jacobian diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

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
    curvature: torch.Tensor
    gram_eigenvalue_ratio: torch.Tensor


@dataclass(frozen=True)
class LoopGradientMetrics:
    """Per-visit hidden-state adjoint magnitudes and cosine geometry."""

    gradient_rms: torch.Tensor
    adjacent_cosines: torch.Tensor
    pairwise_cosines: torch.Tensor


@dataclass(frozen=True)
class LoopGradientProbe:
    """References to retained post-visit states; evaluate after ``backward``."""

    states: tuple[torch.Tensor, ...]
    valid_token_mask: torch.Tensor | None = None

    def metrics(self, *, eps: float = 1e-12) -> LoopGradientMetrics:
        gradients: list[torch.Tensor] = []
        for index, state in enumerate(self.states):
            if state.grad is None:
                raise RuntimeError(f"loop state {index} has no retained gradient; call backward first")
            gradient = state.grad
            if self.valid_token_mask is not None:
                if gradient.ndim < 2 or self.valid_token_mask.shape != gradient.shape[:-1]:
                    raise ValueError("loop gradient mask must align with all non-hidden axes")
                gradient = gradient[self.valid_token_mask]
            gradients.append(gradient)
        return loop_gradient_metrics(tuple(gradients), eps=eps)


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
    input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Power-iterate on ``J.T J`` using JVP/VJP and an optional input projection."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if input_mask is not None:
        if input_mask.shape != primal.shape or input_mask.device != primal.device:
            raise ValueError("Jacobian input_mask must align with the primal")
        input_mask = input_mask.bool()
        if not bool(input_mask.any()):
            raise ValueError("Jacobian input_mask must retain at least one element")
    generator = torch.Generator(device=primal.device).manual_seed(int(seed))
    direction = torch.randn(
        primal.shape,
        generator=generator,
        device=primal.device,
        dtype=primal.dtype,
    )
    if input_mask is not None:
        direction = direction.masked_fill(~input_mask, 0.0)
    direction = direction / direction.float().norm().clamp_min(1e-12).to(direction.dtype)
    _output, vjp = torch.func.vjp(function, primal)
    sigma = primal.new_zeros((), dtype=torch.float32)
    for _ in range(iterations):
        jet = first_order_jet(function, primal, direction)
        sigma = jet.tangent.float().norm()
        (normal_direction,) = vjp(jet.tangent)
        if input_mask is not None:
            normal_direction = normal_direction.masked_fill(~input_mask, 0.0)
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
    wedge_gram = (velocity_sq * acceleration_sq - inner.square()).clamp_min(0.0)
    wedge_norm = wedge_gram.sqrt()
    speed = velocity_sq.sqrt()
    trace = velocity_sq + acceleration_sq
    discriminant = (
        (velocity_sq - acceleration_sq).square() + 4.0 * inner.square()
    ).sqrt()
    eigen_max = ((trace + discriminant) * 0.5).clamp_min(0.0)
    eigen_min = ((trace - discriminant) * 0.5).clamp_min(0.0)
    return TrajectoryJetMetrics(
        velocity_rms=velocity.square().mean(dim=-1).sqrt(),
        acceleration_rms=acceleration.square().mean(dim=-1).sqrt(),
        turning_cosine=turning,
        wedge_gram=wedge_gram,
        curvature=wedge_norm / (speed.pow(3) + eps),
        gram_eigenvalue_ratio=eigen_min / eigen_max.clamp_min(eps),
    )


def loop_gradient_metrics(
    gradients: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> LoopGradientMetrics:
    """Measure hidden-state adjoint cosines across recurrent visits.

    These are state-adjoint cosines, not per-visit shared-parameter contribution
    cosines. The latter requires a separately functionalized parameter probe.
    """

    if len(gradients) < 1:
        raise ValueError("at least one loop gradient is required")
    shape = gradients[0].shape
    if any(gradient.shape != shape for gradient in gradients):
        raise ValueError("all loop gradients must have identical shapes")
    flattened = torch.stack([gradient.float().reshape(-1) for gradient in gradients])
    norms = flattened.norm(dim=-1, keepdim=True).clamp_min(eps)
    normalized = flattened / norms
    pairwise = normalized @ normalized.T
    adjacent = pairwise.diagonal(offset=1)
    return LoopGradientMetrics(
        gradient_rms=flattened.square().mean(dim=-1).sqrt(),
        adjacent_cosines=adjacent,
        pairwise_cosines=pairwise,
    )


def plane_probe_features(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Seedable JL probes of the oriented plane spanned by ``(v, a)``."""

    if velocity.shape != acceleration.shape:
        raise ValueError("velocity and acceleration must have identical shapes")
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != velocity.shape[-1]:
        raise ValueError("p and q must share shape [probes, width]")
    velocity_unit = torch.nn.functional.normalize(velocity.float(), dim=-1, eps=eps)
    radial = (acceleration.float() * velocity_unit).sum(dim=-1, keepdim=True)
    perpendicular = torch.nn.functional.normalize(
        acceleration.float() - radial * velocity_unit,
        dim=-1,
        eps=eps,
    )
    p_float = p.float()
    q_float = q.float()
    features = (
        (velocity_unit @ p_float.T) * (perpendicular @ q_float.T)
        - (velocity_unit @ q_float.T) * (perpendicular @ p_float.T)
    )
    return features / math.sqrt(2.0 * p.shape[0])
