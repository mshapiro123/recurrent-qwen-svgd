"""Typed lane coordinates and explicitly Euclidean Clifford primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LaneModeState:
    """Mean/disagreement coordinates for two lanes.

    This is the idempotent coordinate system of the two-by-two Birkhoff carrier,
    not a Clifford rotor and not a claim about a split-signature geometry.
    """

    mu: torch.Tensor
    delta: torch.Tensor

    def lanes(self) -> torch.Tensor:
        if self.mu.shape != self.delta.shape:
            raise ValueError("mu and delta must have identical shapes")
        return torch.stack((self.mu + self.delta, self.mu - self.delta), dim=-2)


def lanes_to_modes(lanes: torch.Tensor) -> LaneModeState:
    """Convert a final-but-one lane axis of length two into ``(mu, delta)``."""

    if lanes.ndim < 2 or lanes.shape[-2] != 2:
        raise ValueError("lanes must have a final-but-one axis of length two")
    lane_a, lane_b = lanes.unbind(dim=-2)
    return LaneModeState(mu=(lane_a + lane_b) / 2, delta=(lane_a - lane_b) / 2)


@dataclass(frozen=True)
class Cl20Rotor:
    """A Euclidean ``Cl(2,0)`` rotor with bivector square ``B**2 = -1``.

    ``scalar + bivector*B`` stores ``cos(theta/2) + sin(theta/2) B``.
    The sandwich action on a two-coordinate vector is an ordinary orthogonal
    rotation. Hyperbolic ``Cl(1,0)`` boosts are deliberately absent.
    """

    angle: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.angle, torch.Tensor) or not self.angle.is_floating_point():
            raise TypeError("Cl(2,0) rotor angle must be a floating tensor")
        if not bool(torch.isfinite(self.angle.detach()).all()):
            raise ValueError("Cl(2,0) rotor angle must be finite")

    @property
    def scalar(self) -> torch.Tensor:
        return (self.angle / 2).cos()

    @property
    def bivector(self) -> torch.Tensor:
        return (self.angle / 2).sin()

    @classmethod
    def from_angle(cls, angle: torch.Tensor | float) -> "Cl20Rotor":
        angle_tensor = torch.as_tensor(angle)
        if not angle_tensor.is_floating_point() or angle_tensor.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            angle_tensor = angle_tensor.float()
        return cls(angle=angle_tensor)

    def rotate(self, vector: torch.Tensor) -> torch.Tensor:
        return cl20_rotate(vector, self.angle)


def cl20_rotate(vector: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    """Apply a pure-tensor Euclidean rotor action, safe for JVP/vmap/compile.

    ``angle`` must be scalar or have exactly ``vector.shape[:-1]``. Requiring an
    exact batch shape prevents accidental outer-product broadcasting; callers
    that want a shared angle along an axis must expand it explicitly.
    """

    if vector.ndim < 1 or vector.shape[-1] != 2 or not vector.is_floating_point():
        raise ValueError("Cl(2,0) rotor input must be floating with final width two")
    compute_dtype = torch.float64 if vector.dtype == torch.float64 else torch.float32
    angle_tensor = torch.as_tensor(angle, device=vector.device).to(dtype=compute_dtype)
    if angle_tensor.ndim != 0 and tuple(angle_tensor.shape) != tuple(vector.shape[:-1]):
        raise ValueError(
            "Cl(2,0) angle must be scalar or exactly match the vector batch shape"
        )
    cosine = angle_tensor.cos()
    sine = angle_tensor.sin()
    x, y = vector.to(dtype=compute_dtype).unbind(dim=-1)
    output = torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)
    return output.to(dtype=vector.dtype)
