"""Exact two-lane Clifford coordinates used by the bicameral substrate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplitCliffordState:
    """Coefficients of ``mu + delta*e`` in Cl(1, 0), where ``e**2 = 1``.

    The two primitive idempotents ``(1+e)/2`` and ``(1-e)/2`` correspond to
    the two hemispheric lanes.  This is an exact change of coordinates, not a
    metaphorical geometric-algebra layer.
    """

    mu: torch.Tensor
    delta: torch.Tensor

    def lanes(self) -> torch.Tensor:
        if self.mu.shape != self.delta.shape:
            raise ValueError("mu and delta must have identical shapes")
        return torch.stack((self.mu + self.delta, self.mu - self.delta), dim=-2)


def lanes_to_split_clifford(lanes: torch.Tensor) -> SplitCliffordState:
    """Convert a final-but-one lane axis of length two into ``(mu, delta)``."""

    if lanes.ndim < 2 or lanes.shape[-2] != 2:
        raise ValueError("lanes must have a final-but-one axis of length two")
    lane_a, lane_b = lanes.unbind(dim=-2)
    return SplitCliffordState(mu=(lane_a + lane_b) / 2, delta=(lane_a - lane_b) / 2)


def split_clifford_product(
    left: SplitCliffordState, right: SplitCliffordState
) -> SplitCliffordState:
    """Multiply ``(a+b e)(c+d e)`` under the exact relation ``e**2=1``."""

    if left.mu.shape != left.delta.shape or right.mu.shape != right.delta.shape:
        raise ValueError("each Clifford state must have aligned coefficients")
    return SplitCliffordState(
        mu=left.mu * right.mu + left.delta * right.delta,
        delta=left.mu * right.delta + left.delta * right.mu,
    )
