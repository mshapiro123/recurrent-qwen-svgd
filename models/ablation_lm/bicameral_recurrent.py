"""Cache-policy- and recombination-agnostic bicameral recurrence seam.

This module deliberately owns only the repeated execution schedule.  It does
not choose how the two live states are initialized, how a block represents its
fixed-context K/V payload, or how the terminal pair is recombined.  Those
interfaces are the still-open C-S5-1 and C-S5-2 strategy rulings.

Keeping the seam this narrow lets the full-width paired core enter the
recurrent path without encoding either open answer.  A concrete block remains
responsible for validating the opaque cache object it receives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

import torch
from torch import nn


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return converted


@dataclass(frozen=True)
class BicameralRecurrenceReceipt:
    """Execution-only receipt; it makes no architecture-choice claim."""

    recurrent_steps: int
    unique_core_blocks: int
    executed_block_passes: int
    recurrence_c: float
    residual_scale: float
    cache_policy: str
    terminal_recombination: str

    def __post_init__(self) -> None:
        steps = _positive_integer(self.recurrent_steps, name="recurrent_steps")
        blocks = _positive_integer(self.unique_core_blocks, name="unique_core_blocks")
        recurrence_c = _finite_positive(self.recurrence_c, name="recurrence_c")
        expected_scale = recurrence_c / steps
        if self.executed_block_passes != steps * blocks:
            raise ValueError("executed_block_passes does not match steps times blocks")
        if self.residual_scale != expected_scale:
            raise ValueError("residual_scale does not equal recurrence_c / recurrent_steps")
        if self.cache_policy != "caller-owned (C-S5-2 unbound)":
            raise ValueError("the recurrence seam may not claim a selected cache policy")
        if self.terminal_recombination != "not executed (C-S5-1 unbound)":
            raise ValueError("the recurrence seam may not claim terminal recombination")


@dataclass(frozen=True)
class BicameralRecurrentOutput:
    """The still-separated terminal hemisphere states plus schedule receipt."""

    h_a: torch.Tensor
    h_b: torch.Tensor
    receipt: BicameralRecurrenceReceipt


class BicameralRecurrentCore(nn.Module):
    """Repeat caller-supplied full-width paired blocks for ``K`` visits.

    ``projected_kv`` is intentionally opaque at this layer.  Every concrete
    block receives exactly its caller-owned entry and must validate ownership,
    shape, positions, dtype, and device itself.  The method returns the two
    live states rather than silently selecting a terminal combiner.
    """

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise TypeError("blocks must be a sequence of modules")
        if not blocks:
            raise ValueError("blocks must contain at least one core block")
        if not all(isinstance(block, nn.Module) for block in blocks):
            raise TypeError("every core block must be an nn.Module")
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def _validate_state_pair(h_a: torch.Tensor, h_b: torch.Tensor) -> None:
        if not isinstance(h_a, torch.Tensor) or not isinstance(h_b, torch.Tensor):
            raise TypeError("h_a and h_b must be tensors")
        if h_a.shape != h_b.shape:
            raise ValueError("h_a and h_b must have identical shapes")
        if h_a.ndim != 3:
            raise ValueError("h_a and h_b must have shape [batch, sequence, width]")
        if not h_a.is_floating_point() or not h_b.is_floating_point():
            raise TypeError("h_a and h_b must be floating point")
        if h_a.device != h_b.device or h_a.dtype != h_b.dtype:
            raise ValueError("h_a and h_b must share one dtype and device")

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        projected_kv: Sequence[Any],
        recurrent_steps: int,
        recurrence_c: float,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        force_math_attention: bool = False,
    ) -> BicameralRecurrentOutput:
        """Execute the core schedule without choosing either open S5 policy."""

        self._validate_state_pair(h_a, h_b)
        steps = _positive_integer(recurrent_steps, name="recurrent_steps")
        c_value = _finite_positive(recurrence_c, name="recurrence_c")
        if not isinstance(projected_kv, Sequence) or isinstance(
            projected_kv,
            (str, bytes),
        ):
            raise TypeError("projected_kv must be a sequence")
        if len(projected_kv) != len(self.blocks):
            raise ValueError("projected_kv must contain one entry per unique core block")

        residual_scale = c_value / steps
        for _visit in range(steps):
            for block, cache in zip(self.blocks, projected_kv, strict=True):
                h_a, h_b = block(
                    h_a,
                    h_b,
                    projected_kv=cache,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    document_ids=document_ids,
                    residual_scale=residual_scale,
                    force_math_attention=force_math_attention,
                )
                self._validate_state_pair(h_a, h_b)

        receipt = BicameralRecurrenceReceipt(
            recurrent_steps=steps,
            unique_core_blocks=len(self.blocks),
            executed_block_passes=steps * len(self.blocks),
            recurrence_c=c_value,
            residual_scale=residual_scale,
            cache_policy="caller-owned (C-S5-2 unbound)",
            terminal_recombination="not executed (C-S5-1 unbound)",
        )
        return BicameralRecurrentOutput(h_a=h_a, h_b=h_b, receipt=receipt)
