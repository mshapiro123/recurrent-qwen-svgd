"""Identity-preserving adapters for recurrent loop re-entry."""

from __future__ import annotations

import torch
from torch import nn


class ReentryAffineAdapter(nn.Module):
    """A tiny loop-only affine correction.

    The adapter is initialized to exact identity, so enabling it does not change
    the initial recurrent computation. Unlike a zero-gated residual, both scale
    and bias receive gradients immediately when the adapter is used.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.scale.fill_(1.0)
            self.bias.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        work_dtype = self.scale.dtype
        work = hidden_states.to(dtype=work_dtype)
        scale = self.scale.to(device=hidden_states.device, dtype=work_dtype)
        bias = self.bias.to(device=hidden_states.device, dtype=work_dtype)
        return (work * scale.view(1, 1, -1) + bias.view(1, 1, -1)).to(dtype=hidden_states.dtype)
