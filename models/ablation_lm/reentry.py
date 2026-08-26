"""Causal anchored bridge for recurrent re-entry."""

from __future__ import annotations

import torch
from torch import nn

from .layers import RMSNorm
from .optim import ParameterRole, tag_optimizer_role


class AnchoredReentryBridge(nn.Module):
    """Inject the fixed causal prelude at each additional recurrent visit.

    The control arm omits this module structurally.  The active arm uses a
    small, strictly nonzero LayerScale and a nonzero projection, so both the
    scale and the branch internals receive gradients on their first use.
    """

    def __init__(self, d_model: int, *, layer_scale: float, norm_eps: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.prelude_norm = RMSNorm(d_model, norm_eps)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))
        tag_optimizer_role(self, "layer_scale", ParameterRole.GATE)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(
        self,
        state: torch.Tensor,
        prelude: torch.Tensor,
        *,
        residual_scale: float,
    ) -> torch.Tensor:
        if state.shape != prelude.shape or state.ndim != 3 or state.shape[-1] != self.d_model:
            raise ValueError("state and prelude must share [batch, sequence, d_model]")
        update = self.projection(self.prelude_norm(prelude))
        scale = float(residual_scale) * self.layer_scale.to(dtype=state.dtype)
        return state + scale * update.to(dtype=state.dtype)
