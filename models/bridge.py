"""Identity-preserving bridge between recurrent transformer passes."""

from __future__ import annotations

import torch
from torch import nn


class IdentityGatedBridge(nn.Module):
    """A gated identity-initialized hidden-state translator.

    Recurrent passes need both the previous recurrent state and the fixed
    prelude/question representation. The projection therefore maps
    ``[prelude, state]`` back to one hidden stream. Its initialization zeros the
    prelude half and applies identity to the state half, so it exactly matches
    the older autonomous bridge until training learns to use the injected input.
    """

    def __init__(self, hidden_size: int, gate_init: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prelude_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.prelude_norm.reset_parameters()
            self.proj.weight.zero_()
            nn.init.eye_(self.proj.weight[:, self.hidden_size :])
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prelude_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        work = hidden_states.to(dtype=self.proj.weight.dtype)
        if prelude_hidden is None:
            prelude = torch.zeros_like(work)
        else:
            prelude = prelude_hidden.to(device=hidden_states.device, dtype=self.proj.weight.dtype)
            if prelude.shape != work.shape:
                raise ValueError(
                    "prelude_hidden must have the same shape as hidden_states; "
                    f"got prelude_hidden={tuple(prelude.shape)}, hidden_states={tuple(work.shape)}"
                )
            prelude = self.prelude_norm(prelude)
        translated = self.proj(torch.cat([prelude, work], dim=-1))
        gate = self.bridge_gate.to(device=hidden_states.device, dtype=work.dtype)
        return (work + gate * (translated - work)).to(dtype=input_dtype)
