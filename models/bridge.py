"""Identity-preserving bridge between recurrent transformer passes."""

from __future__ import annotations

import torch
from torch import nn


class IdentityGatedBridge(nn.Module):
    """A gated identity-initialized hidden-state translator.

    This intentionally does not use ``h = h + bridge(h)``. With identity
    initialization that would double the hidden state. Instead, the bridge
    learns a distribution translation from the previous recurrent output back
    into a plausible recurrent-block input.
    """

    def __init__(self, hidden_size: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.bridge_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        work = hidden_states.to(dtype=self.proj.weight.dtype)
        translated = self.proj(work)
        gate = self.bridge_gate.to(device=hidden_states.device, dtype=work.dtype)
        return (work + gate * (translated - work)).to(dtype=input_dtype)
