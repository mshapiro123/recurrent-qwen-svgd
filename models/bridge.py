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

    def __init__(
        self,
        hidden_size: int,
        gate_init: float = 1.0,
        projection_mode: str = "concat",
    ) -> None:
        super().__init__()
        if projection_mode not in {"concat", "split"}:
            raise ValueError("projection_mode must be one of: concat, split")
        self.hidden_size = int(hidden_size)
        self.split_projection = False
        self.prelude_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.reset_parameters()
        if projection_mode == "split":
            self.convert_to_split_projection()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.prelude_norm.reset_parameters()
            self.proj.weight.zero_()
            nn.init.eye_(self.proj.weight[:, self.hidden_size :])
            nn.init.zeros_(self.proj.bias)
            if self.split_projection:
                self.prelude_proj.weight.zero_()
                self.state_proj.weight.copy_(
                    torch.eye(
                        self.hidden_size,
                        device=self.state_proj.weight.device,
                        dtype=self.state_proj.weight.dtype,
                    )
                )
                if self.state_proj.bias is not None:
                    self.state_proj.bias.zero_()

    def convert_to_split_projection(self) -> None:
        """Split the concat projection into separate prelude/state parameters.

        This is functionally equivalent to the default concat projection, but it
        lets optimizers assign a true parameter-group LR to the prelude path.
        The legacy concat projection is kept frozen for checkpoint compatibility.
        """

        if self.split_projection:
            return
        hidden = self.hidden_size
        state_proj = nn.Linear(hidden, hidden, bias=self.proj.bias is not None)
        prelude_proj = nn.Linear(hidden, hidden, bias=False)
        state_proj.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        prelude_proj.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        with torch.no_grad():
            prelude_proj.weight.copy_(self.proj.weight[:, :hidden])
            state_proj.weight.copy_(self.proj.weight[:, hidden:])
            if self.proj.bias is not None and state_proj.bias is not None:
                state_proj.bias.copy_(self.proj.bias)
        self.prelude_proj = prelude_proj
        self.state_proj = state_proj
        for param in self.proj.parameters():
            param.requires_grad_(False)
        self.split_projection = True

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
        if self.split_projection:
            translated = self.prelude_proj(prelude) + self.state_proj(work)
        else:
            translated = self.proj(torch.cat([prelude, work], dim=-1))
        gate = self.bridge_gate.to(device=hidden_states.device, dtype=work.dtype)
        return (work + gate * (translated - work)).to(dtype=input_dtype)
