"""Oracle-conditioned re-entry interfaces for the terminal Phase G probe."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .halting import masked_mean


@dataclass(frozen=True)
class OracleConditioningOutput:
    states: torch.Tensor
    branch_a_rms: torch.Tensor
    branch_b_rms: torch.Tensor
    residual_rms_ratio: torch.Tensor


class _ConditioningBranch(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.to(dtype=self.net[0].weight.dtype))


class OracleReentryConditioner(nn.Module):
    """Parameter-matched additive and FiLM command interfaces.

    Both modes use two independently learned branches with identical shapes.
    The additive arm folds both outputs into one residual. The FiLM arm treats
    them as ``gamma - 1`` and ``beta``. Zero-initialized output layers make
    both modes exact identities at installation.
    """

    MODES = ("additive", "film")

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        input_dim = 2 * self.hidden_dim
        self.branch_a = _ConditioningBranch(
            input_dim,
            self.bottleneck_dim,
            self.hidden_dim,
        )
        self.branch_b = _ConditioningBranch(
            input_dim,
            self.bottleneck_dim,
            self.hidden_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        command_embedding: torch.Tensor,
        *,
        mode: str,
        force_identity: bool = False,
    ) -> OracleConditioningOutput:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be [batch, sequence, hidden]")
        pooled = masked_mean(hidden_states, attention_mask)
        if command_embedding.shape != pooled.shape:
            raise ValueError(
                "command_embedding must match the pooled re-entry state; "
                f"got {tuple(command_embedding.shape)} versus {tuple(pooled.shape)}"
            )
        zero = hidden_states.new_zeros(())
        if force_identity:
            return OracleConditioningOutput(
                states=hidden_states,
                branch_a_rms=zero,
                branch_b_rms=zero,
                residual_rms_ratio=zero,
            )

        conditioning = torch.cat(
            [
                pooled.to(dtype=torch.float32),
                command_embedding.to(device=pooled.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        branch_a = self.branch_a(conditioning)
        branch_b = self.branch_b(conditioning)
        if mode == "additive":
            residual = (branch_a + branch_b) / math.sqrt(2.0)
            states = hidden_states + residual.unsqueeze(1).to(dtype=hidden_states.dtype)
        else:
            gamma = 1.0 + branch_a
            beta = branch_b
            states = (
                gamma.unsqueeze(1) * hidden_states.to(dtype=gamma.dtype)
                + beta.unsqueeze(1)
            ).to(dtype=hidden_states.dtype)
            residual = states.float() - hidden_states.float()

        state_rms = hidden_states.float().pow(2).mean().sqrt().clamp_min(1e-8)
        return OracleConditioningOutput(
            states=states,
            branch_a_rms=branch_a.float().pow(2).mean().sqrt(),
            branch_b_rms=branch_b.float().pow(2).mean().sqrt(),
            residual_rms_ratio=residual.float().pow(2).mean().sqrt() / state_rms,
        )
