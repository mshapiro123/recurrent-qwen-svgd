"""Causal position-aligned scratch lanes and optional Birkhoff carrier."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import lanes_to_modes
from .layers import RMSNorm
from .optim import ParameterRole, mark_coupled_mode_adamw, tag_optimizer_role


class TwoLaneBirkhoffMixer(nn.Module):
    """The exact two-by-two Birkhoff family with non-oscillatory coupling.

    ``A=(1-rho)I+rho*P`` has eigenvalue one on the consensus coordinate and
    ``1-2*rho`` on disagreement.  Restricting ``rho`` to ``(0, 1/2)`` makes
    disagreement decay without sign flips while retaining spectral norm one.
    """

    def __init__(
        self,
        rho_init: float = 0.005,
        *,
        max_steps: int = 8,
        retention_floor: float = 0.9,
    ) -> None:
        super().__init__()
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not 0 < float(retention_floor) < 1:
            raise ValueError("retention_floor must lie strictly between zero and one")
        self.max_steps = max_steps
        self.retention_floor = float(retention_floor)
        self.rho_cap = (1.0 - self.retention_floor ** (1 / max_steps)) / 2
        if not 0 < float(rho_init) < self.rho_cap:
            raise ValueError(f"rho_init must lie below the retention cap {self.rho_cap:.8f}")
        probability = float(rho_init) / self.rho_cap
        raw = math.log(probability / (1.0 - probability))
        self.raw_rho = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        tag_optimizer_role(self, "raw_rho", ParameterRole.GATE)

    def rho(self) -> torch.Tensor:
        return self.rho_cap * torch.sigmoid(self.raw_rho)

    def minimum_retention(self, steps: int) -> torch.Tensor:
        """Exact minimum singular-value retention after ``steps`` carries."""

        if type(steps) is not int or steps < 1 or steps > self.max_steps:
            raise ValueError("steps must lie within the carrier horizon")
        return (1.0 - 2.0 * self.rho()).pow(steps)

    def matrix(self) -> torch.Tensor:
        rho = self.rho()
        one = torch.ones_like(rho)
        return torch.stack((one - rho, rho, rho, one - rho)).view(2, 2)

    def forward(self, lanes: torch.Tensor) -> torch.Tensor:
        if lanes.ndim < 2 or lanes.shape[-2] != 2:
            raise ValueError("lanes must have a final-but-one axis of length two")
        rho = self.rho().to(device=lanes.device, dtype=lanes.dtype)
        return (1.0 - rho) * lanes + rho * lanes.flip(dims=(-2,))


class PositionAlignedScratch(nn.Module):
    """Two causal lanes per token, with no cross-position read or write."""

    def __init__(
        self,
        d_model: int,
        *,
        lane_width: int,
        max_steps: int,
        layer_scale: float,
        rho_init: float,
        retention_floor: float,
        norm_eps: float,
        use_carrier: bool,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.lane_width = int(lane_width)
        self.max_steps = int(max_steps)
        self.hidden_norm = RMSNorm(d_model, norm_eps)
        self.lane_norm = RMSNorm(lane_width, norm_eps)
        self.initializer = mark_coupled_mode_adamw(
            nn.Linear(d_model, 2 * lane_width, bias=False)
        )
        self.context_projection = mark_coupled_mode_adamw(
            nn.Linear(d_model, lane_width, bias=False)
        )
        self.step_embedding = nn.Embedding(max_steps, lane_width)
        self.update_in = mark_coupled_mode_adamw(
            nn.Linear(2 * lane_width, 2 * lane_width, bias=False)
        )
        self.update_out = mark_coupled_mode_adamw(
            nn.Linear(2 * lane_width, lane_width, bias=False)
        )
        self.readout = mark_coupled_mode_adamw(
            nn.Linear(2 * lane_width, d_model, bias=False)
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))
        tag_optimizer_role(self, "layer_scale", ParameterRole.GATE)
        self.carrier = (
            TwoLaneBirkhoffMixer(
                rho_init,
                max_steps=max_steps,
                retention_floor=retention_floor,
            )
            if use_carrier
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initializer.weight)
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.xavier_uniform_(self.update_in.weight)
        nn.init.normal_(self.update_out.weight, mean=0.0, std=1e-3)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.step_embedding.weight)

    def initialize(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, length, _ = hidden.shape
        return self.initializer(self.hidden_norm(hidden)).view(
            batch, length, 2, self.lane_width
        )

    def inject(
        self,
        hidden: torch.Tensor,
        lanes: torch.Tensor,
        *,
        residual_scale: float,
    ) -> torch.Tensor:
        self._validate_alignment(hidden, lanes)
        coordinates = lanes_to_modes(lanes)
        bridge_input = torch.cat((coordinates.mu, coordinates.delta), dim=-1)
        update = self.readout(bridge_input)
        scale = float(residual_scale) * self.layer_scale.to(dtype=hidden.dtype)
        return hidden + scale * update.to(dtype=hidden.dtype)

    def step(
        self,
        lanes: torch.Tensor,
        hidden: torch.Tensor,
        *,
        step_index: int,
        residual_scale: float,
    ) -> torch.Tensor:
        """Advance both lanes from one shared bring-up hidden stream."""

        self._validate_alignment(hidden, lanes)
        if step_index < 0 or step_index >= self.max_steps:
            raise ValueError("step_index exceeds the configured recurrence cap")
        context = self.context_projection(self.hidden_norm(hidden)).unsqueeze(-2)
        context = context + self.step_embedding.weight[step_index].view(1, 1, 1, -1)
        context = context.expand_as(lanes)
        features = torch.cat((self.lane_norm(lanes), context), dim=-1)
        update = self.update_out(F.silu(self.update_in(features)))
        updated = lanes + float(residual_scale) * update
        return self.carrier(updated) if self.carrier is not None else updated

    def step_bicameral(
        self,
        lanes: torch.Tensor,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        step_index: int,
        residual_scale: float,
    ) -> torch.Tensor:
        """Advance lane A from ``h_a`` and lane B from ``h_b`` position-wise.

        The projection weights remain shared; only the already-causal source
        state differs.  This is the full-width bicameral lane update and adds no
        cross-position or cross-hemisphere read.
        """

        if h_a.shape != h_b.shape:
            raise ValueError("bicameral hidden states must have identical shapes")
        self._validate_alignment(h_a, lanes)
        self._validate_alignment(h_b, lanes)
        if h_a.device != h_b.device or h_a.dtype != h_b.dtype:
            raise ValueError("bicameral hidden states must share one dtype and device")
        if step_index < 0 or step_index >= self.max_steps:
            raise ValueError("step_index exceeds the configured recurrence cap")
        context = torch.stack(
            (
                self.context_projection(self.hidden_norm(h_a)),
                self.context_projection(self.hidden_norm(h_b)),
            ),
            dim=-2,
        )
        context = context + self.step_embedding.weight[step_index].view(1, 1, 1, -1)
        features = torch.cat((self.lane_norm(lanes), context), dim=-1)
        update = self.update_out(F.silu(self.update_in(features)))
        updated = lanes + float(residual_scale) * update
        return self.carrier(updated) if self.carrier is not None else updated

    def _validate_alignment(self, hidden: torch.Tensor, lanes: torch.Tensor) -> None:
        expected = (*hidden.shape[:2], 2, self.lane_width)
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if tuple(lanes.shape) != expected:
            raise ValueError(f"lanes must have shape {expected}, got {tuple(lanes.shape)}")
