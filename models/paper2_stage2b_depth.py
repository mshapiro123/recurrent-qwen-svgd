"""Stage 2B-D multi-lane recurrent scratchpad primitives."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules, SharedResidualFlow


def log_sinkhorn(logits: torch.Tensor, *, iterations: int = 20) -> torch.Tensor:
    """Return an approximately doubly stochastic lane-routing matrix."""

    if logits.ndim != 3 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Sinkhorn logits must be [batch, lanes, lanes]")
    if iterations < 1:
        raise ValueError("Sinkhorn requires at least one iteration")
    log_p = logits.float()
    for _ in range(iterations):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def routing_residuals(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row = (matrix.sum(dim=-1) - 1.0).abs().amax(dim=-1)
    column = (matrix.sum(dim=-2) - 1.0).abs().amax(dim=-1)
    return row, column


def second_eigenvalue_magnitude(matrix: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.eigvals(matrix.float()).abs().sort(dim=-1, descending=True).values
    return values[..., 1] if values.shape[-1] > 1 else values.new_zeros(values.shape[:-1])


def lane_effective_rank(state: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank of lane covariance, invariant to lane permutation."""

    if state.ndim != 4:
        raise ValueError("lane state must be [batch, lanes, slots, dim]")
    features = state.float().flatten(start_dim=2)
    features = features - features.mean(dim=1, keepdim=True)
    covariance = features @ features.transpose(-1, -2)
    covariance = covariance / max(features.shape[-1] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probabilities = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return entropy.exp()


@dataclass(frozen=True)
class MultiLaneStepOutput:
    state: torch.Tensor
    read_state: torch.Tensor
    routing: torch.Tensor
    sinkhorn_row_residual: torch.Tensor
    sinkhorn_column_residual: torch.Tensor
    lambda2: torch.Tensor
    effective_rank: torch.Tensor
    flow_update: torch.Tensor
    constitutive_update: torch.Tensor


@dataclass(frozen=True)
class MultiLaneFlowOutput:
    state: torch.Tensor
    read_state: torch.Tensor
    steps: tuple[MultiLaneStepOutput, ...]


class MultiLaneScratchFlow(nn.Module):
    """Four routed lane copies of the validated 8-slot residual flow.

    Carry routing acts only on the lane axis. Each lane retains the validated
    slot geometry and a lane read collapses the state back to [batch, 8, 128]
    before the existing bridge.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        latent_dim: int = 128,
        n_slots: int = 8,
        n_lanes: int = 4,
        max_steps: int = 4,
        sinkhorn_iterations: int = 20,
        base_flow: SharedResidualFlow | None = None,
    ) -> None:
        super().__init__()
        if n_lanes != 4 or n_slots != 8:
            raise ValueError("Stage 2B-D registers exactly four lanes and eight slots")
        self.context_dim = int(context_dim)
        self.latent_dim = int(latent_dim)
        self.n_slots = int(n_slots)
        self.n_lanes = int(n_lanes)
        self.max_steps = int(max_steps)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.base_flow = base_flow or SharedResidualFlow(
            latent_dim=latent_dim,
            context_dim=context_dim,
            n_slots=n_slots,
            max_steps=max_steps,
        )
        self.router_norm = nn.RMSNorm(latent_dim)
        self.router = nn.Linear(n_lanes * latent_dim, n_lanes * n_lanes)
        self.rho_logits = nn.Parameter(torch.full((max_steps,), math.log(0.01 / 0.99)))
        self.read_logits = nn.Parameter(torch.full((max_steps, n_lanes), -4.0))
        self.hidden_innovation = nn.Linear(context_dim, latent_dim, bias=False)
        self.prompt_gate = nn.Linear(context_dim, latent_dim, bias=False)
        self.lane_slot_scale = nn.Parameter(torch.ones(n_lanes, n_slots, latent_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.router.weight)
        bias = torch.zeros(self.n_lanes, self.n_lanes)
        bias.diagonal().fill_(4.0)
        with torch.no_grad():
            self.router.bias.copy_(bias.flatten())
            self.read_logits[:, 0] = 4.0
        nn.init.normal_(self.hidden_innovation.weight, std=1e-3)
        nn.init.zeros_(self.prompt_gate.weight)

    def replicate(self, scratch: torch.Tensor) -> torch.Tensor:
        if scratch.shape[-2:] != (self.n_slots, self.latent_dim):
            raise ValueError("single-lane scratch geometry changed")
        return scratch.unsqueeze(1).expand(-1, self.n_lanes, -1, -1).clone()

    def _routing(self, state: torch.Tensor, index: int, dynamic: bool) -> torch.Tensor:
        batch = state.shape[0]
        identity = torch.eye(self.n_lanes, device=state.device, dtype=torch.float32)
        identity = identity.unsqueeze(0).expand(batch, -1, -1)
        if not dynamic:
            return identity
        summary = self.router_norm(state.float()).mean(dim=2).flatten(start_dim=1)
        logits = self.router(summary).reshape(batch, self.n_lanes, self.n_lanes)
        projected = log_sinkhorn(logits, iterations=self.sinkhorn_iterations)
        rho = torch.sigmoid(self.rho_logits[index]).float()
        return (1.0 - rho) * identity + rho * projected

    def read(self, state: torch.Tensor, index: int, *, forced_lane_one: bool) -> torch.Tensor:
        if forced_lane_one:
            return state[:, 0]
        weights = torch.softmax(self.read_logits[index].float(), dim=-1)
        return torch.einsum("l,blsd->bsd", weights, state.float()).to(state.dtype)

    def step(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        index: int,
        *,
        prompt_context: torch.Tensor | None = None,
        dynamic_routing: bool = True,
        constitutive_active: bool = True,
        forced_lane_one: bool = False,
    ) -> MultiLaneStepOutput:
        if state.shape[1:] != (self.n_lanes, self.n_slots, self.latent_dim):
            raise ValueError("multi-lane scratch geometry changed")
        if not 0 <= index < self.max_steps:
            raise ValueError("loop index violates the registered cap")
        if context.shape != (state.shape[0], self.context_dim):
            raise ValueError("context shape changed")
        prompt_context = context if prompt_context is None else prompt_context
        if prompt_context.shape != context.shape:
            raise ValueError("prompt context must match recurrent context")

        routing = self._routing(state, index, dynamic_routing)
        carry = torch.einsum("bij,bjsd->bisd", routing.to(state.dtype), state)
        flat = carry.flatten(0, 1)
        repeated_context = context.repeat_interleave(self.n_lanes, dim=0)
        flowed, update, _magnitude, _ratio = self.base_flow.step(
            flat, repeated_context, index
        )
        flowed = flowed.unflatten(0, (state.shape[0], self.n_lanes))
        update = update.unflatten(0, (state.shape[0], self.n_lanes))
        if constitutive_active:
            hidden = self.hidden_innovation(context.float())
            gate = torch.sigmoid(self.prompt_gate(prompt_context.float()))
            constitutive = (hidden * gate)[:, None, None, :] * self.lane_slot_scale[None]
            flowed = flowed + constitutive.to(flowed.dtype)
        else:
            constitutive = flowed.new_zeros(flowed.shape)
        row_residual, column_residual = routing_residuals(routing)
        return MultiLaneStepOutput(
            state=flowed,
            read_state=self.read(flowed, index, forced_lane_one=forced_lane_one),
            routing=routing,
            sinkhorn_row_residual=row_residual,
            sinkhorn_column_residual=column_residual,
            lambda2=second_eigenvalue_magnitude(routing),
            effective_rank=lane_effective_rank(flowed),
            flow_update=update,
            constitutive_update=constitutive,
        )

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        *,
        steps: int,
        prompt_context: torch.Tensor | None = None,
        dynamic_routing: bool = True,
        constitutive_active: bool = True,
        forced_lane_one: bool = False,
    ) -> MultiLaneFlowOutput:
        if not 0 <= steps <= self.max_steps:
            raise ValueError("requested steps violate the registered loop cap")
        current = state
        outputs = []
        read_state = state[:, 0]
        for index in range(steps):
            output = self.step(
                current,
                context,
                index,
                prompt_context=prompt_context,
                dynamic_routing=dynamic_routing,
                constitutive_active=constitutive_active,
                forced_lane_one=forced_lane_one,
            )
            outputs.append(output)
            current = output.state
            read_state = output.read_state
        return MultiLaneFlowOutput(state=current, read_state=read_state, steps=tuple(outputs))


@dataclass
class Stage2BDepthTrace:
    lane_state: torch.Tensor | None
    reference: torch.Tensor
    control_state: torch.Tensor | None
    routing_steps: list[MultiLaneStepOutput]
    writeback_ratios: list[torch.Tensor]
    position_gates: list[torch.Tensor]

    def metrics(self) -> dict[str, torch.Tensor]:
        if not self.routing_steps:
            zero = self.reference.new_zeros(())
            return {
                "stage2b_reentry_steps": zero,
                "stage2b_sinkhorn_row_residual_max": zero,
                "stage2b_sinkhorn_column_residual_max": zero,
                "stage2b_lambda2_mean": zero,
                "stage2b_lane_effective_rank_mean": zero,
                "stage2b_writeback_ratio_mean": zero,
                "stage2b_position_gate_mean": zero,
            }
        row = torch.stack([step.sinkhorn_row_residual for step in self.routing_steps])
        column = torch.stack([step.sinkhorn_column_residual for step in self.routing_steps])
        lambda2 = torch.stack([step.lambda2 for step in self.routing_steps])
        rank = torch.stack([step.effective_rank for step in self.routing_steps])
        return {
            "stage2b_reentry_steps": self.reference.new_tensor(len(self.routing_steps)),
            "stage2b_sinkhorn_row_residual_max": row.amax().detach(),
            "stage2b_sinkhorn_column_residual_max": column.amax().detach(),
            "stage2b_lambda2_mean": lambda2.mean().detach(),
            "stage2b_lane_effective_rank_mean": rank.mean().detach(),
            "stage2b_writeback_ratio_mean": torch.stack(self.writeback_ratios).mean().detach(),
            "stage2b_position_gate_mean": torch.stack(self.position_gates).mean().detach(),
        }


@dataclass(frozen=True)
class Stage2BReentryOutput:
    hidden: torch.Tensor
    trace: Stage2BDepthTrace


class Stage2BDepthAttachment(nn.Module):
    """Phase-3-initialized mHC attachment at the recurrent boundary.

    The first recurrent pass never invokes this module. Each subsequent pass
    advances the four-lane scratch state once, reads it back through the
    validated Phase 3 control/bridge path, and returns the next recurrent input.
    """

    STAGES = {"M1", "M2", "M3", "M4"}

    def __init__(
        self,
        *,
        initializer: nn.Module,
        base_flow: SharedResidualFlow,
        control: nn.Module,
        bridge: nn.Module,
        context_dim: int,
        n_lanes: int = 4,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        self.initializer = initializer
        self.control = control
        self.bridge = bridge
        self.flow = MultiLaneScratchFlow(
            context_dim=context_dim,
            latent_dim=base_flow.latent_dim,
            n_slots=base_flow.n_slots,
            n_lanes=n_lanes,
            max_steps=base_flow.max_steps,
            sinkhorn_iterations=sinkhorn_iterations,
            base_flow=base_flow,
        )
        self.context_dim = int(context_dim)

    @classmethod
    def from_phase3(cls, sidecar: Phase3StudentModules) -> "Stage2BDepthAttachment":
        hidden_size = int(sidecar.bridge.hidden_norm.normalized_shape[0])
        return cls(
            initializer=copy.deepcopy(sidecar.initializer),
            base_flow=copy.deepcopy(sidecar.flow),
            control=copy.deepcopy(sidecar.control),
            bridge=copy.deepcopy(sidecar.bridge),
            context_dim=hidden_size,
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def begin(
        self, *, prelude_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> Stage2BDepthTrace:
        return Stage2BDepthTrace(
            lane_state=None,
            reference=prelude_hidden,
            control_state=None,
            routing_steps=[],
            writeback_ratios=[],
            position_gates=[],
        )

    def observe(
        self,
        *,
        trace: Stage2BDepthTrace,
        coda_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        loop_index: int,
    ) -> Stage2BDepthTrace:
        if loop_index == 0:
            if trace.lane_state is not None:
                raise RuntimeError("Stage 2B scratch was initialized more than once")
            scratch = self.initializer(coda_hidden, attention_mask.bool())
            trace.lane_state = self.flow.replicate(scratch)
        elif trace.lane_state is None:
            raise RuntimeError("Stage 2B scratch was not initialized after pass one")
        return trace

    def reenter(
        self,
        *,
        trace: Stage2BDepthTrace,
        prelude_hidden: torch.Tensor,
        recurrent_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        loop_index: int,
        stage: str,
        amplitude: float,
    ) -> Stage2BReentryOutput:
        if stage not in self.STAGES:
            raise ValueError(f"unknown Stage 2B curriculum stage: {stage}")
        if loop_index < 1 or loop_index > self.flow.max_steps:
            raise ValueError("Stage 2B re-entry index violates the loop cap")
        if not 0.0 < float(amplitude) <= 0.11:
            raise ValueError("Stage 2B amplitude must be inside the registered (0, 0.11] range")
        if trace.lane_state is None:
            raise RuntimeError("Stage 2B re-entry preceded post-coda scratch initialization")
        context = self._masked_mean(recurrent_hidden.float(), attention_mask)
        prompt_context = self._masked_mean(prelude_hidden.float(), attention_mask)
        forced_lane_one = stage == "M1"
        step = self.flow.step(
            trace.lane_state,
            context,
            loop_index - 1,
            prompt_context=prompt_context,
            dynamic_routing=stage in {"M3", "M4"},
            constitutive_active=stage in {"M2", "M3", "M4"},
            forced_lane_one=forced_lane_one,
        )
        innovation_norm = step.flow_update.float().square().mean(dim=-1).sqrt().mean(dim=(1, 2))
        control_state = self.control(
            scratch=step.read_state,
            previous=trace.control_state,
            innovation_norm=innovation_norm,
            student_entropy=context.new_zeros((context.shape[0],)),
            top2_margin=context.new_zeros((context.shape[0],)),
            position_bucket=torch.zeros(context.shape[0], dtype=torch.long, device=context.device),
        )
        self.bridge.set_gate_ceiling(float(amplitude))
        bridge = self.bridge(
            h0=prelude_hidden,
            previous=recurrent_hidden,
            scratch=step.read_state,
            control_state=control_state,
            loop_index=loop_index - 1,
            active=True,
            write_position_mask=attention_mask.bool().unsqueeze(-1),
        )
        trace.lane_state = step.state
        trace.control_state = control_state
        trace.routing_steps.append(step)
        trace.writeback_ratios.append(bridge.realized_writeback_ratio)
        trace.position_gates.append(bridge.position_gate)
        return Stage2BReentryOutput(hidden=bridge.hidden, trace=trace)
