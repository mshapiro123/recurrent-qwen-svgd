"""Build-only Phase-2 student modules governed by the DC2 E1 r3 contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _rms(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values.float().square().mean(dim=-1) + eps).sqrt()


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slot_mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss over populated slots only; reserved slots never enter the reduction."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [batch, slots, width]")
    if slot_mask.shape != prediction.shape[:2]:
        raise ValueError("slot_mask must match [batch, slots]")
    selected = slot_mask.bool().unsqueeze(-1).expand_as(prediction)
    if not bool(selected.any()):
        raise ValueError("masked loss requires at least one populated coordinate")
    return F.huber_loss(prediction[selected], target[selected], delta=float(delta))


def masked_effective_rank(states: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    """Entropy effective rank using only populated slots."""

    if states.ndim != 3 or slot_mask.shape != states.shape[:2]:
        raise ValueError("states and slot_mask must align as [batch, slots, width]")
    selected = states[slot_mask.bool()].float()
    if selected.shape[0] < 2:
        raise ValueError("effective rank requires at least two populated rows")
    singular = torch.linalg.svdvals(selected)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return entropy.exp()


class ScratchpadInitializer(nn.Module):
    """Anchor-dominated scratch state with one low-rank cross-attention read."""

    def __init__(self, *, hidden_size: int, latent_dim: int = 128, n_slots: int = 8) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.n_slots = int(n_slots)
        self.anchors = nn.Parameter(torch.empty(n_slots, latent_dim))
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(hidden_size, latent_dim, bias=False)
        self.value = nn.Linear(hidden_size, latent_dim, bias=False)
        self.output = nn.Linear(latent_dim, latent_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.anchors, std=0.02)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.normal_(self.output.weight, std=1e-3)

    def forward(
        self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [batch, sequence, hidden_size]")
        batch = hidden.shape[0]
        anchors = self.anchors.unsqueeze(0).expand(batch, -1, -1)
        normalized = self.hidden_norm(hidden.float())
        query = self.query(anchors)
        key = self.key(normalized)
        value = self.value(normalized)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.latent_dim)
        if attention_mask is not None:
            if attention_mask.shape != hidden.shape[:2]:
                raise ValueError("attention_mask must match hidden sequence")
            scores = scores.masked_fill(~attention_mask.bool().unsqueeze(1), float("-inf"))
        read = torch.softmax(scores, dim=-1) @ value
        return anchors + self.output(read)


@dataclass
class ResidualFlowOutput:
    state: torch.Tensor
    states: tuple[torch.Tensor, ...]
    updates: tuple[torch.Tensor, ...]
    magnitudes: torch.Tensor
    update_ratios: torch.Tensor
    state_update_ratios: torch.Tensor
    endpoint_update_ratios: torch.Tensor
    initial_update_ratio: torch.Tensor
    trust_penalty: torch.Tensor
    endpoint_reference_available: bool


class SharedResidualFlow(nn.Module):
    """Shared serial flow with direction/magnitude factorization and no state projection."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        context_dim: int,
        n_slots: int = 8,
        max_steps: int = 4,
        hidden_dim: int = 512,
        trust_max: float = 0.5,
        trust_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_slots = int(n_slots)
        self.max_steps = int(max_steps)
        self.trust_max = float(trust_max)
        self.trust_weight = float(trust_weight)
        self.state_norm = nn.RMSNorm(latent_dim)
        self.context_projection = nn.Linear(context_dim, latent_dim, bias=False)
        self.step_embedding = nn.Embedding(max_steps, latent_dim)
        self.input_projection = nn.Linear(3 * latent_dim, hidden_dim)
        self.innovation = nn.Linear(hidden_dim, latent_dim)
        self.direction_norm = nn.RMSNorm(latent_dim)
        self.magnitude = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.innovation.weight, std=1e-3)
        nn.init.zeros_(self.innovation.bias)
        nn.init.zeros_(self.magnitude.weight)
        nn.init.constant_(self.magnitude.bias, -4.0)
        nn.init.ones_(self.direction_norm.weight)

    def step(
        self, state: torch.Tensor, context: torch.Tensor, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.max_steps:
            raise ValueError(f"step index violates loop cap {self.max_steps}")
        projected_context = self.context_projection(context).unsqueeze(1).expand_as(state)
        step_feature = self.step_embedding.weight[index].view(1, 1, -1).expand_as(state)
        features = F.silu(
            self.input_projection(
                torch.cat([self.state_norm(state), projected_context, step_feature], dim=-1)
            )
        )
        raw = self.innovation(features)
        direction = self.direction_norm(raw)
        magnitude = F.softplus(self.magnitude(features).mean(dim=1)).squeeze(-1)
        update = magnitude[:, None, None] * direction
        ratio = _rms(update).mean(dim=1) / _rms(state).mean(dim=1).clamp_min(1e-6)
        return state + update, update, magnitude, ratio

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        *,
        steps: int,
        target_state: Optional[torch.Tensor] = None,
        apply_trust_penalty: bool = False,
    ) -> ResidualFlowOutput:
        if state.shape[1:] != (self.n_slots, self.latent_dim):
            raise ValueError("state does not match registered scratch geometry")
        if steps < 0 or steps > self.max_steps:
            raise ValueError(f"requested steps violate loop cap {self.max_steps}")
        if target_state is not None and target_state.shape != state.shape:
            raise ValueError("target_state must match the registered scratch geometry")
        if apply_trust_penalty and target_state is None:
            raise ValueError("target_state is required when the trust penalty is active")
        current = state
        states = [state]
        updates: list[torch.Tensor] = []
        magnitudes: list[torch.Tensor] = []
        state_ratios: list[torch.Tensor] = []
        endpoint_ratios: list[torch.Tensor] = []
        endpoint_rms = (
            _rms(target_state.detach()).mean(dim=1) if target_state is not None else None
        )
        for index in range(steps):
            current, update, magnitude, state_ratio = self.step(current, context, index)
            states.append(current)
            updates.append(update)
            magnitudes.append(magnitude)
            state_ratios.append(state_ratio)
            if endpoint_rms is not None:
                update_rms = _rms(update).mean(dim=1)
                state_rms = _rms(states[-2]).mean(dim=1)
                denominator = torch.maximum(state_rms, endpoint_rms) + 1e-6
                endpoint_ratios.append(update_rms / denominator)
        if steps:
            magnitude_tensor = torch.stack(magnitudes, dim=1)
            state_ratio_tensor = torch.stack(state_ratios, dim=1)
            endpoint_ratio_tensor = (
                torch.stack(endpoint_ratios, dim=1)
                if endpoint_ratios
                else state.new_zeros((state.shape[0], steps))
            )
            ratio_tensor = (
                endpoint_ratio_tensor if target_state is not None else state_ratio_tensor
            )
            initial_ratio = ratio_tensor[:, 0]
            trust = (
                F.relu(endpoint_ratio_tensor - self.trust_max).square().sum(dim=1).mean()
                if apply_trust_penalty
                else state.new_zeros(())
            )
        else:
            magnitude_tensor = state.new_zeros((state.shape[0], 0))
            ratio_tensor = state.new_zeros((state.shape[0], 0))
            state_ratio_tensor = state.new_zeros((state.shape[0], 0))
            endpoint_ratio_tensor = state.new_zeros((state.shape[0], 0))
            initial_ratio = state.new_zeros((state.shape[0],))
            trust = state.new_zeros(())
        return ResidualFlowOutput(
            state=current,
            states=tuple(states),
            updates=tuple(updates),
            magnitudes=magnitude_tensor,
            update_ratios=ratio_tensor,
            state_update_ratios=state_ratio_tensor,
            endpoint_update_ratios=endpoint_ratio_tensor,
            initial_update_ratio=initial_ratio,
            trust_penalty=self.trust_weight * trust,
            endpoint_reference_available=target_state is not None,
        )


@dataclass
class BridgeOutput:
    hidden: torch.Tensor
    delta: torch.Tensor
    gate: torch.Tensor
    rho: torch.Tensor
    realized_writeback_ratio: torch.Tensor
    position_zero_gate_closed: bool
    position_gate: Optional[torch.Tensor] = None


class AnchoredBridge(nn.Module):
    """Bounded cross-attention writeback anchored to the frozen prelude state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_dim: int = 128,
        max_steps: int = 4,
        rms_cap: float = 0.550893,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.max_steps = int(max_steps)
        self.rms_cap = float(rms_cap)
        self.eps = float(eps)
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.scratch_norm = nn.RMSNorm(latent_dim)
        self.query = nn.Linear(hidden_size, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.output_projection = nn.Linear(latent_dim, hidden_size, bias=False)
        self.gate_logits = nn.Parameter(torch.full((max_steps,), -4.0))
        rho_logit = math.log(0.95 / 0.05)
        self.rho_logits = nn.Parameter(torch.full((max_steps,), rho_logit))
        nn.init.normal_(self.output_projection.weight, std=1e-3)

    def forward(
        self,
        *,
        h0: torch.Tensor,
        previous: torch.Tensor,
        scratch: torch.Tensor,
        loop_index: int,
        active: bool = True,
    ) -> BridgeOutput:
        if loop_index < 0 or loop_index >= self.max_steps:
            raise ValueError(f"loop index violates loop cap {self.max_steps}")
        if h0.shape != previous.shape or h0.ndim != 3:
            raise ValueError("h0 and previous must share [batch, sequence, hidden]")
        if not active:
            zero = previous.new_zeros(previous.shape)
            return BridgeOutput(
                hidden=previous,
                delta=zero,
                gate=previous.new_zeros(()),
                rho=previous.new_ones(()),
                realized_writeback_ratio=previous.new_zeros(previous.shape[:2]),
                position_zero_gate_closed=True,
            )
        query = self.query(self.hidden_norm(h0.float()))
        normalized_scratch = self.scratch_norm(scratch.float())
        key = self.key(normalized_scratch)
        value = self.value(normalized_scratch)
        attention = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(self.latent_dim), dim=-1)
        delta = self.output_projection(attention @ value)
        delta_rms = _rms(delta).unsqueeze(-1)
        reference = _rms(h0).clamp_max(self.rms_cap).unsqueeze(-1).detach()
        delta = delta / delta_rms.clamp_min(self.eps) * reference
        gate = torch.sigmoid(self.gate_logits[loop_index])
        rho = torch.sigmoid(self.rho_logits[loop_index])
        gate_mask = torch.ones_like(delta[..., :1])
        gate_mask[:, 0] = 0
        writeback = gate * gate_mask * delta
        hidden = h0 + rho * (previous - h0) + writeback
        ratio = _rms(writeback) / _rms(h0).clamp_min(self.eps)
        return BridgeOutput(
            hidden=hidden,
            delta=delta,
            gate=gate,
            rho=rho,
            realized_writeback_ratio=ratio,
            position_zero_gate_closed=bool(torch.equal(gate_mask[:, 0], torch.zeros_like(gate_mask[:, 0]))),
        )


class Phase3PerPositionAnchoredBridge(AnchoredBridge):
    """Phase 3 gate extension with scalar-compatible per-position control."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_dim: int = 128,
        control_dim: int = 32,
        max_steps: int = 4,
        rms_cap: float = 0.550893,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            max_steps=max_steps,
            rms_cap=rms_cap,
            eps=eps,
        )
        self.control_dim = int(control_dim)
        self.gate_hidden = nn.Linear(hidden_size, 1, bias=False)
        self.gate_scratch = nn.Linear(latent_dim, 1, bias=False)
        self.gate_control = nn.Linear(control_dim, 1, bias=False)
        nn.init.zeros_(self.gate_hidden.weight)
        nn.init.zeros_(self.gate_scratch.weight)
        nn.init.zeros_(self.gate_control.weight)

    def forward(
        self,
        *,
        h0: torch.Tensor,
        previous: torch.Tensor,
        scratch: torch.Tensor,
        control_state: Optional[torch.Tensor],
        loop_index: int,
        active: bool = True,
    ) -> BridgeOutput:
        if loop_index < 0 or loop_index >= self.max_steps:
            raise ValueError(f"loop index violates loop cap {self.max_steps}")
        if h0.shape != previous.shape or h0.ndim != 3:
            raise ValueError("h0 and previous must share [batch, sequence, hidden]")
        if control_state is not None and control_state.shape != (h0.shape[0], self.control_dim):
            raise ValueError("control state must share the bridge batch and registered control width")
        if not active:
            zero = previous.new_zeros(previous.shape)
            return BridgeOutput(
                hidden=previous,
                delta=zero,
                gate=previous.new_zeros(()),
                rho=previous.new_ones(()),
                realized_writeback_ratio=previous.new_zeros(previous.shape[:2]),
                position_zero_gate_closed=True,
                position_gate=previous.new_zeros((*previous.shape[:2], 1)),
            )
        query = self.query(self.hidden_norm(h0.float()))
        normalized_scratch = self.scratch_norm(scratch.float())
        key = self.key(normalized_scratch)
        value = self.value(normalized_scratch)
        attention = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(self.latent_dim), dim=-1)
        attended_scratch = attention @ value
        delta = self.output_projection(attended_scratch)
        delta_rms = _rms(delta).unsqueeze(-1)
        reference = _rms(h0).clamp_max(self.rms_cap).unsqueeze(-1).detach()
        delta = delta / delta_rms.clamp_min(self.eps) * reference
        if control_state is None:
            control_state = h0.new_zeros((h0.shape[0], self.control_dim))
        gate_logit = (
            self.gate_logits[loop_index]
            + self.gate_hidden(self.hidden_norm(previous.float()))
            + self.gate_scratch(attended_scratch)
            + self.gate_control(control_state.float()).unsqueeze(1)
        )
        position_gate = torch.sigmoid(gate_logit)
        rho = torch.sigmoid(self.rho_logits[loop_index])
        gate_mask = torch.ones_like(delta[..., :1])
        gate_mask[:, 0] = 0
        position_gate = position_gate * gate_mask
        writeback = position_gate * delta
        hidden = h0 + rho * (previous - h0) + writeback
        ratio = _rms(writeback) / _rms(h0).clamp_min(self.eps)
        gate = (
            position_gate[:, 1:].mean()
            if position_gate.shape[1] > 1
            else position_gate.new_zeros(())
        )
        return BridgeOutput(
            hidden=hidden,
            delta=delta,
            gate=gate,
            rho=rho,
            realized_writeback_ratio=ratio,
            position_zero_gate_closed=bool(
                torch.equal(gate_mask[:, 0], torch.zeros_like(gate_mask[:, 0]))
            ),
            position_gate=position_gate,
        )


class ControlState(nn.Module):
    """Deployable recurrent control state with no oracle-distance input."""

    def __init__(self, *, latent_dim: int = 128, control_dim: int = 32) -> None:
        super().__init__()
        self.control_dim = int(control_dim)
        self.position_embedding = nn.Embedding(5, 8)
        self.cell = nn.GRUCell(latent_dim + 8 + 3, control_dim)

    def forward(
        self,
        *,
        scratch: torch.Tensor,
        previous: Optional[torch.Tensor],
        innovation_norm: torch.Tensor,
        student_entropy: torch.Tensor,
        top2_margin: torch.Tensor,
        position_bucket: torch.Tensor,
    ) -> torch.Tensor:
        batch = scratch.shape[0]
        if previous is None:
            previous = scratch.new_zeros((batch, self.control_dim))
        position = self.position_embedding(position_bucket.long().clamp(0, 4))
        scalars = torch.stack([innovation_norm, student_entropy, top2_margin], dim=-1).float()
        features = torch.cat([scratch.float().mean(dim=1), scalars, position], dim=-1)
        return self.cell(features, previous.float())


@dataclass
class DraftHeadOutput:
    logits: torch.Tensor
    delta_logits: torch.Tensor
    write_gates: torch.Tensor


class ResidualDraftHead(nn.Module):
    """Per-horizon low-rank residual readout tied to the drafter vocabulary embedding."""

    def __init__(
        self,
        *,
        tied_embedding: nn.Embedding,
        latent_dim: int = 128,
        control_dim: int = 32,
        hidden_size: int,
        rank: int = 64,
        horizons: int = 4,
    ) -> None:
        super().__init__()
        if tied_embedding.embedding_dim != hidden_size:
            raise ValueError("tied embedding width must equal hidden_size")
        self.tied_embedding = tied_embedding
        self.horizons = int(horizons)
        self.down = nn.ModuleList([nn.Linear(latent_dim, rank, bias=False) for _ in range(horizons)])
        self.up = nn.ModuleList([nn.Linear(rank, hidden_size, bias=False) for _ in range(horizons)])
        self.write_gate = nn.Linear(control_dim, horizons)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -3.5)

    def forward(
        self,
        *,
        previous_logits: torch.Tensor,
        scratch: torch.Tensor,
        control_state: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
    ) -> DraftHeadOutput:
        if previous_logits.shape[1] != self.horizons:
            raise ValueError("previous logits do not match registered horizon count")
        pooled = scratch.mean(dim=1)
        hidden_updates = torch.stack(
            [up(F.silu(down(pooled))) for down, up in zip(self.down, self.up)], dim=1
        )
        if candidate_ids is None:
            delta = hidden_updates @ self.tied_embedding.weight.T
        else:
            if candidate_ids.shape[:2] != previous_logits.shape[:2]:
                raise ValueError("candidate ids must match batch and horizon dimensions")
            if candidate_ids.shape != previous_logits.shape:
                raise ValueError("candidate ids and sparse logits must share shape")
            candidate_embeddings = self.tied_embedding(
                candidate_ids.clamp(min=0, max=self.tied_embedding.num_embeddings - 1)
            )
            delta = torch.einsum("bhd,bhcd->bhc", hidden_updates, candidate_embeddings)
        gates = torch.sigmoid(self.write_gate(control_state))
        return DraftHeadOutput(
            logits=previous_logits + gates.unsqueeze(-1) * delta,
            delta_logits=delta,
            write_gates=gates,
        )


@dataclass
class Phase2StudentOutput:
    loss: None
    scratch: torch.Tensor
    flow: ResidualFlowOutput
    hidden: torch.Tensor
    logits: torch.Tensor
    control_state: torch.Tensor
    control_read: torch.Tensor
    bridge: BridgeOutput
    draft: DraftHeadOutput


class Phase2StudentModules(nn.Module):
    """Loss-free build surface used to assert the complete student plumbing."""

    def __init__(
        self,
        *,
        tied_embedding: nn.Embedding,
        hidden_size: int,
        latent_dim: int = 128,
        n_slots: int = 8,
        control_dim: int = 32,
        draft_rank: int = 64,
        max_steps: int = 4,
        rms_cap: float = 0.550893,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.initializer = ScratchpadInitializer(
            hidden_size=hidden_size, latent_dim=latent_dim, n_slots=n_slots
        )
        self.flow = SharedResidualFlow(
            latent_dim=latent_dim,
            context_dim=hidden_size,
            n_slots=n_slots,
            max_steps=max_steps,
        )
        self.bridge = AnchoredBridge(
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            max_steps=max_steps,
            rms_cap=rms_cap,
        )
        self.control = ControlState(latent_dim=latent_dim, control_dim=control_dim)
        self.draft = ResidualDraftHead(
            tied_embedding=tied_embedding,
            latent_dim=latent_dim,
            control_dim=control_dim,
            hidden_size=hidden_size,
            rank=draft_rank,
            horizons=4,
        )

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        previous_logits: torch.Tensor,
        steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_bucket: Optional[torch.Tensor] = None,
        target_scratch: Optional[torch.Tensor] = None,
        apply_trust_penalty: bool = False,
        candidate_ids: Optional[torch.Tensor] = None,
    ) -> Phase2StudentOutput:
        if steps < 0 or steps > self.max_steps:
            raise ValueError(f"requested steps violate loop cap {self.max_steps}")
        scratch0 = self.initializer(hidden, attention_mask)
        context = hidden.float().mean(dim=1)
        flow = self.flow(
            scratch0,
            context,
            steps=steps,
            target_state=target_scratch,
            apply_trust_penalty=apply_trust_penalty,
        )
        if flow.updates:
            innovation_norm = _rms(flow.updates[-1]).mean(dim=1)
        else:
            innovation_norm = hidden.new_zeros((hidden.shape[0],))
        if position_bucket is None:
            position_bucket = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        control = self.control(
            scratch=flow.state,
            previous=None,
            innovation_norm=innovation_norm,
            student_entropy=hidden.new_zeros((hidden.shape[0],)),
            top2_margin=hidden.new_zeros((hidden.shape[0],)),
            position_bucket=position_bucket,
        )
        bridge = self.bridge(
            h0=hidden,
            previous=hidden,
            scratch=flow.state,
            loop_index=max(0, steps - 1),
            active=steps > 0,
        )
        if steps > 0:
            draft = self.draft(
                previous_logits=previous_logits,
                scratch=flow.state,
                control_state=control,
                candidate_ids=candidate_ids,
            )
        else:
            zero_delta = torch.zeros_like(previous_logits)
            draft = DraftHeadOutput(
                logits=previous_logits,
                delta_logits=zero_delta,
                write_gates=previous_logits.new_zeros(previous_logits.shape[:2]),
            )
        return Phase2StudentOutput(
            loss=None,
            scratch=flow.state,
            flow=flow,
            hidden=bridge.hidden,
            logits=draft.logits,
            control_state=control,
            control_read=control,
            bridge=bridge,
            draft=draft,
        )


class Phase3StudentModules(Phase2StudentModules):
    """Phase 3 sidecar with a separately versioned per-position bridge gate."""

    def __init__(
        self,
        *,
        tied_embedding: nn.Embedding,
        hidden_size: int,
        latent_dim: int = 128,
        n_slots: int = 8,
        control_dim: int = 32,
        draft_rank: int = 64,
        max_steps: int = 4,
        rms_cap: float = 0.550893,
    ) -> None:
        super().__init__(
            tied_embedding=tied_embedding,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            n_slots=n_slots,
            control_dim=control_dim,
            draft_rank=draft_rank,
            max_steps=max_steps,
            rms_cap=rms_cap,
        )
        self.bridge = Phase3PerPositionAnchoredBridge(
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            control_dim=control_dim,
            max_steps=max_steps,
            rms_cap=rms_cap,
        )

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        previous_logits: torch.Tensor,
        steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_bucket: Optional[torch.Tensor] = None,
        target_scratch: Optional[torch.Tensor] = None,
        apply_trust_penalty: bool = False,
        candidate_ids: Optional[torch.Tensor] = None,
    ) -> Phase2StudentOutput:
        if steps < 0 or steps > self.max_steps:
            raise ValueError(f"requested steps violate loop cap {self.max_steps}")
        scratch0 = self.initializer(hidden, attention_mask)
        context = hidden.float().mean(dim=1)
        flow = self.flow(
            scratch0,
            context,
            steps=steps,
            target_state=target_scratch,
            apply_trust_penalty=apply_trust_penalty,
        )
        if flow.updates:
            innovation_norm = _rms(flow.updates[-1]).mean(dim=1)
        else:
            innovation_norm = hidden.new_zeros((hidden.shape[0],))
        if position_bucket is None:
            position_bucket = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        control = self.control(
            scratch=flow.state,
            previous=None,
            innovation_norm=innovation_norm,
            student_entropy=hidden.new_zeros((hidden.shape[0],)),
            top2_margin=hidden.new_zeros((hidden.shape[0],)),
            position_bucket=position_bucket,
        )
        bridge = self.bridge(
            h0=hidden,
            previous=hidden,
            scratch=flow.state,
            control_state=control,
            loop_index=max(0, steps - 1),
            active=steps > 0,
        )
        if steps > 0:
            draft = self.draft(
                previous_logits=previous_logits,
                scratch=flow.state,
                control_state=control,
                candidate_ids=candidate_ids,
            )
        else:
            zero_delta = torch.zeros_like(previous_logits)
            draft = DraftHeadOutput(
                logits=previous_logits,
                delta_logits=zero_delta,
                write_gates=previous_logits.new_zeros(previous_logits.shape[:2]),
            )
        return Phase2StudentOutput(
            loss=None,
            scratch=flow.state,
            flow=flow,
            hidden=bridge.hidden,
            logits=draft.logits,
            control_state=control,
            control_read=control,
            bridge=bridge,
            draft=draft,
        )
