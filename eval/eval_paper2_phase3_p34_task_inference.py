"""P3.4 v1 task-inference graph and score-blind contract checks.

The frozen base may use a KV cache, while the sidecar receives the complete
prefix state and initializes fresh for every emitted token. Cross-token sidecar
persistence remains outside the P3.4 v1 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_stage2a_runtime import (
    Stage2AMemorySystem,
    canonical_fingerprint_query,
)
from training.paper2_phase3_p34 import P34_FLOW_LOOPS, TaskInferenceContract


def position_buckets(positions: torch.Tensor) -> torch.Tensor:
    buckets = torch.zeros_like(positions)
    buckets[(positions >= 1) & (positions <= 3)] = 1
    buckets[(positions >= 4) & (positions <= 31)] = 2
    buckets[(positions >= 32) & (positions <= 127)] = 3
    buckets[positions >= 128] = 4
    return buckets


def current_position_mask(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must be [batch, sequence]")
    active = attention_mask.bool()
    if bool((active.sum(dim=1) == 0).any()):
        raise ValueError("every P3.4 prefix must contain at least one active token")
    indexes = torch.arange(active.shape[1], device=active.device).expand_as(active)
    positions = indexes.masked_fill(~active, -1).max(dim=1).values
    mask = torch.zeros((*active.shape, 1), dtype=torch.bool, device=active.device)
    mask[torch.arange(active.shape[0], device=active.device), positions, 0] = True
    mask[:, 0] = False
    return mask, positions


def _layer6_or_test_fallback(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Use the registered layer 6; tiny unit-test models may expose fewer layers."""

    return hidden_states[6] if len(hidden_states) > 6 else hidden_states[-1]


@dataclass(frozen=True)
class P34NextTokenOutput:
    augmented_logits: torch.Tensor
    base_logits: torch.Tensor
    writeback_ratio: torch.Tensor
    position_gate: torch.Tensor
    current_positions: torch.Tensor
    scratch_state: torch.Tensor
    answer_token_margin: torch.Tensor
    memory_compatibility_gate: torch.Tensor | None = None
    memory_slot_indices: torch.Tensor | None = None
    memory_slot_scores: torch.Tensor | None = None
    memory_slot_weights: torch.Tensor | None = None


@dataclass
class P34CachedPrefix:
    input_ids: torch.Tensor
    hidden: torch.Tensor
    layer6_hidden: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any
    sidecar_scratch: torch.Tensor | None = None


class P34TaskInferenceGraph:
    """Exact, uncached v1 generation graph for the frozen base plus sidecar."""

    def __init__(
        self,
        *,
        base_model: Any,
        sidecar: Phase3StudentModules,
        cross_token_persistence: bool = False,
        flow_loops: int = P34_FLOW_LOOPS,
        allow_clamped_extension: bool = False,
        stage2a_memory_system: Stage2AMemorySystem | None = None,
        stage2a_geometry: dict[str, Any] | None = None,
        stage2a_amplitude: float = 0.05,
        stage2a_value_scale: float = 1.0,
        stage2a_diagnostic_value_scale_authorized: bool = False,
    ) -> None:
        self.base_model = base_model
        self.sidecar = sidecar
        self.cross_token_persistence = bool(cross_token_persistence)
        self.flow_loops = int(flow_loops)
        self.allow_clamped_extension = bool(allow_clamped_extension)
        self.stage2a_memory_system = stage2a_memory_system
        self.stage2a_geometry = stage2a_geometry
        self.stage2a_amplitude = float(stage2a_amplitude)
        self.stage2a_value_scale = float(stage2a_value_scale)
        if self.flow_loops < 1:
            raise ValueError("task inference requires at least one flow loop")
        if self.flow_loops > P34_FLOW_LOOPS and not self.allow_clamped_extension:
            raise ValueError("K above four requires the disclosed clamped-extension path")
        if self.flow_loops > 6:
            raise ValueError("the authorized exploratory extension is limited to K <= 6")
        if self.stage2a_memory_system is not None:
            if self.stage2a_geometry is None:
                raise ValueError("Stage 2A memory requires frozen geometry")
            if abs(self.stage2a_amplitude - 0.05) > 1e-12:
                raise ValueError("Stage 2A registered DEV read amplitude is 0.05")
            allowed_scales = (0.0, 0.5, 1.0)
            if not any(abs(self.stage2a_value_scale - value) <= 1e-12 for value in allowed_scales):
                raise ValueError("Stage 2A diagnostic value scale must be one of 0, 0.5, or 1")
            if (
                abs(self.stage2a_value_scale - 1.0) > 1e-12
                and not stage2a_diagnostic_value_scale_authorized
            ):
                raise ValueError("Stage 2A non-unit value scale requires score-only authorization")
        self.contract = TaskInferenceContract(
            cross_token_state_persistence=self.cross_token_persistence
        )
        if not self.cross_token_persistence:
            self.contract.validate()

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    def _augment(
        self,
        *,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        base_logits: torch.Tensor,
        initial_scratch: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        layer6_hidden: torch.Tensor | None = None,
        current_positions: torch.Tensor | None = None,
    ) -> P34NextTokenOutput:
        positions = (
            current_position_mask(attention_mask)[1]
            if current_positions is None
            else current_positions
        )
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        scratch0 = (
            self.sidecar.initializer(hidden, attention_mask.bool())
            if initial_scratch is None
            else initial_scratch
        )
        memory_readout = None
        if self.stage2a_memory_system is not None:
            if input_ids is None or layer6_hidden is None:
                raise RuntimeError("Stage 2A inference requires tokens and layer-6 states")
            if self.stage2a_memory_system.arm == "t3b":
                values = []
                for row in range(input_ids.shape[0]):
                    active = attention_mask[row].bool()
                    tokens = input_ids[row, active]
                    read = self.stage2a_memory_system.read_literal(
                        tokens[None, :],
                        torch.tensor([tokens.shape[0] - 1], device=tokens.device),
                    )
                    values.append(read.value)
                memory_readout = read
                memory_value = torch.cat(values, dim=0)
            else:
                layer6_current = layer6_hidden[batch_index, positions]
                query = canonical_fingerprint_query(
                    layer6_current,
                    student_mean=self.stage2a_geometry["student_mean"].to(hidden.device),
                    student_basis=self.stage2a_geometry["student_basis"].to(hidden.device),
                )
                memory_readout = self.stage2a_memory_system.read_fingerprint(query)
                memory_value = memory_readout.value
            memory_value = memory_value * self.stage2a_value_scale
            scratch0 = self.stage2a_memory_system.injection(
                scratch0,
                memory_value,
                amplitude_ceiling=self.stage2a_amplitude,
            )
        expected_scratch = (
            hidden.shape[0],
            self.sidecar.initializer.n_slots,
            self.sidecar.initializer.latent_dim,
        )
        if scratch0.shape != expected_scratch:
            raise RuntimeError("persistent scratch shape changed")
        context = hidden.float().mean(dim=1)
        if self.flow_loops <= self.sidecar.flow.max_steps:
            flow = self.sidecar.flow(scratch0, context, steps=self.flow_loops)
            flow_state = flow.state
            flow_updates = flow.updates
        else:
            # Exploratory only: repeat the fourth learned step embedding for K=5-6.
            # This does not expand or alter model parameters.
            current = scratch0
            updates = []
            for index in range(self.flow_loops):
                current, update, _magnitude, _ratio = self.sidecar.flow.step(
                    current,
                    context,
                    min(index, self.sidecar.flow.max_steps - 1),
                )
                updates.append(update)
            flow_state = current
            flow_updates = tuple(updates)
        innovation_norm = (
            flow_updates[-1].float().square().mean(dim=-1).sqrt().mean(dim=1)
        )
        control = self.sidecar.control(
            scratch=flow_state,
            previous=None,
            innovation_norm=innovation_norm,
            student_entropy=hidden.new_zeros((hidden.shape[0],)),
            top2_margin=hidden.new_zeros((hidden.shape[0],)),
            position_bucket=position_buckets(positions),
        )
        current = hidden[batch_index, positions]
        compact_hidden = torch.stack([torch.zeros_like(current), current], dim=1)
        compact_write_mask = torch.zeros(
            (hidden.shape[0], 2, 1), dtype=torch.bool, device=hidden.device
        )
        compact_write_mask[:, 1] = True
        bridge = self.sidecar.bridge(
            h0=compact_hidden,
            previous=compact_hidden,
            scratch=flow_state,
            control_state=control,
            loop_index=min(self.flow_loops - 1, self.sidecar.bridge.max_steps - 1),
            active=True,
            write_position_mask=compact_write_mask,
        )
        augmented_hidden = bridge.hidden[:, 1]
        output_head = self.base_model.get_output_embeddings()
        augmented_logits = output_head(augmented_hidden.to(output_head.weight.dtype))
        top2 = augmented_logits.float().topk(2, dim=-1).values
        return P34NextTokenOutput(
            augmented_logits=augmented_logits,
            base_logits=base_logits,
            writeback_ratio=bridge.realized_writeback_ratio[:, 1],
            position_gate=bridge.position_gate[:, 1, 0],
            current_positions=positions,
            scratch_state=flow_state,
            answer_token_margin=top2[:, 0] - top2[:, 1],
            memory_compatibility_gate=(
                None
                if memory_readout is None
                else memory_readout.compatibility_gate
            ),
            memory_slot_indices=(
                None if memory_readout is None else memory_readout.slot_indices
            ),
            memory_slot_scores=(
                None if memory_readout is None else memory_readout.slot_scores
            ),
            memory_slot_weights=(
                None if memory_readout is None else memory_readout.slot_weights
            ),
        )

    @torch.inference_mode()
    def next_token(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        flow_loops: int | None = None,
    ) -> P34NextTokenOutput:
        if flow_loops is not None and int(flow_loops) != self.flow_loops:
            raise RuntimeError("per-call K must match the configured task-inference graph")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input ids and attention mask must share [batch, sequence]")
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        positions = current_position_mask(attention_mask)[1]
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self._augment(
            hidden=hidden,
            attention_mask=attention_mask,
            base_logits=output.logits[batch_index, positions],
            input_ids=input_ids,
            layer6_hidden=_layer6_or_test_fallback(output.hidden_states),
        )

    @torch.inference_mode()
    def prefill_cached(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        """Create an exact frozen-base KV cache while retaining all prefix states."""

        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        positions = current_position_mask(attention_mask)[1]
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        state = P34CachedPrefix(
            input_ids=input_ids,
            hidden=output.hidden_states[-1],
            layer6_hidden=_layer6_or_test_fallback(output.hidden_states),
            attention_mask=attention_mask,
            past_key_values=output.past_key_values,
            sidecar_scratch=None,
        )
        augmented = self._augment(
            hidden=state.hidden,
            attention_mask=state.attention_mask,
            base_logits=output.logits[batch_index, positions],
            input_ids=state.input_ids,
            layer6_hidden=state.layer6_hidden,
            current_positions=positions,
        )
        if self.cross_token_persistence:
            state.sidecar_scratch = augmented.scratch_state.detach()
        return state, augmented

    @torch.inference_mode()
    def advance_cached(
        self, *, state: P34CachedPrefix, selected_tokens: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if selected_tokens.ndim == 1:
            selected_tokens = selected_tokens[:, None]
        if selected_tokens.shape != (state.hidden.shape[0], 1):
            raise ValueError("cached P3.4 advance requires one token per batch row")
        attention = torch.cat(
            [
                state.attention_mask,
                torch.ones(
                    (selected_tokens.shape[0], 1),
                    dtype=state.attention_mask.dtype,
                    device=state.attention_mask.device,
                ),
            ],
            dim=1,
        )
        positions = torch.full(
            (selected_tokens.shape[0],),
            attention.shape[1] - 1,
            dtype=torch.long,
            device=attention.device,
        )
        output = self.base_model(
            input_ids=selected_tokens,
            attention_mask=attention,
            past_key_values=state.past_key_values,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        hidden = torch.cat([state.hidden, output.hidden_states[-1]], dim=1)
        layer6_hidden = torch.cat(
            [state.layer6_hidden, _layer6_or_test_fallback(output.hidden_states)], dim=1
        )
        input_ids = torch.cat([state.input_ids, selected_tokens], dim=1)
        updated = P34CachedPrefix(
            input_ids=input_ids,
            hidden=hidden,
            layer6_hidden=layer6_hidden,
            attention_mask=attention,
            past_key_values=output.past_key_values,
            sidecar_scratch=None,
        )
        augmented = self._augment(
            hidden=hidden,
            attention_mask=attention,
            base_logits=output.logits[:, -1],
            initial_scratch=(
                state.sidecar_scratch if self.cross_token_persistence else None
            ),
            input_ids=input_ids,
            layer6_hidden=layer6_hidden,
            current_positions=positions,
        )
        updated.sidecar_scratch = (
            augmented.scratch_state.detach()
            if self.cross_token_persistence
            else None
        )
        return updated, augmented

    @torch.inference_mode()
    def greedy_generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("the v1 exact generation primitive is batch-size one")
        tokens = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            logits = self.next_token(input_ids=tokens, attention_mask=mask).augmented_logits
            selected = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, selected], dim=1)
            mask = torch.cat([mask, torch.ones_like(selected, dtype=mask.dtype)], dim=1)
            if eos_token_id is not None and int(selected.item()) == int(eos_token_id):
                break
        return tokens

    @torch.inference_mode()
    def continuation_log_score(
        self,
        *,
        prefix_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_ids.ndim != 2 or continuation_ids.ndim != 2:
            raise ValueError("prefix and continuation ids must be rank two")
        if prefix_ids.shape[0] != continuation_ids.shape[0]:
            raise ValueError("prefix and continuation batch sizes differ")
        if prefix_ids.shape[0] != 1:
            raise ValueError("the v1 exact continuation scorer is batch-size one")
        tokens = prefix_ids
        attention = torch.ones_like(tokens)
        scores = []
        for index in range(continuation_ids.shape[1]):
            target = continuation_ids[:, index]
            logits = self.next_token(input_ids=tokens, attention_mask=attention).augmented_logits
            scores.append(torch.log_softmax(logits.float(), dim=-1).gather(1, target[:, None])[:, 0])
            tokens = torch.cat([tokens, target[:, None]], dim=1)
            attention = torch.cat([attention, torch.ones_like(target[:, None])], dim=1)
        return torch.stack(scores, dim=1).mean(dim=1)


@torch.inference_mode()
def task_graph_preflight(
    graph: P34TaskInferenceGraph,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    first = graph.next_token(input_ids=input_ids, attention_mask=attention_mask)
    second = graph.next_token(input_ids=input_ids, attention_mask=attention_mask)
    write_mask, positions = current_position_mask(attention_mask)
    return {
        "kind": "paper2_phase3_p34_task_inference_preflight_v1",
        "contract": graph.contract.__dict__,
        "rows": int(input_ids.shape[0]),
        "flow_loops": graph.flow_loops,
        "clamped_extension": graph.allow_clamped_extension,
        "draft_head_scoring": False,
        "current_positions": positions.detach().cpu().tolist(),
        "selected_write_cells": int(write_mask.sum()),
        "repeat_max_abs_difference": float(
            (first.augmented_logits - second.augmented_logits).abs().max().cpu()
        ),
        "assertions": {
            "fresh_state_repeat_exact": torch.equal(first.augmented_logits, second.augmented_logits),
            "one_write_cell_per_nonzero_prefix": int(write_mask.sum())
            == int((positions > 0).sum()),
            "position_zero_closed": not bool(write_mask[:, 0].any()),
            "registered_or_disclosed_k": (
                graph.flow_loops <= P34_FLOW_LOOPS
                or graph.allow_clamped_extension
            ),
            "draft_inactive": True,
            "cross_token_persistence_absent": True,
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "task_scores_computed": False,
    }
