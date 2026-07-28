"""COCONUT-style horizontal latent feedback over recurrent-depth Qwen."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .lora import LoRALinear
from .recurrent_wrapper import RecurrentQwenForCausalLM, RecurrentQwenOutput


class HorizontalIdentityBridge(nn.Module):
    """Learnable ``I + delta`` map with an exactly zero-initialized delta."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.delta = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.delta.weight.zero_()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        work = states.to(dtype=self.delta.weight.dtype)
        return (work + self.delta(work)).to(dtype=states.dtype)


@dataclass
class CompositeCoconutOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    recurrent_output: RecurrentQwenOutput
    input_embeddings: torch.Tensor
    horizontal_fed_states: tuple[torch.Tensor, ...] = ()
    recurrent_application_states: tuple[tuple[torch.Tensor, ...], ...] = ()
    requested_horizontal_steps: int = 0
    executed_horizontal_steps: int = 0
    vertical_loops: int = 1
    feedback_grid_applications: int = 0
    total_grid_applications: int = 1
    requested_execution_mode: str = "recompute"
    executed_execution_mode: str = "recompute"
    cache_prefix_lengths: tuple[int, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthByAppendOutput:
    """Auditable counters for one transient depth-by-append execution."""

    predictions: torch.Tensor
    requested_append_steps: int
    executed_decision_positions: int
    total_grid_applications: int
    feedback_grid_applications: int
    evicted_slots: int
    eviction_assertions: int
    readout_grid_applications: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    real_logits: Optional[torch.Tensor] = None

    def assert_accounting(self) -> dict[str, Any]:
        expected_feedback = self.requested_append_steps * self.executed_decision_positions
        expected_total = (
            self.executed_decision_positions
            + expected_feedback
            + self.readout_grid_applications
        )
        if self.feedback_grid_applications != expected_feedback:
            raise ValueError(
                f"feedback-grid count {self.feedback_grid_applications} != {expected_feedback}"
            )
        if self.total_grid_applications != expected_total:
            raise ValueError(f"total-grid count {self.total_grid_applications} != {expected_total}")
        expected_evicted = expected_feedback + self.readout_grid_applications
        if self.evicted_slots != expected_evicted:
            raise ValueError(f"evicted-slot count {self.evicted_slots} != {expected_evicted}")
        if self.eviction_assertions != self.executed_decision_positions:
            raise ValueError(
                f"eviction-assertion count {self.eviction_assertions} "
                f"!= {self.executed_decision_positions}"
            )
        return {
            "status": "exact",
            "feedback_grid_applications": expected_feedback,
            "total_grid_applications": expected_total,
            "evicted_slots": expected_evicted,
        }


def rebuild_embedding_slot(
    embeddings: torch.Tensor,
    position: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Replace one sequence slot by reconstruction, never by in-place mutation."""

    index = int(position)
    if embeddings.dim() != 3 or replacement.shape != embeddings[:, index].shape:
        raise ValueError("replacement must match one [batch, hidden] embedding slot")
    return torch.cat(
        [embeddings[:, :index], replacement.unsqueeze(1), embeddings[:, index + 1 :]],
        dim=1,
    )


def _sequence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _control_positions(input_ids: torch.Tensor, latent_token_id: int, horizontal_steps: int) -> list[int]:
    matches = input_ids.eq(int(latent_token_id))
    counts = matches.sum(dim=-1)
    if bool(counts.ne(int(horizontal_steps)).any()):
        raise ValueError(
            "every row must contain exactly horizontal_steps latent placeholders; "
            f"observed={counts.tolist()}, requested={horizontal_steps}"
        )
    rows = [tuple(row.nonzero(as_tuple=False).view(-1).tolist()) for row in matches]
    if len(set(rows)) != 1:
        raise ValueError("one composite batch requires aligned latent positions")
    positions = list(rows[0])
    if positions and positions[0] < 1:
        raise ValueError("a latent placeholder cannot be the first token")
    return positions


def _cache_length(cache: Any) -> int:
    if cache is None or not hasattr(cache, "get_seq_length"):
        return 0
    return int(cache.get_seq_length())


class CoconutRecurrentQwen(nn.Module):
    """Compose horizontal hidden-state feedback with vertical recurrent depth.

    ``horizontal_steps=0`` delegates directly to the registered recurrent
    wrapper. The graph-preserving full-recompute path is the training
    reference. The sliced-cache path is permitted only for one vertical loop
    and without gradient checkpointing until RG-5 establishes equivalence.
    """

    MODES = {"recompute", "sliced_cache"}

    def __init__(
        self,
        recurrent: RecurrentQwenForCausalLM,
        *,
        latent_token_id: int,
    ) -> None:
        super().__init__()
        self.recurrent = recurrent
        hidden_size = int(recurrent.bridge.hidden_size)
        self.horizontal_bridge = HorizontalIdentityBridge(hidden_size)
        base_parameter = next(recurrent.parameters())
        self.horizontal_bridge.to(device=base_parameter.device, dtype=base_parameter.dtype)
        self.latent_token_id = int(latent_token_id)

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        horizontal_steps: int = 0,
        max_loops: int = 1,
        execution_mode: str = "recompute",
        raw_feedback: bool = False,
        horizontal_state_additions: Optional[dict[int, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> CompositeCoconutOutput:
        if input_ids.dim() != 2:
            raise ValueError("composite input_ids must be [batch, sequence]")
        if execution_mode not in self.MODES:
            raise ValueError(f"execution_mode must be one of {sorted(self.MODES)}")
        if int(horizontal_steps) < 0:
            raise ValueError("horizontal_steps must be nonnegative")
        if int(max_loops) < 1:
            raise ValueError("max_loops must be positive")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if horizontal_steps == 0:
            output = self.recurrent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                max_loops=max_loops,
                use_cache=False,
                output_attentions=output_attentions,
                return_dict=True,
            )
            return CompositeCoconutOutput(
                loss=output.loss,
                logits=output.logits,
                recurrent_output=output,
                input_embeddings=self.recurrent.qwen.embed_tokens(input_ids),
                requested_horizontal_steps=0,
                executed_horizontal_steps=0,
                vertical_loops=int(max_loops),
                feedback_grid_applications=0,
                total_grid_applications=int(max_loops),
                requested_execution_mode=execution_mode,
                executed_execution_mode="recompute",
            )

        positions = _control_positions(input_ids, self.latent_token_id, int(horizontal_steps))
        input_embeddings = self.recurrent.qwen.embed_tokens(input_ids)
        if torch.is_grad_enabled() and not input_embeddings.requires_grad:
            # Frozen embeddings otherwise produce a non-differentiable boundary
            # tensor. A leaf activation keeps RG-3 observable without unfreezing
            # or updating the embedding table.
            input_embeddings = input_embeddings.detach().requires_grad_(True)
        if input_embeddings.requires_grad:
            input_embeddings.retain_grad()
        if execution_mode == "sliced_cache":
            if int(max_loops) != 1:
                raise ValueError("sliced cache is permitted only for one vertical loop")
            if self.training and getattr(self.recurrent.qwen, "gradient_checkpointing", False):
                raise ValueError("sliced cache is forbidden while gradient checkpointing is active")
            return self._forward_sliced_cache(
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                positions=positions,
                raw_feedback=raw_feedback,
                horizontal_state_additions=horizontal_state_additions or {},
                output_attentions=output_attentions,
            )
        return self._forward_recompute(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            positions=positions,
            max_loops=int(max_loops),
            raw_feedback=raw_feedback,
            horizontal_state_additions=horizontal_state_additions or {},
            output_attentions=output_attentions,
        )

    def _feedback(
        self,
        states: torch.Tensor,
        *,
        step: int,
        raw_feedback: bool,
        additions: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if int(step) in additions:
            states = states + additions[int(step)].to(device=states.device, dtype=states.dtype)
        return states if raw_feedback else self.horizontal_bridge(states)

    @torch.inference_mode()
    def depth_by_append(
        self,
        *,
        input_ids: torch.LongTensor,
        append_steps: int,
        feedback_mode: str,
        reference_rms: float | None = None,
        neutral_token_id: int | None = None,
        read_at_t_query: bool = False,
        capture_real_logits: bool = False,
        prediction_vocab_size: int | None = None,
    ) -> DepthByAppendOutput:
        """Run transient per-position feedback slots with exact cache eviction.

        ``read_at_t_query`` is the causal operationalization of the addendum's
        fifth arm: after each feedback slot, a transient query carrying the
        original token embedding and rotary position id is evaluated after the
        slot. It is not literal backward attention to an earlier cached query.
        """

        from transformers.cache_utils import DynamicCache

        if input_ids.dim() != 2 or input_ids.shape[1] < 2:
            raise ValueError("depth-by-append requires [batch, sequence>=2] input ids")
        if int(append_steps) < 0:
            raise ValueError("append_steps must be nonnegative")
        if feedback_mode not in {"raw", "rms_matched", "neutral"}:
            raise ValueError("feedback_mode must be raw, rms_matched, or neutral")
        if feedback_mode == "rms_matched" and (reference_rms is None or reference_rms <= 0):
            raise ValueError("rms_matched feedback requires a positive reference_rms")
        if feedback_mode == "neutral" and neutral_token_id is None:
            raise ValueError("neutral feedback requires neutral_token_id")
        if int(append_steps) == 0 and read_at_t_query:
            raise ValueError("read_at_t_query requires at least one appended feedback slot")

        batch, sequence = input_ids.shape
        device = input_ids.device
        if int(append_steps) == 0:
            registered = self.recurrent(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_loops=1,
                use_cache=False,
                return_dict=True,
            )
            if registered.final_post_norm_hidden is None:
                raise RuntimeError("M7 registered k=0 pass did not expose post-norm state")
            logits = registered.logits[:, :-1, :prediction_vocab_size].float()
            hidden = registered.final_post_norm_hidden[:, 0, :-1].float()
            result = DepthByAppendOutput(
                predictions=logits.argmax(dim=-1).cpu().unsqueeze(-1),
                requested_append_steps=0,
                executed_decision_positions=batch * (sequence - 1),
                total_grid_applications=batch * (sequence - 1),
                feedback_grid_applications=0,
                evicted_slots=0,
                eviction_assertions=batch * (sequence - 1),
                diagnostics={
                    "feedback_mode": feedback_mode,
                    "read_at_t_query": False,
                    "execution_mode": "registered_full_sequence_k0",
                    "real_position_ids": list(range(sequence - 1)),
                    "expected_real_position_ids": list(range(sequence - 1)),
                    "cache_lengths_after_eviction": [],
                    "fed_hidden_rms_mean": float(
                        hidden.square().mean(dim=-1).sqrt().mean().cpu()
                    ),
                    "injected_rms_mean": None,
                },
                real_logits=logits.cpu() if capture_real_logits else None,
            )
            result.assert_accounting()
            return result

        cache = DynamicCache(config=self.recurrent.config)
        predictions: list[torch.Tensor] = []
        real_logits: list[torch.Tensor] = []
        fed_rms: list[float] = []
        injected_rms: list[float] = []
        cache_lengths_after_eviction: list[int] = []
        real_position_ids: list[int] = []
        feedback_slots = 0
        readout_slots = 0
        eviction_assertions = 0

        for position in range(sequence - 1):
            expected_prefix = position
            if _cache_length(cache) != expected_prefix:
                raise RuntimeError(
                    f"M7 pre-real cache length {_cache_length(cache)} != {expected_prefix}"
                )
            token = input_ids[:, position : position + 1]
            token_embedding = self.recurrent.qwen.embed_tokens(token)
            output = self.recurrent(
                inputs_embeds=token_embedding,
                attention_mask=torch.ones((batch, position + 1), dtype=torch.long, device=device),
                position_ids=torch.full((batch, 1), position, dtype=torch.long, device=device),
                cache_position=torch.tensor([position], dtype=torch.long, device=device),
                past_key_values=cache,
                max_loops=1,
                use_cache=True,
                return_dict=True,
            )
            if output.final_post_norm_hidden is None:
                raise RuntimeError("M7 real-token pass did not expose final post-norm hidden state")
            base_length = position + 1
            if _cache_length(cache) != base_length:
                raise RuntimeError(f"M7 real-token cache length mismatch at position {position}")
            real_position_ids.append(position)
            base_logits = output.logits[:, -1, :prediction_vocab_size].float()
            if capture_real_logits:
                real_logits.append(base_logits.cpu())
            step_predictions = [base_logits.argmax(dim=-1).cpu()]
            state = output.final_post_norm_hidden[:, 0, -1]
            fed_rms.append(float(state.float().square().mean().sqrt().cpu()))

            for step in range(1, int(append_steps) + 1):
                if feedback_mode == "neutral":
                    slot_ids = torch.full(
                        (batch, 1), int(neutral_token_id), dtype=torch.long, device=device
                    )
                    slot_embedding = self.recurrent.qwen.embed_tokens(slot_ids)
                else:
                    slot = self.horizontal_bridge(state)
                    if feedback_mode == "rms_matched":
                        observed = slot.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                        slot = slot * (float(reference_rms) / observed).to(dtype=slot.dtype)
                    slot_embedding = slot.unsqueeze(1)
                injected_rms.append(
                    float(slot_embedding.float().square().mean().sqrt().cpu())
                )
                cache_position = base_length + step - 1
                slot_output = self.recurrent(
                    inputs_embeds=slot_embedding,
                    attention_mask=torch.ones(
                        (batch, cache_position + 1), dtype=torch.long, device=device
                    ),
                    position_ids=torch.full(
                        (batch, 1), position + step, dtype=torch.long, device=device
                    ),
                    cache_position=torch.tensor([cache_position], dtype=torch.long, device=device),
                    past_key_values=cache,
                    max_loops=1,
                    use_cache=True,
                    return_dict=True,
                )
                if slot_output.final_post_norm_hidden is None:
                    raise RuntimeError("M7 feedback pass did not expose final post-norm hidden state")
                feedback_slots += batch
                state = slot_output.final_post_norm_hidden[:, 0, -1]
                if read_at_t_query:
                    query_cache_position = cache_position + 1
                    query_output = self.recurrent(
                        inputs_embeds=token_embedding,
                        attention_mask=torch.ones(
                            (batch, query_cache_position + 1), dtype=torch.long, device=device
                        ),
                        position_ids=torch.full(
                            (batch, 1), position, dtype=torch.long, device=device
                        ),
                        cache_position=torch.tensor(
                            [query_cache_position], dtype=torch.long, device=device
                        ),
                        past_key_values=cache,
                        max_loops=1,
                        use_cache=True,
                        return_dict=True,
                    )
                    step_predictions.append(
                        query_output.logits[:, -1, :prediction_vocab_size]
                        .float()
                        .argmax(dim=-1)
                        .cpu()
                    )
                    readout_slots += batch
                    cache.crop(query_cache_position)
                    if _cache_length(cache) != query_cache_position:
                        raise RuntimeError("M7 readout-query eviction failed")
                else:
                    step_predictions.append(
                        slot_output.logits[:, -1, :prediction_vocab_size]
                        .float()
                        .argmax(dim=-1)
                        .cpu()
                    )

            cache.crop(base_length)
            if _cache_length(cache) != base_length:
                raise RuntimeError("M7 feedback eviction did not restore the real-token prefix")
            cache_lengths_after_eviction.append(_cache_length(cache))
            eviction_assertions += batch
            predictions.append(torch.stack(step_predictions, dim=-1))

        result = DepthByAppendOutput(
            predictions=torch.stack(predictions, dim=1),
            requested_append_steps=int(append_steps),
            executed_decision_positions=batch * (sequence - 1),
            total_grid_applications=(
                batch * (sequence - 1) + feedback_slots + readout_slots
            ),
            feedback_grid_applications=feedback_slots,
            readout_grid_applications=readout_slots,
            evicted_slots=feedback_slots + readout_slots,
            eviction_assertions=eviction_assertions,
            diagnostics={
                "feedback_mode": feedback_mode,
                "read_at_t_query": bool(read_at_t_query),
                "execution_mode": "incremental_cache_append",
                "read_at_t_query_operationalization": (
                    "transient post-feedback query with original token embedding and rotary position"
                    if read_at_t_query
                    else None
                ),
                "real_position_ids": real_position_ids,
                "expected_real_position_ids": list(range(sequence - 1)),
                "cache_lengths_after_eviction": cache_lengths_after_eviction,
                "fed_hidden_rms_mean": sum(fed_rms) / len(fed_rms),
                "injected_rms_mean": sum(injected_rms) / len(injected_rms)
                if injected_rms
                else None,
            },
            real_logits=torch.stack(real_logits, dim=1) if real_logits else None,
        )
        result.assert_accounting()
        if result.diagnostics["real_position_ids"] != result.diagnostics["expected_real_position_ids"]:
            raise RuntimeError("M7 post-eviction real position ids diverged from the source sequence")
        return result

    def _forward_recompute(
        self,
        *,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        positions: list[int],
        max_loops: int,
        raw_feedback: bool,
        horizontal_state_additions: dict[int, torch.Tensor],
        output_attentions: bool,
    ) -> CompositeCoconutOutput:
        working = input_embeddings
        fed_states: list[torch.Tensor] = []
        grid: list[tuple[torch.Tensor, ...]] = []
        prefix_rms: list[float] = []
        embedding_rms: list[float] = []
        for step, position in enumerate(positions, start=1):
            applications: list[torch.Tensor] = []
            prefix = working[:, :position]
            output = self.recurrent(
                inputs_embeds=prefix,
                attention_mask=attention_mask[:, :position],
                labels=None,
                max_loops=max_loops,
                use_cache=False,
                output_attentions=output_attentions,
                return_dict=True,
                recurrent_application_sink=applications,
            )
            if output.final_post_norm_hidden is None:
                raise RuntimeError("recurrent wrapper did not expose post-norm hidden states")
            fed = output.final_post_norm_hidden[:, 0, -1]
            if fed.requires_grad:
                fed.retain_grad()
            fed_states.append(fed)
            grid.append(tuple(applications))
            prefix_rms.append(float(fed.detach().float().square().mean().sqrt().cpu()))
            replacement = self._feedback(
                fed,
                step=step,
                raw_feedback=raw_feedback,
                additions=horizontal_state_additions,
            )
            embedding_rms.append(
                float(input_embeddings[:, position].detach().float().square().mean().sqrt().cpu())
            )
            working = rebuild_embedding_slot(working, position, replacement)

        final_applications: list[torch.Tensor] = []
        final = self.recurrent(
            inputs_embeds=working,
            attention_mask=attention_mask,
            labels=labels,
            max_loops=max_loops,
            use_cache=False,
            output_attentions=output_attentions,
            return_dict=True,
            recurrent_application_sink=final_applications,
        )
        grid.append(tuple(final_applications))
        return CompositeCoconutOutput(
            loss=final.loss,
            logits=final.logits,
            recurrent_output=final,
            input_embeddings=input_embeddings,
            horizontal_fed_states=tuple(fed_states),
            recurrent_application_states=tuple(grid),
            requested_horizontal_steps=len(positions),
            executed_horizontal_steps=len(fed_states),
            vertical_loops=max_loops,
            feedback_grid_applications=len(positions) * max_loops,
            total_grid_applications=(len(positions) + 1) * max_loops,
            requested_execution_mode="recompute",
            executed_execution_mode="recompute",
            diagnostics={
                "fed_hidden_rms": prefix_rms,
                "placeholder_embedding_rms": embedding_rms,
            },
        )

    def _forward_sliced_cache(
        self,
        *,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        positions: list[int],
        raw_feedback: bool,
        horizontal_state_additions: dict[int, torch.Tensor],
        output_attentions: bool,
    ) -> CompositeCoconutOutput:
        from transformers.cache_utils import DynamicCache

        working = input_embeddings
        cache = DynamicCache(config=self.recurrent.config)
        current_start = 0
        logit_segments: list[torch.Tensor] = []
        fed_states: list[torch.Tensor] = []
        grid: list[tuple[torch.Tensor, ...]] = []
        cache_lengths: list[int] = []
        for step, position in enumerate(positions, start=1):
            applications: list[torch.Tensor] = []
            segment_end = position + 1
            segment = working[:, current_start:segment_end]
            output = self.recurrent(
                inputs_embeds=segment,
                attention_mask=attention_mask[:, :segment_end],
                labels=None,
                max_loops=1,
                past_key_values=cache,
                use_cache=True,
                output_attentions=output_attentions,
                return_dict=True,
                recurrent_application_sink=applications,
            )
            if output.final_post_norm_hidden is None or segment.shape[1] < 2:
                raise RuntimeError("cache segment did not expose the pre-placeholder hidden state")
            fed = output.final_post_norm_hidden[:, 0, -2]
            if fed.requires_grad:
                fed.retain_grad()
            fed_states.append(fed)
            grid.append(tuple(applications))
            logit_segments.append(output.logits[:, :-1])
            if _cache_length(cache) != segment_end:
                raise RuntimeError(
                    f"cache length {_cache_length(cache)} != processed prefix {segment_end}"
                )
            cache.crop(position)
            if _cache_length(cache) != position:
                raise RuntimeError("cache crop did not remove the placeholder position")
            cache_lengths.append(_cache_length(cache))
            replacement = self._feedback(
                fed,
                step=step,
                raw_feedback=raw_feedback,
                additions=horizontal_state_additions,
            )
            working = rebuild_embedding_slot(working, position, replacement)
            current_start = position

        final_applications: list[torch.Tensor] = []
        final_segment = working[:, current_start:]
        final = self.recurrent(
            inputs_embeds=final_segment,
            attention_mask=attention_mask,
            labels=None,
            max_loops=1,
            past_key_values=cache,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
            recurrent_application_sink=final_applications,
        )
        grid.append(tuple(final_applications))
        logit_segments.append(final.logits)
        logits = torch.cat(logit_segments, dim=1)
        if logits.shape[:2] != input_embeddings.shape[:2]:
            raise RuntimeError(
                f"assembled cached logits {tuple(logits.shape[:2])} do not match input "
                f"{tuple(input_embeddings.shape[:2])}"
            )
        loss = _sequence_cross_entropy(logits, labels) if labels is not None else None
        return CompositeCoconutOutput(
            loss=loss,
            logits=logits,
            recurrent_output=final,
            input_embeddings=input_embeddings,
            horizontal_fed_states=tuple(fed_states),
            recurrent_application_states=tuple(grid),
            requested_horizontal_steps=len(positions),
            executed_horizontal_steps=len(fed_states),
            vertical_loops=1,
            feedback_grid_applications=len(positions),
            total_grid_applications=len(positions) + 1,
            requested_execution_mode="sliced_cache",
            executed_execution_mode="sliced_cache",
            cache_prefix_lengths=tuple(cache_lengths),
        )


def configure_composite_trainable_set(
    model: CoconutRecurrentQwen,
    *,
    budget: str,
    horizontal_bridge_trainable: bool,
) -> set[str]:
    """Select exactly the registered future C1 trainable families."""

    if budget not in {"full_block", "adapter_r16"}:
        raise ValueError("budget must be full_block or adapter_r16")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    recurrent = model.recurrent
    if budget == "full_block":
        for layer_index in range(recurrent.layer_split.prelude_end, recurrent.layer_split.recurrent_end):
            for parameter in recurrent.qwen.layers[layer_index].parameters():
                parameter.requires_grad_(True)
    else:
        found = 0
        for module in recurrent.base_model.modules():
            if isinstance(module, LoRALinear):
                for parameter in module.lora_parameters():
                    parameter.requires_grad_(True)
                found += 1
        if found == 0:
            raise RuntimeError("adapter_r16 budget requires installed recurrent-block LoRA modules")

    for name, parameter in recurrent.bridge.named_parameters():
        if recurrent.bridge.split_projection and name.startswith("proj."):
            continue
        parameter.requires_grad_(True)
    for name, parameter in recurrent.base_model.named_parameters():
        if name.endswith("control_rows"):
            parameter.requires_grad_(True)
    for parameter in model.horizontal_bridge.parameters():
        parameter.requires_grad_(bool(horizontal_bridge_trainable))
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def assert_parameter_group_coverage(
    intended_names: Iterable[str],
    optimizer_names: Iterable[str],
    ema_names: Iterable[str],
) -> dict[str, Any]:
    intended = set(intended_names)
    optimizer = set(optimizer_names)
    ema = set(ema_names)
    if optimizer != intended or ema != intended:
        raise AssertionError(
            "optimizer/EMA parameter-name coverage differs from the intended composite set: "
            f"optimizer_missing={sorted(intended - optimizer)[:8]}, "
            f"optimizer_extra={sorted(optimizer - intended)[:8]}, "
            f"ema_missing={sorted(intended - ema)[:8]}, ema_extra={sorted(ema - intended)[:8]}"
        )
    def names_sha256(names: set[str]) -> str:
        payload = "\n".join(sorted(names)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "intended_name_count": len(intended),
        "optimizer_name_sha256": names_sha256(optimizer),
        "ema_name_sha256": names_sha256(ema),
        "parameter_names": sorted(intended),
        "passed": True,
    }
