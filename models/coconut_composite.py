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
