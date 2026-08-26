"""End-to-end PyTorch language model assembled from ablation-safe modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import AblationLMConfig
from .engram import CausalTokenEngram, TokenEngramConfig
from .geometry import lanes_to_split_clifford
from .layers import ModifiedHadamardExpertBank, RMSNorm, TransformerBlock
from .memory import ReadOnlyLatentMemory
from .reentry import AnchoredReentryBridge
from .scratch import PositionAlignedScratch


@dataclass
class AblationLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    diagnostics: dict[str, Any]


class AblationLM(nn.Module):
    """A dense causal LM whose innovations are independently removable.

    When structural recurrence is enabled, the unique core blocks are revisited
    ``T`` times and every recurrent residual is scaled by ``alpha_T = c/T``.
    With recurrence and all optional modules structurally absent, the graph is
    exactly an ordinary sequential Transformer.
    """

    def __init__(
        self,
        config: AblationLMConfig,
        *,
        long_term_memory: ReadOnlyLatentMemory | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.use_long_term_memory != (long_term_memory is not None):
            raise ValueError(
                "use_long_term_memory must exactly match an explicitly supplied frozen store"
            )
        if long_term_memory is not None:
            if long_term_memory.d_model != config.d_model:
                raise ValueError("long-term memory d_model does not match the model")
            if long_term_memory.slots != config.long_term_memory_slots:
                raise ValueError("long-term memory record count does not match the config")
            if long_term_memory.memory_width != config.long_term_memory_width:
                raise ValueError("long-term memory width does not match the config")
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.front_hadamard = (
            ModifiedHadamardExpertBank(
                config.d_model,
                experts=config.hadamard_experts,
                layer_scale=config.hadamard_layer_scale,
                norm_eps=config.norm_eps,
                seed=config.hadamard_seed,
            )
            if config.use_front_hadamard_experts
            else None
        )
        self.prelude_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_prelude_layers)
        )
        self.engram = (
            CausalTokenEngram(
                TokenEngramConfig(
                    hidden_dim=config.d_model,
                    num_slots=config.engram_table_size,
                    ngram_orders=config.engram_orders,
                    num_hash_heads=config.engram_hashes_per_order,
                    head_dim=config.engram_row_dim,
                    initial_scale=config.engram_layer_scale,
                    seed=config.engram_hash_seed,
                )
            )
            if config.use_engram
            else None
        )
        self.core_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_core_blocks)
        )
        self.loop_embedding = (
            nn.Embedding(config.max_recurrent_steps, config.d_model)
            if config.use_recurrence
            else None
        )
        self.reentry_bridge = (
            AnchoredReentryBridge(
                config.d_model,
                layer_scale=config.bridge_layer_scale,
                norm_eps=config.norm_eps,
            )
            if config.use_reentry_bridge
            else None
        )
        self.scratch = (
            PositionAlignedScratch(
                config.d_model,
                lane_width=config.scratch_width,
                max_steps=config.max_recurrent_steps,
                layer_scale=config.scratch_layer_scale,
                rho_init=config.lane_carrier_rho_init,
                norm_eps=config.norm_eps,
                use_carrier=config.use_lane_carrier,
            )
            if config.use_scratch
            else None
        )
        self.long_term_memory = long_term_memory
        self.coda_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_coda_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.initialization_seed)
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        physical_blocks = (
            self.config.n_prelude_layers
            + self.config.n_core_blocks
            + self.config.n_coda_layers
        )
        residual_std = 0.02 / (2 * physical_blocks) ** 0.5
        for block in (*self.prelude_blocks, *self.core_blocks, *self.coda_blocks):
            ordinary = (
                block.attention.q_proj,
                block.attention.k_proj,
                block.attention.v_proj,
                block.feed_forward.gate_proj,
                block.feed_forward.up_proj,
            )
            residual_outputs = (
                block.attention.output_proj,
                block.feed_forward.down_proj,
            )
            for projection in ordinary:
                nn.init.normal_(
                    projection.weight,
                    mean=0.0,
                    std=0.02,
                    generator=generator,
                )
            for projection in residual_outputs:
                nn.init.normal_(
                    projection.weight,
                    mean=0.0,
                    std=residual_std,
                    generator=generator,
                )
        if self.loop_embedding is not None:
            nn.init.zeros_(self.loop_embedding.weight)

    @staticmethod
    def _document_position_ids(document_ids: torch.Tensor) -> torch.Tensor:
        """Reset text positions at every contiguous packed-document boundary."""

        batch, length = document_ids.shape
        absolute = torch.arange(length, device=document_ids.device).view(1, -1).expand(batch, -1)
        boundaries = torch.ones_like(document_ids, dtype=torch.bool)
        if length > 1:
            boundaries[:, 1:] = document_ids[:, 1:].ne(document_ids[:, :-1])
        starts = torch.where(boundaries, absolute, torch.zeros_like(absolute)).cummax(dim=1).values
        local = absolute - starts
        return local.masked_fill(document_ids.lt(0), 0)

    @staticmethod
    def _contiguous_document_segments(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Give each contiguous valid segment a distinct local identifier."""

        if attention_mask is None and document_ids is None:
            return None
        if document_ids is None:
            source_ids = torch.zeros_like(input_ids)
            valid = attention_mask.bool()
        else:
            source_ids = document_ids.long()
            valid = source_ids.ge(0)
            if attention_mask is not None:
                valid &= attention_mask.bool()
        boundaries = torch.ones_like(source_ids, dtype=torch.bool)
        if source_ids.shape[1] > 1:
            boundaries[:, 1:] = source_ids[:, 1:].ne(source_ids[:, :-1])
            boundaries[:, 1:] |= valid[:, 1:].ne(valid[:, :-1])
        segment_ids = boundaries.long().cumsum(dim=1) - 1
        return segment_ids.masked_fill(~valid, -1)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor [batch, sequence]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size):
            raise ValueError("input_ids contain values outside the configured vocabulary")
        for name, values in (("attention_mask", attention_mask), ("document_ids", document_ids)):
            if values is not None and values.shape != input_ids.shape:
                raise ValueError(f"{name} must match input_ids")
            if values is not None and values.device != input_ids.device:
                raise ValueError(f"{name} must be on the same device as input_ids")
        if attention_mask is not None:
            integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            if attention_mask.dtype != torch.bool and attention_mask.dtype not in integer_dtypes:
                raise TypeError("attention_mask must be boolean or an exact 0/1 integer tensor")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask values must be exactly zero or one")
        if document_ids is not None and document_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("document_ids must use an integer dtype")
        if position_ids is not None:
            if position_ids.shape != input_ids.shape:
                raise ValueError("position_ids must match input_ids")
            if position_ids.device != input_ids.device:
                raise ValueError("position_ids must be on the same device as input_ids")
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("position_ids must use an integer dtype")
            if position_ids.numel() and int(position_ids.min()) < 0:
                raise ValueError("position_ids must be non-negative")

    def _language_model_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must match [batch, sequence]")
        if labels.device != logits.device:
            raise ValueError("labels must be on the same device as logits")
        if labels.dtype not in (torch.int32, torch.int64):
            raise TypeError("labels must use an integer dtype")
        valid_labels = labels.eq(-100) | (labels.ge(0) & labels.lt(self.config.vocab_size))
        if not bool(valid_labels.all()):
            raise ValueError("labels must be -100 or valid vocabulary IDs")
        if logits.shape[1] < 2:
            raise ValueError("language-model loss requires at least two tokens")
        targets = labels[:, 1:].long().clone()
        if attention_mask is not None:
            valid_pair = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
            targets.masked_fill_(~valid_pair, -100)
        if document_ids is not None:
            same_document = document_ids[:, 1:].eq(document_ids[:, :-1])
            same_document &= document_ids[:, 1:].ge(0)
            targets.masked_fill_(~same_document, -100)
        if not bool(targets.ne(-100).any()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[:, :-1].float().reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        memory_record_ids: torch.Tensor | None = None,
        recurrent_steps: int | None = None,
        return_diagnostics: bool = False,
        use_cache: bool = False,
    ) -> AblationLMOutput:
        """Run the causal graph.

        KV caching is intentionally gated off in this first build.  Its future
        implementation must prove cache identity and the no-``T``-multiplier
        storage contract before it can be enabled.
        """

        if use_cache:
            raise NotImplementedError("KV cache requires a separate identity and accounting gate")
        self._validate_inputs(input_ids, attention_mask, document_ids, position_ids)
        if memory_record_ids is not None:
            if memory_record_ids.shape != input_ids.shape:
                raise ValueError("memory_record_ids must match input_ids")
            if memory_record_ids.device != input_ids.device:
                raise ValueError("memory_record_ids must be on the same device as input_ids")
            if memory_record_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("memory_record_ids must use an integer dtype")
        if recurrent_steps is not None and type(recurrent_steps) is not int:
            raise TypeError("recurrent_steps must be an exact integer")
        if labels is not None and self.training and self.long_term_memory is not None:
            if memory_record_ids is None:
                raise ValueError(
                    "training with long-term memory requires leave-one-out memory_record_ids"
                )
        effective_document_ids = self._contiguous_document_segments(
            input_ids,
            attention_mask,
            document_ids,
        )
        if self.config.use_recurrence:
            steps = self.config.recurrent_steps if recurrent_steps is None else recurrent_steps
            alpha = self.config.recurrence_scale(steps)
            if self.reentry_bridge is not None and steps < 2:
                raise ValueError("the active re-entry bridge requires at least two visits")
        else:
            if recurrent_steps not in (None, 1):
                raise ValueError("recurrent_steps override requires structural recurrence")
            steps = 1
            alpha = 1.0
        if position_ids is None and effective_document_ids is not None:
            position_ids = self._document_position_ids(effective_document_ids)

        hidden = self.token_embedding(input_ids)
        if self.front_hadamard is not None:
            hidden = self.front_hadamard(hidden)

        engram_audit: dict[str, torch.Tensor] | None = None
        for index, block in enumerate(self.prelude_blocks):
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=effective_document_ids,
            )
            if index == 0 and self.engram is not None:
                hidden, engram_audit = self.engram(
                    hidden,
                    input_ids,
                    document_ids=effective_document_ids,
                    enabled=True,
                )

        prelude = hidden
        lanes = self.scratch.initialize(prelude) if self.scratch is not None else None
        loop_rms: list[torch.Tensor] = []
        for step_index in range(steps):
            if step_index > 0 and self.reentry_bridge is not None:
                hidden = self.reentry_bridge(hidden, prelude, residual_scale=alpha)
            if lanes is not None:
                assert self.scratch is not None
                lanes = self.scratch.step(
                    lanes,
                    hidden,
                    step_index=step_index,
                    residual_scale=alpha,
                )
                hidden = self.scratch.inject(hidden, lanes, residual_scale=alpha)
            if self.loop_embedding is not None:
                hidden = hidden + alpha * self.loop_embedding.weight[step_index].to(
                    dtype=hidden.dtype
                )
            for block in self.core_blocks:
                hidden = block(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    document_ids=effective_document_ids,
                    residual_scale=alpha,
                )
            if return_diagnostics:
                loop_rms.append(hidden.float().square().mean().sqrt().detach())

        memory_audit: dict[str, torch.Tensor] | None = None
        if self.long_term_memory is not None:
            hidden, memory_audit = self.long_term_memory(
                hidden,
                record_ids=memory_record_ids,
            )
        for block in self.coda_blocks:
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=effective_document_ids,
            )
        logits = self.lm_head(self.final_norm(hidden))
        loss = (
            self._language_model_loss(
                logits,
                labels,
                attention_mask,
                effective_document_ids,
            )
            if labels is not None
            else None
        )

        diagnostics: dict[str, Any] = {}
        if return_diagnostics:
            diagnostics["alpha_t"] = alpha
            diagnostics["recurrence_enabled"] = self.config.use_recurrence
            diagnostics["executed_core_visits"] = steps
            diagnostics["executed_core_block_passes"] = steps * len(self.core_blocks)
            diagnostics["loop_rms"] = torch.stack(loop_rms)
            if engram_audit is not None:
                diagnostics["engram"] = engram_audit
            if memory_audit is not None:
                diagnostics["long_term_memory"] = memory_audit
            if lanes is not None:
                coordinates = lanes_to_split_clifford(lanes)
                diagnostics["scratch_mu_rms"] = coordinates.mu.float().square().mean().sqrt().detach()
                diagnostics["scratch_delta_rms"] = (
                    coordinates.delta.float().square().mean().sqrt().detach()
                )
                if self.scratch.carrier is not None:
                    diagnostics["lane_carrier_rho"] = self.scratch.carrier.rho().detach()
        return AblationLMOutput(logits=logits, loss=loss, diagnostics=diagnostics)
