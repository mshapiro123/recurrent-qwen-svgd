"""End-to-end PyTorch language model assembled from ablation-safe modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import AblationLMConfig
from .engram import CausalTokenEngram, TokenEngramConfig
from .geometry import lanes_to_modes
from .jets import (
    LoopGradientProbe,
    estimate_jacobian_spectral_norm,
    plane_probe_features,
    trajectory_jet_metrics,
)
from .layers import ModifiedHadamardExpertBank, RMSNorm, TransformerBlock
from .memory import ReadOnlyLatentMemory
from .optim import RANK_ONLY_MUON_PROHIBITED_ATTR, REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR
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

    _ablation_lm_require_closed_muon_allowlist = True

    assert REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR == (
        "_ablation_lm_require_closed_muon_allowlist"
    )

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
                retention_floor=config.lane_carrier_retention_floor,
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
        self._restore_parameter_contract()

    def _mark_rank_only_muon_prohibited(self) -> None:
        for parameter in self.parameters():
            setattr(parameter, RANK_ONLY_MUON_PROHIBITED_ATTR, True)

    def _restore_parameter_contract(self) -> None:
        if self.lm_head.weight.shape != self.token_embedding.weight.shape:
            raise RuntimeError("language-model head and token embedding shapes differ")
        self.lm_head.weight = self.token_embedding.weight
        self._mark_rank_only_muon_prohibited()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore tying and safety markers that ``Parameter`` deepcopy drops."""

        super().__setstate__(state)
        self._restore_parameter_contract()

    def _apply(self, fn: Any, recurse: bool = True) -> "AblationLM":
        """Preserve tied identity and safety provenance across device transforms."""

        super()._apply(fn, recurse=recurse)
        self._restore_parameter_contract()
        return self

    def load_state_dict(
        self,
        state_dict: Any,
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Validate tied checkpoint aliases, then restore identity after assign-load."""

        embedding = state_dict.get("token_embedding.weight")
        head = state_dict.get("lm_head.weight")
        if (embedding is None) != (head is None):
            raise RuntimeError(
                "tied token_embedding.weight and lm_head.weight must appear together"
            )
        if embedding is not None and head is not None:
            if embedding.device.type == "meta" or head.device.type == "meta":
                raise RuntimeError("cannot validate tied aliases from a meta checkpoint")
            if not torch.equal(embedding, head):
                raise RuntimeError(
                    "checkpoint token_embedding.weight and lm_head.weight disagree"
                )
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            self._restore_parameter_contract()

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

    def _run_recurrent_visit(
        self,
        hidden: torch.Tensor,
        *,
        prelude: torch.Tensor,
        lanes: torch.Tensor | None,
        step_index: int,
        alpha: float,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        force_math_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one complete recurrent transition at a fixed visit index."""

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
                document_ids=document_ids,
                residual_scale=alpha,
                force_math_attention=force_math_attention,
            )
        return hidden, lanes

    def _visit_jacobian_spectral_norm(
        self,
        hidden: torch.Tensor,
        *,
        prelude: torch.Tensor,
        lanes: torch.Tensor | None,
        step_index: int,
        alpha: float,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        valid_token_mask: torch.Tensor,
        iterations: int,
    ) -> torch.Tensor:
        """Estimate the local Jacobian norm of the joint hidden/scratch map."""

        if self.training and self.config.attention_dropout:
            raise ValueError("Jacobian probes require deterministic dropout-free transitions")
        lane_shape = tuple(lanes.shape) if lanes is not None else None
        lane_metric_scale = (
            (self.config.d_model / (2 * self.config.scratch_width)) ** 0.5
            if lanes is not None
            else 1.0
        )
        packed = (
            hidden
            if lanes is None
            else torch.cat(
                (hidden, lanes.flatten(start_dim=-2) * lane_metric_scale),
                dim=-1,
            )
        )
        packed_input_mask = valid_token_mask.unsqueeze(-1).expand_as(packed)
        hidden_mask = valid_token_mask.unsqueeze(-1)
        lane_mask = valid_token_mask.unsqueeze(-1).unsqueeze(-1)
        base_hidden = hidden.detach()
        base_lanes = lanes.detach() if lanes is not None else None

        def transition(state: torch.Tensor) -> torch.Tensor:
            state_hidden = state[..., : self.config.d_model]
            state_hidden = torch.where(hidden_mask, state_hidden, base_hidden)
            state_lanes = None
            if lane_shape is not None:
                state_lanes = (
                    state[..., self.config.d_model :] / lane_metric_scale
                ).reshape(lane_shape)
                assert base_lanes is not None
                state_lanes = torch.where(lane_mask, state_lanes, base_lanes)
            next_hidden, next_lanes = self._run_recurrent_visit(
                state_hidden,
                prelude=prelude,
                lanes=state_lanes,
                step_index=step_index,
                alpha=alpha,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
                force_math_attention=True,
            )
            next_hidden = next_hidden.masked_fill(~hidden_mask, 0.0)
            if next_lanes is None:
                return next_hidden
            next_lanes = next_lanes.masked_fill(~lane_mask, 0.0)
            return torch.cat(
                (next_hidden, next_lanes.flatten(start_dim=-2) * lane_metric_scale),
                dim=-1,
            )

        return estimate_jacobian_spectral_norm(
            transition,
            packed,
            iterations=iterations,
            seed=self.config.initialization_seed + step_index,
            input_mask=packed_input_mask,
        )

    def _horizon_jacobian_spectral_norm(
        self,
        prelude: torch.Tensor,
        *,
        steps: int,
        alpha: float,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        valid_token_mask: torch.Tensor,
        iterations: int,
    ) -> torch.Tensor:
        """Estimate hidden-to-hidden gain across the full unrolled recurrence."""

        if self.training and self.config.attention_dropout:
            raise ValueError("Jacobian probes require deterministic dropout-free transitions")

        hidden_mask = valid_token_mask.unsqueeze(-1)
        base_prelude = prelude.detach()

        def unrolled(initial_hidden: torch.Tensor) -> torch.Tensor:
            initial_hidden = torch.where(hidden_mask, initial_hidden, base_prelude)
            hidden = initial_hidden
            lanes = self.scratch.initialize(initial_hidden) if self.scratch is not None else None
            for step_index in range(steps):
                hidden, lanes = self._run_recurrent_visit(
                    hidden,
                    prelude=initial_hidden,
                    lanes=lanes,
                    step_index=step_index,
                    alpha=alpha,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    document_ids=document_ids,
                    force_math_attention=True,
                )
            return hidden.masked_fill(~hidden_mask, 0.0)

        return estimate_jacobian_spectral_norm(
            unrolled,
            prelude,
            iterations=iterations,
            seed=self.config.initialization_seed,
            input_mask=hidden_mask.expand_as(prelude),
        )

    @staticmethod
    def _valid_token_rms(values: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        """RMS over hidden coordinates at valid sequence positions only."""

        if values.shape[:-1] != valid_token_mask.shape:
            raise ValueError("valid-token RMS mask must align with non-hidden axes")
        per_token_square = values.float().square().mean(dim=-1)
        return per_token_square.masked_select(valid_token_mask).mean().sqrt()

    def _language_model_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        total, _cross_entropy, _z_loss = self._language_model_loss_components(
            logits,
            labels,
            attention_mask,
            document_ids,
        )
        return total

    def _language_model_loss_components(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CE and z-loss to the exact same valid next-token positions."""

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
            zero = logits.float().sum() * 0.0
            return zero, zero, zero
        next_logits = logits[:, :-1].float()
        cross_entropy = F.cross_entropy(
            next_logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        if self.config.z_loss_coefficient == 0.0:
            return cross_entropy, cross_entropy, next_logits.new_zeros(())
        valid = targets.ne(-100)
        log_partition = torch.logsumexp(next_logits, dim=-1)
        z_loss = (
            self.config.z_loss_coefficient
            * log_partition.masked_select(valid).square().mean()
        )
        return cross_entropy + z_loss, cross_entropy, z_loss

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
        capture_loop_gradients: bool = False,
        jacobian_probe_iterations: int = 0,
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
        if type(capture_loop_gradients) is not bool:
            raise TypeError("capture_loop_gradients must be boolean")
        if type(jacobian_probe_iterations) is not int or jacobian_probe_iterations < 0:
            raise ValueError("jacobian_probe_iterations must be a non-negative integer")
        if (capture_loop_gradients or jacobian_probe_iterations) and not return_diagnostics:
            raise ValueError("loop gradient/Jacobian probes require return_diagnostics=True")
        if jacobian_probe_iterations and not self.config.use_recurrence:
            raise ValueError("Jacobian visit probes require structural recurrence")
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
        valid_token_mask = (
            effective_document_ids.ge(0)
            if effective_document_ids is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        if return_diagnostics and not bool(valid_token_mask.any()):
            raise ValueError("diagnostic forwards require at least one valid token")
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
        router_audit: dict[str, torch.Tensor] | None = None
        if self.front_hadamard is not None:
            if return_diagnostics:
                hidden, router_audit = self.front_hadamard(
                    hidden,
                    return_audit=True,
                    audit_mask=valid_token_mask,
                )
            else:
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
        loop_update_rms: list[torch.Tensor] = []
        trajectory_states: list[torch.Tensor] = [prelude.detach()] if return_diagnostics else []
        gradient_states: list[torch.Tensor] = []
        jacobian_norms: list[torch.Tensor] = []
        horizon_jacobian_norm: torch.Tensor | None = None
        if jacobian_probe_iterations:
            horizon_jacobian_norm = self._horizon_jacobian_spectral_norm(
                prelude,
                steps=steps,
                alpha=alpha,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=effective_document_ids,
                valid_token_mask=valid_token_mask,
                iterations=jacobian_probe_iterations,
            ).detach()
        for step_index in range(steps):
            hidden_before_visit = hidden
            if jacobian_probe_iterations:
                jacobian_norms.append(
                    self._visit_jacobian_spectral_norm(
                        hidden,
                        prelude=prelude,
                        lanes=lanes,
                        step_index=step_index,
                        alpha=alpha,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        document_ids=effective_document_ids,
                        valid_token_mask=valid_token_mask,
                        iterations=jacobian_probe_iterations,
                    ).detach()
                )
            hidden, lanes = self._run_recurrent_visit(
                hidden,
                prelude=prelude,
                lanes=lanes,
                step_index=step_index,
                alpha=alpha,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=effective_document_ids,
            )
            if return_diagnostics:
                loop_rms.append(self._valid_token_rms(hidden, valid_token_mask).detach())
                loop_update_rms.append(
                    self._valid_token_rms(
                        hidden.float() - hidden_before_visit.float(),
                        valid_token_mask,
                    ).detach()
                )
                trajectory_states.append(hidden.detach())
            if capture_loop_gradients:
                if not hidden.requires_grad:
                    raise RuntimeError("loop gradient capture requires autograd-enabled execution")
                hidden.retain_grad()
                gradient_states.append(hidden)

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
        loss: torch.Tensor | None = None
        cross_entropy_loss: torch.Tensor | None = None
        z_loss: torch.Tensor | None = None
        if labels is not None:
            loss, cross_entropy_loss, z_loss = self._language_model_loss_components(
                logits,
                labels,
                attention_mask,
                effective_document_ids,
            )

        diagnostics: dict[str, Any] = {}
        if return_diagnostics:
            diagnostics["alpha_t"] = alpha
            diagnostics["recurrence_enabled"] = self.config.use_recurrence
            diagnostics["executed_core_visits"] = steps
            diagnostics["executed_core_block_passes"] = steps * len(self.core_blocks)
            diagnostics["loop_rms"] = torch.stack(loop_rms)
            diagnostics["loop_update_rms"] = torch.stack(loop_update_rms)
            diagnostics["trajectory_state_count"] = len(trajectory_states)
            if len(trajectory_states) >= 3:
                trajectory_tensor = torch.stack(trajectory_states)
                trajectory = trajectory_jet_metrics(trajectory_tensor)
                velocity = trajectory_tensor[1:].float() - trajectory_tensor[:-1].float()
                acceleration = velocity[1:] - velocity[:-1]
                generator = torch.Generator(device=trajectory_tensor.device).manual_seed(
                    self.config.jet_plane_probe_seed
                )
                p = torch.randn(
                    (self.config.jet_plane_probe_count, self.config.d_model),
                    generator=generator,
                    device=trajectory_tensor.device,
                    dtype=torch.float32,
                )
                q = torch.randn(
                    (self.config.jet_plane_probe_count, self.config.d_model),
                    generator=generator,
                    device=trajectory_tensor.device,
                    dtype=torch.float32,
                )
                plane_probes = plane_probe_features(velocity[1:], acceleration, p, q)
                valid_trajectory_tokens = valid_token_mask
                plane_probes = plane_probes.masked_fill(
                    ~valid_trajectory_tokens.unsqueeze(0).unsqueeze(-1),
                    0.0,
                )
                diagnostics["trajectory_jets"] = {
                    "velocity_rms": trajectory.velocity_rms.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "acceleration_rms": trajectory.acceleration_rms.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "turning_cosine": trajectory.turning_cosine.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "wedge_gram": trajectory.wedge_gram.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "curvature": trajectory.curvature.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "gram_eigenvalue_ratio": trajectory.gram_eigenvalue_ratio.masked_fill(
                        ~valid_trajectory_tokens.unsqueeze(0), 0.0
                    ),
                    "plane_probes": plane_probes,
                    "plane_probe_seed": self.config.jet_plane_probe_seed,
                    "plane_probe_count": self.config.jet_plane_probe_count,
                    "valid_token_mask": valid_trajectory_tokens,
                }
            if gradient_states:
                diagnostics["loop_gradient_probe"] = LoopGradientProbe(
                    tuple(gradient_states),
                    valid_token_mask.detach(),
                )
            if jacobian_norms:
                diagnostics["loop_jacobian_spectral_norm"] = torch.stack(jacobian_norms)
                diagnostics["joint_state_lane_metric_scale"] = (
                    (self.config.d_model / (2 * self.config.scratch_width)) ** 0.5
                    if lanes is not None
                    else 1.0
                )
            if horizon_jacobian_norm is not None:
                diagnostics["horizon_jacobian_spectral_norm"] = horizon_jacobian_norm
            if cross_entropy_loss is not None and z_loss is not None:
                diagnostics["cross_entropy_loss"] = cross_entropy_loss.detach()
                diagnostics["z_loss"] = z_loss.detach()
            if router_audit is not None:
                diagnostics["hadamard_router"] = router_audit
            if engram_audit is not None:
                diagnostics["engram"] = engram_audit
            if memory_audit is not None:
                diagnostics["long_term_memory"] = memory_audit
            if lanes is not None:
                coordinates = lanes_to_modes(lanes)
                diagnostics["scratch_mu_rms"] = self._valid_token_rms(
                    coordinates.mu,
                    valid_token_mask,
                ).detach()
                diagnostics["scratch_delta_rms"] = self._valid_token_rms(
                    coordinates.delta,
                    valid_token_mask,
                ).detach()
                if self.scratch.carrier is not None:
                    diagnostics["lane_carrier_rho"] = self.scratch.carrier.rho().detach()
                    diagnostics["lane_carrier_minimum_retention"] = (
                        self.scratch.carrier.minimum_retention(steps).detach()
                    )
                    diagnostics["lane_carrier_retention_floor"] = (
                        self.config.lane_carrier_retention_floor
                    )
        return AblationLMOutput(logits=logits, loss=loss, diagnostics=diagnostics)
