"""End-to-end PyTorch language model assembled from ablation-safe modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .accounting import composition_receipt
from .bicameral_combiner import PerBandUnitCircleCombiner
from .bicameral_core import BicameralTransformerBlock
from .bicameral_recurrent import (
    AfterBlockResult,
    BicameralRecurrenceReceipt,
    execute_bicameral_recurrence,
)
from .config import AblationLMConfig
from .engram import CausalTokenEngram, TokenEngramConfig
from .geometry import lanes_to_modes
from .jets import (
    LoopGradientProbe,
    estimate_jacobian_spectral_norm,
    plane_probe_features,
    trajectory_jet_metrics,
)
from .layers import (
    ModifiedHadamardExpertBank,
    ProjectedKeyValue,
    RMSNorm,
    TransformerBlock,
)
from .memory import ReadOnlyLatentMemory
from .optim import RANK_ONLY_MUON_PROHIBITED_ATTR, REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR
from .reentry import AnchoredReentryBridge
from .rng import ModuleRNGStream, construct_with_isolated_rng, derive_module_seed
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
        if config.use_bicameral_core and (
            config.use_front_hadamard_experts or config.use_reentry_bridge
        ):
            raise ValueError(
                "the production bicameral path structurally retires front-WHT "
                "and the legacy h0 re-entry bridge"
            )
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
            if long_term_memory.initialization_seed != config.initialization_seed:
                raise ValueError(
                    "long-term memory initialization_seed must match the model config"
                )

        def construct(source_key: str, factory: Any) -> Any:
            return construct_with_isolated_rng(
                factory,
                base_seed=config.initialization_seed,
                source_key=source_key,
                replica=0,
            )

        self.token_embedding = construct(
            "model.token_embedding.constructor",
            lambda: nn.Embedding(config.vocab_size, config.d_model),
        )
        self.front_hadamard = (
            construct(
                "model.front_hadamard.constructor",
                lambda: ModifiedHadamardExpertBank(
                    config.d_model,
                    experts=config.hadamard_experts,
                    layer_scale=config.hadamard_layer_scale,
                    norm_eps=config.norm_eps,
                    seed=config.hadamard_seed,
                ),
            )
            if config.use_front_hadamard_experts
            else None
        )
        self.prelude_blocks = nn.ModuleList(
            construct(
                f"model.prelude.{index}.constructor",
                lambda index=index: TransformerBlock(
                    config,
                    module_path=f"model.prelude.{index}",
                ),
            )
            for index in range(config.n_prelude_layers)
        )
        self.engram = (
            construct(
                "model.engram.constructor",
                lambda: CausalTokenEngram(
                    TokenEngramConfig(
                        hidden_dim=config.d_model,
                        num_slots=config.engram_table_size,
                        ngram_orders=config.engram_orders,
                        num_hash_heads=config.engram_hashes_per_order,
                        head_dim=config.engram_row_dim,
                        initial_scale=config.engram_layer_scale,
                        seed=config.engram_hash_seed,
                    )
                ),
            )
            if config.use_engram
            else None
        )
        self.core_blocks = nn.ModuleList(
            construct(
                f"model.core.{index}.constructor",
                (
                    (
                        lambda index=index: BicameralTransformerBlock(
                            config.d_model,
                            n_heads=config.n_heads,
                            n_kv_heads=config.n_kv_heads,
                            d_ff=config.d_ff,
                            max_sequence_length=config.max_sequence_length,
                            rope_theta=config.rope_theta,
                            norm_eps=config.norm_eps,
                            initialization_seed=config.initialization_seed,
                            module_path=f"model.core.{index}",
                            attention_dropout=config.attention_dropout,
                        )
                    )
                    if config.use_bicameral_core
                    else (
                        lambda index=index: TransformerBlock(
                            config,
                            module_path=f"model.core.{index}",
                        )
                    )
                ),
            )
            for index in range(config.n_core_blocks)
        )
        self.bicameral_combiner = (
            construct(
                "model.bicameral_combiner.constructor",
                lambda: PerBandUnitCircleCombiner(config.d_model, num_bands=8),
            )
            if config.use_bicameral_core
            else None
        )
        self.loop_embedding = (
            construct(
                "model.loop_embedding.constructor",
                lambda: nn.Embedding(config.max_recurrent_steps, config.d_model),
            )
            if config.use_recurrence and not config.use_bicameral_core
            else None
        )
        self.reentry_bridge = (
            construct(
                "model.reentry_bridge.constructor",
                lambda: AnchoredReentryBridge(
                    config.d_model,
                    layer_scale=config.bridge_layer_scale,
                    norm_eps=config.norm_eps,
                ),
            )
            if config.use_reentry_bridge
            else None
        )
        self.scratch = (
            construct(
                "model.scratch.constructor",
                lambda: PositionAlignedScratch(
                    config.d_model,
                    lane_width=config.scratch_width,
                    max_steps=config.max_recurrent_steps,
                    layer_scale=config.scratch_layer_scale,
                    rho_init=config.lane_carrier_rho_init,
                    retention_floor=config.lane_carrier_retention_floor,
                    norm_eps=config.norm_eps,
                    use_carrier=config.use_lane_carrier,
                ),
            )
            if config.use_scratch
            else None
        )
        self.long_term_memory = long_term_memory
        self.coda_blocks = nn.ModuleList(
            construct(
                f"model.coda.{index}.constructor",
                lambda index=index: TransformerBlock(
                    config,
                    module_path=f"model.coda.{index}",
                ),
            )
            for index in range(config.n_coda_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = construct(
            "model.lm_head.constructor",
            lambda: nn.Linear(config.d_model, config.vocab_size, bias=False),
        )
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
            if strict or assign:
                raise RuntimeError(
                    "one tied alias may be absent only for strict=False, assign=False "
                    "shared-tensor loading"
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
        def generator(source_key: str) -> torch.Generator:
            return torch.Generator(device="cpu").manual_seed(
                derive_module_seed(
                    self.config.initialization_seed,
                    source_key,
                    0,
                )
            )

        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
            generator=generator("model.token_embedding.initialization"),
        )
        physical_blocks = (
            self.config.n_prelude_layers
            + self.config.n_core_blocks
            + self.config.n_coda_layers
        )
        residual_std = 0.02 / (2 * physical_blocks) ** 0.5
        staged_blocks = [
            ("prelude", self.prelude_blocks),
        ]
        # Bicameral blocks own isolated, mode-aware initialization.  Applying
        # the inherited dense reset here would both access nonexistent dense
        # attributes and erase the nonzero disagreement initialization.
        if not self.config.use_bicameral_core:
            staged_blocks.append(("core", self.core_blocks))
        staged_blocks.append(("coda", self.coda_blocks))
        for stage, blocks in staged_blocks:
            for index, block in enumerate(blocks):
                block_generator = generator(
                    f"model.{stage}.{index}.initialization"
                )
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
                        generator=block_generator,
                    )
                for projection in residual_outputs:
                    nn.init.normal_(
                        projection.weight,
                        mean=0.0,
                        std=residual_std,
                        generator=block_generator,
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
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.config.vocab_size
        ):
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

    def _run_bicameral_core(
        self,
        prelude: torch.Tensor,
        *,
        lanes: torch.Tensor | None,
        steps: int,
        alpha: float,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        capture_trajectory: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        BicameralRecurrenceReceipt,
        tuple[torch.Tensor, ...],
        int,
        int,
        tuple[str, ...],
    ]:
        """Execute the ratified Step-2 separated-state recurrence.

        A live visit snapshots both hemispheres once at visit entry and builds
        every block's K/V from that same pair.  This is intentionally not a
        block-local refresh: the visit-entry rule is what makes live and static
        exactly identical at ``K=1`` for a multi-block core.
        """

        if not self.config.use_bicameral_core:
            raise RuntimeError("the bicameral core is structurally disabled")
        if self.loop_embedding is not None or self.front_hadamard is not None:
            raise RuntimeError("a retired legacy module entered the bicameral graph")
        if self.reentry_bridge is not None:
            raise RuntimeError("the legacy h0 re-entry bridge entered the bicameral graph")
        policy = self.config.kv_policy
        visit_consensus_states: list[torch.Tensor] = []
        after_block_modules: tuple[str, ...] = ()
        if lanes is not None:
            assert self.scratch is not None
            after_block_modules = ("PositionAlignedScratch.step_bicameral",)
            if self.scratch.carrier is not None:
                after_block_modules = (*after_block_modules, "TwoLaneBirkhoffMixer")

        def after_block(
            h_a: torch.Tensor,
            h_b: torch.Tensor,
            *,
            visit: int,
            block_index: int,
            residual_scale: float,
        ) -> AfterBlockResult:
            nonlocal lanes
            executed_modules: list[str] = []
            if lanes is not None:
                assert self.scratch is not None
                lanes = self.scratch.step_bicameral(
                    lanes,
                    h_a,
                    h_b,
                    step_index=visit,
                    residual_scale=residual_scale,
                )
                executed_modules.append("PositionAlignedScratch.step_bicameral")
                if self.scratch.carrier is not None:
                    executed_modules.append("TwoLaneBirkhoffMixer")
            if capture_trajectory and block_index == len(self.core_blocks) - 1:
                visit_consensus_states.append((h_a + h_b) * 0.5)
            return AfterBlockResult(
                h_a=h_a,
                h_b=h_b,
                executed_modules=tuple(executed_modules),
            )

        recurrent = execute_bicameral_recurrence(
            self.core_blocks,
            prelude,
            prelude,
            recurrent_steps=steps,
            recurrence_c=self.config.recurrence_coefficient,
            projected_kv=None,
            kv_policy=policy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            after_block=after_block,
            expected_after_block_modules=after_block_modules,
        )
        receipt = recurrent.receipt
        if receipt.residual_scale != alpha:
            raise RuntimeError("bicameral recurrence scale disagrees with the model schedule")
        block_count = len(self.core_blocks)
        if policy == "live":
            cache_generation_events = steps
            linear_projection_calls = 4 * block_count * steps
        elif policy == "static" or steps == 1:
            cache_generation_events = 1
            linear_projection_calls = 2 * block_count
        else:
            cache_generation_events = 2
            linear_projection_calls = 6 * block_count
        return (
            recurrent.h_a,
            recurrent.h_b,
            lanes,
            receipt,
            tuple(visit_consensus_states),
            cache_generation_events,
            linear_projection_calls,
            receipt.visit_schedule,
        )

    def _run_recurrent_visit(
        self,
        hidden: torch.Tensor,
        *,
        prelude: torch.Tensor,
        lanes: torch.Tensor | None,
        core_kv_cache: tuple[ProjectedKeyValue, ...] | None = None,
        step_index: int,
        alpha: float,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        force_math_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one complete recurrent transition at a fixed visit index."""

        if self.config.use_static_kv_core:
            if core_kv_cache is None:
                raise ValueError("static core K/V must be projected once by the caller")
            if len(core_kv_cache) != len(self.core_blocks):
                raise ValueError("core K/V cache must contain one entry per core block")
            cache_position_ids = core_kv_cache[0].position_ids
            if any(
                entry.position_ids is not cache_position_ids
                for entry in core_kv_cache[1:]
            ):
                raise ValueError("core K/V entries must share one position-ID receipt")
            if position_ids is None:
                position_ids = cache_position_ids
            elif (
                position_ids is not cache_position_ids
                and not torch.equal(position_ids, cache_position_ids)
            ):
                raise ValueError("core K/V cache position IDs differ from the query")
        elif core_kv_cache is not None:
            raise ValueError("a core K/V cache requires use_static_kv_core=True")

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
        for block_index, block in enumerate(self.core_blocks):
            hidden = block(
                hidden,
                projected_kv=(
                    core_kv_cache[block_index]
                    if core_kv_cache is not None
                    else None
                ),
                _trusted_projected_kv=core_kv_cache is not None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
                residual_scale=alpha,
                force_math_attention=force_math_attention,
                rng_coordinate=step_index,
            )
        return hidden, lanes

    def _project_core_kv(
        self,
        source: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
    ) -> tuple[ProjectedKeyValue, ...]:
        """Project one fixed-context K/V entry for every tied core block."""

        if not self.config.use_static_kv_core:
            raise RuntimeError("static core K/V is structurally disabled")
        batch, length, _ = source.shape
        if position_ids is None:
            position_metadata = torch.arange(length, device=source.device).view(1, -1)
            position_metadata = position_metadata.expand(batch, -1).clone()
        else:
            position_metadata = position_ids.detach().clone()
        return tuple(
            block.project_kv(
                source,
                position_ids=position_metadata,
                _position_metadata=position_metadata,
            )
            for block in self.core_blocks
        )

    def _visit_jacobian_spectral_norm(
        self,
        hidden: torch.Tensor,
        *,
        prelude: torch.Tensor,
        lanes: torch.Tensor | None,
        core_kv_cache: tuple[ProjectedKeyValue, ...] | None,
        step_index: int,
        total_steps: int,
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
            transition_kv_cache = None
            if self.config.use_static_kv_core:
                refresh_at_midpoint = (
                    self.config.static_kv_midpoint_refresh
                    and total_steps >= 2
                    and step_index == total_steps // 2
                )
                if refresh_at_midpoint:
                    transition_kv_cache = self._project_core_kv(
                        state_hidden,
                        position_ids=position_ids,
                    )
                elif core_kv_cache is not None:
                    transition_kv_cache = tuple(
                        ProjectedKeyValue(
                            entry.key.detach(),
                            entry.value.detach(),
                            entry.position_ids,
                            entry.owner_id,
                        )
                        for entry in core_kv_cache
                    )
                else:
                    transition_kv_cache = self._project_core_kv(
                        prelude,
                        position_ids=position_ids,
                    )
            transition_position_ids = (
                transition_kv_cache[0].position_ids
                if transition_kv_cache is not None
                else position_ids
            )
            next_hidden, next_lanes = self._run_recurrent_visit(
                state_hidden,
                prelude=prelude,
                lanes=state_lanes,
                core_kv_cache=transition_kv_cache,
                step_index=step_index,
                alpha=alpha,
                attention_mask=attention_mask,
                position_ids=transition_position_ids,
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
            core_kv_cache = (
                self._project_core_kv(initial_hidden, position_ids=position_ids)
                if self.config.use_static_kv_core
                else None
            )
            transition_position_ids = (
                core_kv_cache[0].position_ids
                if core_kv_cache is not None
                else position_ids
            )
            for step_index in range(steps):
                if (
                    self.config.static_kv_midpoint_refresh
                    and steps >= 2
                    and step_index == steps // 2
                ):
                    core_kv_cache = self._project_core_kv(
                        hidden,
                        position_ids=transition_position_ids,
                    )
                    transition_position_ids = core_kv_cache[0].position_ids
                hidden, lanes = self._run_recurrent_visit(
                    hidden,
                    prelude=initial_hidden,
                    lanes=lanes,
                    core_kv_cache=core_kv_cache,
                    step_index=step_index,
                    alpha=alpha,
                    attention_mask=attention_mask,
                    position_ids=transition_position_ids,
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

        External autoregressive caching remains gated off.  The separate
        static recurrent-core arm projects K/V from the anchor stream once per
        cache generation and carries no recurrent-visit (``K``) multiplier.
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
        if self.config.use_bicameral_core and (
            capture_loop_gradients or jacobian_probe_iterations
        ):
            raise NotImplementedError(
                "separated-state loop-gradient and Jacobian probes remain gated "
                "until the Step-7 live-K/V instrument integration"
            )
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

        engram_audit: dict[str, object] | None = None
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
        core_kv_cache: tuple[ProjectedKeyValue, ...] | None = None
        core_kv_projection_events = 0
        core_kv_linear_projection_calls = 0
        static_kv_elements_per_generation = 0
        static_kv_bytes_per_generation = 0
        static_kv_position_metadata_elements = 0
        static_kv_position_metadata_bytes = 0
        midpoint_refresh_executed = False
        loop_rms: list[torch.Tensor] = []
        loop_update_rms: list[torch.Tensor] = []
        trajectory_states: list[torch.Tensor] = [prelude.detach()] if return_diagnostics else []
        gradient_states: list[torch.Tensor] = []
        jacobian_norms: list[torch.Tensor] = []
        jacobian_cache_semantics: list[tuple[int, str]] = []
        horizon_jacobian_norm: torch.Tensor | None = None
        bicameral_recurrence_receipt: BicameralRecurrenceReceipt | None = None
        bicameral_visit_schedule: tuple[str, ...] = ()
        bicameral_terminal_a: torch.Tensor | None = None
        bicameral_terminal_b: torch.Tensor | None = None
        bicameral_scratch_update_events = 0
        if self.config.use_bicameral_core:
            (
                bicameral_terminal_a,
                bicameral_terminal_b,
                lanes,
                bicameral_recurrence_receipt,
                visit_consensus_states,
                core_kv_projection_events,
                core_kv_linear_projection_calls,
                bicameral_visit_schedule,
            ) = self._run_bicameral_core(
                prelude,
                lanes=lanes,
                steps=steps,
                alpha=alpha,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=effective_document_ids,
                capture_trajectory=return_diagnostics,
            )
            assert self.bicameral_combiner is not None
            hidden = self.bicameral_combiner(
                bicameral_terminal_a,
                bicameral_terminal_b,
            )
            bicameral_visit_schedule = (
                *bicameral_visit_schedule,
                "terminal.PerBandUnitCircleCombiner",
            )
            bicameral_scratch_update_events = (
                steps * len(self.core_blocks) if lanes is not None else 0
            )
            if return_diagnostics:
                prior_state = prelude
                for visit_state in visit_consensus_states:
                    loop_rms.append(
                        self._valid_token_rms(visit_state, valid_token_mask).detach()
                    )
                    loop_update_rms.append(
                        self._valid_token_rms(
                            visit_state.float() - prior_state.float(),
                            valid_token_mask,
                        ).detach()
                    )
                    trajectory_states.append(visit_state.detach())
                    prior_state = visit_state
        else:
            core_kv_cache = (
                self._project_core_kv(prelude, position_ids=position_ids)
                if self.config.use_static_kv_core
                else None
            )
            if core_kv_cache is not None:
                position_ids = core_kv_cache[0].position_ids
            core_kv_projection_events = 1 if core_kv_cache is not None else 0
            static_kv_elements_per_generation = (
                sum(
                    entry.key.numel() + entry.value.numel()
                    for entry in core_kv_cache
                )
                if core_kv_cache is not None
                else 0
            )
            static_kv_bytes_per_generation = (
                sum(
                    entry.key.numel() * entry.key.element_size()
                    + entry.value.numel() * entry.value.element_size()
                    for entry in core_kv_cache
                )
                if core_kv_cache is not None
                else 0
            )
            static_kv_position_metadata_elements = (
                core_kv_cache[0].position_ids.numel()
                if core_kv_cache is not None
                else 0
            )
            static_kv_position_metadata_bytes = (
                core_kv_cache[0].position_ids.numel()
                * core_kv_cache[0].position_ids.element_size()
                if core_kv_cache is not None
                else 0
            )
            midpoint_refresh_executed = (
                self.config.static_kv_midpoint_refresh and steps >= 2
            )
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
                    if not self.config.use_static_kv_core:
                        cache_semantics = "dynamic_kv_total_derivative"
                    elif (
                        self.config.static_kv_midpoint_refresh
                        and steps >= 2
                        and step_index == steps // 2
                    ):
                        cache_semantics = "refresh_cache_total_derivative"
                    else:
                        cache_semantics = "fixed_cache_partial_derivative"
                    jacobian_cache_semantics.append((step_index, cache_semantics))
                    jacobian_norms.append(
                        self._visit_jacobian_spectral_norm(
                            hidden,
                            prelude=prelude,
                            lanes=lanes,
                            core_kv_cache=core_kv_cache,
                            step_index=step_index,
                            total_steps=steps,
                            alpha=alpha,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            document_ids=effective_document_ids,
                            valid_token_mask=valid_token_mask,
                            iterations=jacobian_probe_iterations,
                        ).detach()
                    )
                if (
                    self.config.static_kv_midpoint_refresh
                    and steps >= 2
                    and step_index == steps // 2
                ):
                    core_kv_cache = self._project_core_kv(
                        hidden,
                        position_ids=position_ids,
                    )
                    position_ids = core_kv_cache[0].position_ids
                    core_kv_projection_events += 1
                hidden, lanes = self._run_recurrent_visit(
                    hidden,
                    prelude=prelude,
                    lanes=lanes,
                    core_kv_cache=core_kv_cache,
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
                        raise RuntimeError(
                            "loop gradient capture requires autograd-enabled execution"
                        )
                    hidden.retain_grad()
                    gradient_states.append(hidden)
            core_kv_linear_projection_calls = (
                core_kv_projection_events * len(self.core_blocks) * 2
            )

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

        diagnostics: dict[str, Any] = {
            "composition_receipt": composition_receipt(
                self,
                requested_visits=steps,
                executed_visits=steps,
                kv_policy=(
                    self.config.kv_policy
                    if self.config.use_bicameral_core
                    else None
                ),
                kv_cache_multiplier_at_serving=(
                    bicameral_recurrence_receipt.kv_cache_multiplier_at_serving
                    if bicameral_recurrence_receipt is not None
                    else None
                ),
                visit_schedule=bicameral_visit_schedule,
            ).as_dict()
        }
        if return_diagnostics:
            diagnostics["alpha_t"] = alpha
            diagnostics["recurrence_enabled"] = self.config.use_recurrence
            diagnostics["rng_run_seed"] = self.config.run_seed
            diagnostics["rng_replica"] = self.config.rng_replica
            diagnostics["rng_stream_draw_indices_by_name"] = tuple(
                (name, module.draw_indices)
                for name, module in self.named_modules()
                if isinstance(module, ModuleRNGStream)
            )
            diagnostics["executed_core_visits"] = steps
            diagnostics["executed_core_block_passes"] = steps * len(self.core_blocks)
            diagnostics["bicameral_core_enabled"] = self.config.use_bicameral_core
            diagnostics["loop_state_basis"] = (
                "bicameral_consensus_step2"
                if self.config.use_bicameral_core
                else "single_hidden_stream"
            )
            diagnostics["kv_policy"] = (
                self.config.kv_policy
                if self.config.use_bicameral_core
                else diagnostics["composition_receipt"]["kv_policy"]
            )
            diagnostics["kv_cache_multiplier_at_serving"] = diagnostics[
                "composition_receipt"
            ]["kv_cache_multiplier_at_serving"]
            diagnostics["visit_schedule"] = bicameral_visit_schedule
            diagnostics["bicameral_scratch_update_events"] = (
                bicameral_scratch_update_events
            )
            diagnostics["terminal_s2_combiner_executed"] = (
                bicameral_recurrence_receipt is not None
            )
            if bicameral_recurrence_receipt is not None:
                diagnostics["bicameral_recurrence_receipt"] = asdict(
                    bicameral_recurrence_receipt
                )
                assert bicameral_terminal_a is not None
                assert bicameral_terminal_b is not None
                diagnostics["terminal_hemisphere_disagreement_rms"] = (
                    self._valid_token_rms(
                        bicameral_terminal_a.float() - bicameral_terminal_b.float(),
                        valid_token_mask,
                    ).detach()
                )
            diagnostics["reentry_bridge_requested"] = self.config.use_reentry_bridge
            diagnostics["reentry_bridge_executed_visits"] = (
                max(steps - 1, 0) if self.reentry_bridge is not None else 0
            )
            diagnostics["static_core_kv_enabled"] = self.config.use_static_kv_core
            diagnostics["main_graph_core_kv_projection_events"] = (
                core_kv_projection_events
            )
            diagnostics["main_graph_core_kv_linear_projection_calls"] = (
                core_kv_linear_projection_calls
            )
            diagnostics["static_kv_midpoint_refresh_requested"] = (
                self.config.static_kv_midpoint_refresh
            )
            diagnostics["static_kv_midpoint_refresh_executed"] = (
                midpoint_refresh_executed
            )
            diagnostics["static_kv_midpoint_refresh_visit"] = (
                steps // 2 if midpoint_refresh_executed else None
            )
            diagnostics["static_kv_elements_per_generation"] = (
                static_kv_elements_per_generation
            )
            diagnostics["static_kv_bytes_per_generation"] = (
                static_kv_bytes_per_generation
            )
            diagnostics["static_kv_payload_scope"] = (
                "projected_key_value_only_excludes_shared_position_receipt"
            )
            diagnostics["static_kv_position_metadata_elements_per_generation"] = (
                static_kv_position_metadata_elements
            )
            diagnostics["static_kv_position_metadata_bytes_per_generation"] = (
                static_kv_position_metadata_bytes
            )
            diagnostics["static_kv_total_bytes_per_generation"] = (
                static_kv_bytes_per_generation + static_kv_position_metadata_bytes
            )
            diagnostics["static_kv_cumulative_projected_elements"] = (
                static_kv_elements_per_generation * core_kv_projection_events
            )
            diagnostics["static_kv_peak_elements_upper_bound"] = (
                static_kv_elements_per_generation
                * core_kv_projection_events
            )
            diagnostics["static_kv_peak_total_bytes_upper_bound"] = (
                static_kv_bytes_per_generation + static_kv_position_metadata_bytes
            ) * core_kv_projection_events
            diagnostics["static_kv_receipt_scope"] = (
                "main_graph_only_excludes_jacobian_instrumentation"
            )
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
                diagnostics["local_jacobian_cache_semantics_by_visit"] = tuple(
                    jacobian_cache_semantics
                )
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
