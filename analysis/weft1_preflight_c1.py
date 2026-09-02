"""CPU PRE-FLIGHT C1 width-coordinate audit for the materialized WEFT-1 graph.

C1 is a bug-catching check, not S2 calibration.  This module therefore keeps
the registered width axis and pass threshold literal, records every otherwise
discretionary toy-protocol choice, and refuses to hide a ratified attention
scale mismatch behind aggregate activation statistics.

The production model is not modified here.  In particular, the attention
probe compares the executed graph against both ``1 / d_head`` (ratified muP)
and the ordinary ``1 / sqrt(d_head)`` reference and reports which coordinate
the current implementation actually follows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Callable

import torch
from torch import nn

from models.ablation_lm.bicameral_core import BicameralTransformerBlock
from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.engram import CausalTokenEngram
from models.ablation_lm.layers import (
    GroupedQueryAttention,
    ModifiedHadamardExpertBank,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
)
from models.ablation_lm.liveness import PF1_NOT_MATERIALIZED_INTEGRATIONS
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.model import AblationLM
from models.ablation_lm.reentry import AnchoredReentryBridge
from models.ablation_lm.scratch import TwoLaneBirkhoffMixer


PREFLIGHT_PROGRAM_SHA256 = (
    "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
)
PREFLIGHT_RATIFICATION_SHA256 = (
    "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
)
ATTENTION_SCALE_AUTHORITY = (
    "STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md#8.1"
)
ATTENTION_SCALE_AUTHORITY_BYTES = 61_329
ATTENTION_SCALE_AUTHORITY_SHA256 = (
    "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
)
ATTENTION_SCALE_AUTHORITY_DRIVE_ID = "1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_"
C1_CPU_WIDTHS = (64, 128, 256)
C1_DEFERRED_GPU_WIDTHS = (512,)
C1_WIDTH_DRIFT_LIMIT = 2.0
C1_TRAINING_STEPS = 10
C1_CATCH_NUMBER = 28

_TOY_VOCAB_SIZE = 128
_TOY_BATCH_SIZE = 2
_TOY_SEQUENCE_LENGTH = 8
_TOY_N_HEADS = 8
_TOY_N_KV_HEADS = 4
_TOY_RECURRENT_STEPS = 4
_TOY_MAX_RECURRENT_STEPS = 8
_TOY_LEARNING_RATE = 3.0e-4
_TOY_BETAS = (0.9, 0.95)
_TOY_EPSILON = 1.0e-8
_TOY_WEIGHT_DECAY = 0.0
_TOY_SEED = 20_260_902
_ATTENTION_MATCH_ATOL = 1.0e-5


class C1CoordinateCatch(RuntimeError):
    """Raised when a caller attempts to promote a failed C1 receipt."""


@dataclass(frozen=True)
class AttentionScaleEvidence:
    surface: str
    width: int
    head_dim: int
    ratified_scale: float
    ordinary_scale: float
    ordinary_to_ratified_ratio: float
    sdpa_error_to_ratified: float
    sdpa_error_to_ordinary: float
    math_error_to_ratified: float
    math_error_to_ordinary: float
    sdpa_matches: str
    math_matches: str
    passed: bool


@dataclass(frozen=True)
class ActivationCoordinate:
    module_name: str
    phase: str
    rms_by_width: tuple[tuple[int, float], ...]
    maximum_to_minimum_ratio: float
    all_zero: bool
    passed: bool


@dataclass(frozen=True)
class WidthRun:
    width: int
    head_dim: int
    unique_parameters: int
    unique_decoder_blocks: int
    executed_decoder_block_passes: int
    initial_loss: float
    final_training_loss: float
    activation_rms_at_init: tuple[tuple[str, float], ...]
    activation_rms_after_steps: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class C1PreflightReceipt:
    program_sha256: str
    ratification_sha256: str
    attention_scale_authority: str
    attention_scale_authority_bytes: int
    attention_scale_authority_sha256: str
    attention_scale_authority_drive_id: str
    cpu_widths: tuple[int, ...]
    deferred_gpu_widths: tuple[int, ...]
    width_drift_limit: float
    training_steps: int
    chassis: str
    toy_protocol: str
    not_materialized_integrations: tuple[str, ...]
    attention_scale_evidence: tuple[AttentionScaleEvidence, ...]
    width_runs: tuple[WidthRun, ...]
    activation_coordinates: tuple[ActivationCoordinate, ...]
    failed_activation_coordinates: tuple[tuple[str, str], ...]
    attention_scale_passed: bool
    activation_coordinate_passed: bool
    passed: bool
    catch_number: int | None
    disposition: str
    a100_hours: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def require_passed(self) -> None:
        if self.passed:
            return
        raise C1CoordinateCatch(
            f"CATCH #{self.catch_number}: C1 muP coordinate check failed; "
            f"attention_scale_passed={self.attention_scale_passed}, "
            f"failed_activation_coordinates={self.failed_activation_coordinates}"
        )


@dataclass
class _RMSAccumulator:
    sum_of_squares: float = 0.0
    element_count: int = 0

    def add(self, values: torch.Tensor) -> None:
        values_float = values.detach().float()
        self.sum_of_squares += float(values_float.square().sum().item())
        self.element_count += values_float.numel()

    def rms(self) -> float:
        if self.element_count == 0:
            raise RuntimeError("activation coordinate was registered but never executed")
        return math.sqrt(self.sum_of_squares / self.element_count)


def _toy_config(width: int) -> AblationLMConfig:
    if type(width) is not int or width not in (*C1_CPU_WIDTHS, *C1_DEFERRED_GPU_WIDTHS):
        raise ValueError("width must be one of the registered C1 widths")
    return AblationLMConfig(
        vocab_size=_TOY_VOCAB_SIZE,
        d_model=width,
        n_heads=_TOY_N_HEADS,
        n_kv_heads=_TOY_N_KV_HEADS,
        d_ff=11 * width // 4,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        use_recurrence=True,
        recurrent_steps=_TOY_RECURRENT_STEPS,
        max_recurrent_steps=_TOY_MAX_RECURRENT_STEPS,
        use_static_kv_core=True,
        max_sequence_length=16,
        use_front_hadamard_experts=True,
        hadamard_experts=4,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        scratch_width=width // 8,
        use_engram=True,
        engram_hashes_per_order=2,
        engram_table_size=127,
        engram_row_dim=8,
        use_long_term_memory=True,
        long_term_memory_slots=32,
        long_term_memory_width=width // 8,
        initialization_seed=_TOY_SEED,
        run_seed=_TOY_SEED,
        hadamard_seed=_TOY_SEED,
        engram_hash_seed=_TOY_SEED,
        jet_plane_probe_seed=_TOY_SEED,
    )


def _toy_memory(config: AblationLMConfig) -> ReadOnlyLatentMemory:
    generator = torch.Generator(device="cpu").manual_seed(_TOY_SEED + 1)
    return ReadOnlyLatentMemory(
        config.d_model,
        keys=torch.randn(
            config.long_term_memory_slots,
            config.long_term_memory_width,
            generator=generator,
        ),
        values=torch.randn(
            config.long_term_memory_slots,
            config.long_term_memory_width,
            generator=generator,
        ),
        provenance_ids=torch.arange(config.long_term_memory_slots),
        layer_scale=config.long_term_memory_layer_scale,
        norm_eps=config.norm_eps,
        initialization_seed=config.initialization_seed,
    )


def _toy_model(config: AblationLMConfig) -> AblationLM:
    return AblationLM(config, long_term_memory=_toy_memory(config))


def _toy_batch(config: AblationLMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(_TOY_SEED + 2)
    tokens = torch.randint(
        1,
        config.vocab_size,
        (_TOY_BATCH_SIZE, _TOY_SEQUENCE_LENGTH),
        generator=generator,
    )
    record_ids = torch.arange(_TOY_BATCH_SIZE).view(-1, 1).expand_as(tokens).clone()
    return tokens, record_ids


def _semantic_activation_modules(
    model: AblationLM,
) -> tuple[tuple[str, nn.Module, Callable[[object], torch.Tensor]], ...]:
    """Return comparable, executed module boundaries for every materialized arm."""

    first_tensor = lambda output: output[0] if isinstance(output, tuple) else output
    identity = lambda output: output
    modules: list[tuple[str, nn.Module, Callable[[object], torch.Tensor]]] = [
        ("token_embedding", model.token_embedding, identity),
    ]
    if model.front_hadamard is not None:
        assert isinstance(model.front_hadamard, ModifiedHadamardExpertBank)
        modules.append(("front_hadamard", model.front_hadamard, first_tensor))
    for stage, blocks in (
        ("prelude", model.prelude_blocks),
        ("core", model.core_blocks),
        ("coda", model.coda_blocks),
    ):
        for index, block in enumerate(blocks):
            assert isinstance(block, TransformerBlock)
            assert isinstance(block.attention, GroupedQueryAttention)
            assert isinstance(block.feed_forward, SwiGLU)
            modules.extend(
                (
                    (f"{stage}.{index}.attention", block.attention, identity),
                    (f"{stage}.{index}.feed_forward", block.feed_forward, identity),
                    (f"{stage}.{index}.block_output", block, identity),
                )
            )
    if model.engram is not None:
        assert isinstance(model.engram, CausalTokenEngram)
        modules.append(("engram", model.engram, first_tensor))
    if model.reentry_bridge is not None:
        assert isinstance(model.reentry_bridge, AnchoredReentryBridge)
        modules.append(("reentry_bridge", model.reentry_bridge, identity))
    if model.scratch is not None:
        modules.extend(
            (
                ("scratch.initializer", model.scratch.initializer, identity),
                ("scratch.context_projection", model.scratch.context_projection, identity),
                ("scratch.update_out", model.scratch.update_out, identity),
                ("scratch.readout", model.scratch.readout, identity),
            )
        )
        if model.scratch.carrier is not None:
            assert isinstance(model.scratch.carrier, TwoLaneBirkhoffMixer)
            modules.append(("scratch.carrier", model.scratch.carrier, identity))
    if model.long_term_memory is not None:
        assert isinstance(model.long_term_memory, ReadOnlyLatentMemory)
        modules.append(("long_term_memory", model.long_term_memory, first_tensor))
    assert isinstance(model.final_norm, RMSNorm)
    modules.extend(
        (
            ("final_norm", model.final_norm, identity),
            ("lm_head", model.lm_head, identity),
        )
    )
    return tuple(modules)


def _activation_snapshot(
    model: AblationLM,
    tokens: torch.Tensor,
    record_ids: torch.Tensor,
) -> tuple[tuple[str, float], ...]:
    accumulators: dict[str, _RMSAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module, selector in _semantic_activation_modules(model):
        if name in accumulators:
            raise RuntimeError(f"duplicate activation coordinate {name}")
        accumulators[name] = _RMSAccumulator()

        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            coordinate: str = name,
            select: Callable[[object], torch.Tensor] = selector,
        ) -> None:
            selected = select(output)
            if not isinstance(selected, torch.Tensor) or not selected.is_floating_point():
                raise TypeError(
                    f"activation coordinate {coordinate} did not return a float tensor"
                )
            accumulators[coordinate].add(selected)

        handles.append(module.register_forward_hook(hook))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(
                tokens,
                memory_record_ids=record_ids,
                recurrent_steps=_TOY_RECURRENT_STEPS,
            )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if model.loop_embedding is None:
        raise RuntimeError("C1 chassis requires the recurrent loop embedding")
    loop_update = (
        model.loop_embedding.weight[:_TOY_RECURRENT_STEPS]
        * model.config.recurrence_scale(_TOY_RECURRENT_STEPS)
    )
    accumulators["loop_embedding.applied_update"] = _RMSAccumulator()
    accumulators["loop_embedding.applied_update"].add(loop_update)
    return tuple(
        (name, accumulators[name].rms())
        for name in sorted(accumulators)
    )


def _reference_attention(
    attention: GroupedQueryAttention,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    query = attention.query_norm(
        attention._split_heads(attention.q_proj(hidden), attention.n_heads)
    )
    query = attention.rope.apply_rotary(query, position_ids)
    projected = attention.project_kv(hidden, position_ids=position_ids)
    key = projected.key.repeat_interleave(
        attention.n_heads // attention.n_kv_heads,
        dim=1,
    )
    value = projected.value.repeat_interleave(
        attention.n_heads // attention.n_kv_heads,
        dim=1,
    )
    scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale)
    length = hidden.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.view(1, 1, length, length), float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    attended = probabilities @ value
    merged = attended.transpose(1, 2).contiguous().view(
        hidden.shape[0],
        length,
        attention.d_model,
    )
    return attention.output_proj(merged)


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _scale_match_label(ratified_error: float, ordinary_error: float) -> str:
    if ratified_error <= _ATTENTION_MATCH_ATOL and ratified_error < ordinary_error:
        return "ratified_inverse_head_dim"
    if ordinary_error <= _ATTENTION_MATCH_ATOL and ordinary_error < ratified_error:
        return "ordinary_inverse_sqrt_head_dim"
    return "neither_registered_scale"


def _attention_scale_evidence(
    model: AblationLM,
    tokens: torch.Tensor,
) -> AttentionScaleEvidence:
    attention = model.prelude_blocks[0].attention
    hidden = model.token_embedding(tokens)
    positions = torch.arange(tokens.shape[1]).view(1, -1).expand(tokens.shape[0], -1)
    ratified_scale = 1.0 / attention.head_dim
    ordinary_scale = 1.0 / math.sqrt(attention.head_dim)
    with torch.no_grad():
        sdpa = attention(hidden, position_ids=positions)
        math_path = attention(hidden, position_ids=positions, force_math_attention=True)
        ratified = _reference_attention(
            attention,
            hidden,
            positions,
            scale=ratified_scale,
        )
        ordinary = _reference_attention(
            attention,
            hidden,
            positions,
            scale=ordinary_scale,
        )
    sdpa_ratified = _maximum_error(sdpa, ratified)
    sdpa_ordinary = _maximum_error(sdpa, ordinary)
    math_ratified = _maximum_error(math_path, ratified)
    math_ordinary = _maximum_error(math_path, ordinary)
    sdpa_matches = _scale_match_label(sdpa_ratified, sdpa_ordinary)
    math_matches = _scale_match_label(math_ratified, math_ordinary)
    return AttentionScaleEvidence(
        surface="integrated_grouped_query_attention",
        width=model.config.d_model,
        head_dim=attention.head_dim,
        ratified_scale=ratified_scale,
        ordinary_scale=ordinary_scale,
        ordinary_to_ratified_ratio=ordinary_scale / ratified_scale,
        sdpa_error_to_ratified=sdpa_ratified,
        sdpa_error_to_ordinary=sdpa_ordinary,
        math_error_to_ratified=math_ratified,
        math_error_to_ordinary=math_ordinary,
        sdpa_matches=sdpa_matches,
        math_matches=math_matches,
        passed=(
            sdpa_matches == "ratified_inverse_head_dim"
            and math_matches == "ratified_inverse_head_dim"
        ),
    )


def _reference_bicameral_attention(
    block: BicameralTransformerBlock,
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    hemi: int,
    scale: float,
) -> torch.Tensor:
    query = block.query_norm(
        block._split_heads(
            block.q_proj(block.attention_norm(hidden), hemi),
            block.n_heads,
        )
    )
    query = block.rope.apply_rotary(query, position_ids)
    key = key.repeat_interleave(block.n_heads // block.n_kv_heads, dim=1)
    value = value.repeat_interleave(block.n_heads // block.n_kv_heads, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale)
    length = hidden.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.view(1, 1, length, length), float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    attended = probabilities @ value
    merged = attended.transpose(1, 2).contiguous().view(
        hidden.shape[0],
        length,
        block.d_model,
    )
    return block.o_proj(merged, hemi)


def _bicameral_attention_scale_evidence(width: int) -> AttentionScaleEvidence:
    block = BicameralTransformerBlock(
        width,
        n_heads=_TOY_N_HEADS,
        n_kv_heads=_TOY_N_KV_HEADS,
        d_ff=11 * width // 4,
        max_sequence_length=16,
        rank=8,
        initialization_seed=_TOY_SEED,
        module_path=f"preflight.c1.bicameral.d{width}",
    ).eval()
    generator = torch.Generator(device="cpu").manual_seed(_TOY_SEED + width)
    hidden = 0.02 * torch.randn(
        _TOY_BATCH_SIZE,
        _TOY_SEQUENCE_LENGTH,
        width,
        generator=generator,
    )
    anchor = 0.02 * torch.randn(
        _TOY_BATCH_SIZE,
        _TOY_SEQUENCE_LENGTH,
        width,
        generator=generator,
    )
    positions = torch.arange(_TOY_SEQUENCE_LENGTH).view(1, -1).expand(_TOY_BATCH_SIZE, -1)
    ratified_scale = 1.0 / block.head_dim
    ordinary_scale = 1.0 / math.sqrt(block.head_dim)
    with torch.no_grad():
        cache = block.project_kv(anchor, position_ids=positions)
        sdpa = block._attention(
            hidden,
            hemi=+1,
            key=cache.key_a,
            value=cache.value_a,
            position_ids=positions,
            attention_mask=None,
            document_ids=None,
            force_math_attention=False,
        )
        math_path = block._attention(
            hidden,
            hemi=+1,
            key=cache.key_a,
            value=cache.value_a,
            position_ids=positions,
            attention_mask=None,
            document_ids=None,
            force_math_attention=True,
        )
        ratified = _reference_bicameral_attention(
            block,
            hidden,
            cache.key_a,
            cache.value_a,
            positions,
            hemi=+1,
            scale=ratified_scale,
        )
        ordinary = _reference_bicameral_attention(
            block,
            hidden,
            cache.key_a,
            cache.value_a,
            positions,
            hemi=+1,
            scale=ordinary_scale,
        )
    sdpa_ratified = _maximum_error(sdpa, ratified)
    sdpa_ordinary = _maximum_error(sdpa, ordinary)
    math_ratified = _maximum_error(math_path, ratified)
    math_ordinary = _maximum_error(math_path, ordinary)
    sdpa_matches = _scale_match_label(sdpa_ratified, sdpa_ordinary)
    math_matches = _scale_match_label(math_ratified, math_ordinary)
    return AttentionScaleEvidence(
        surface="standalone_bicameral_transformer_block",
        width=width,
        head_dim=block.head_dim,
        ratified_scale=ratified_scale,
        ordinary_scale=ordinary_scale,
        ordinary_to_ratified_ratio=ordinary_scale / ratified_scale,
        sdpa_error_to_ratified=sdpa_ratified,
        sdpa_error_to_ordinary=sdpa_ordinary,
        math_error_to_ratified=math_ratified,
        math_error_to_ordinary=math_ordinary,
        sdpa_matches=sdpa_matches,
        math_matches=math_matches,
        passed=(
            sdpa_matches == "ratified_inverse_head_dim"
            and math_matches == "ratified_inverse_head_dim"
        ),
    )


def _run_width(
    width: int,
    *,
    training_steps: int,
) -> tuple[WidthRun, tuple[AttentionScaleEvidence, ...]]:
    config = _toy_config(width)
    model = _toy_model(config)
    tokens, record_ids = _toy_batch(config)
    attention_evidence = (
        _attention_scale_evidence(model, tokens),
        _bicameral_attention_scale_evidence(width),
    )
    initial_activations = _activation_snapshot(model, tokens, record_ids)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_TOY_LEARNING_RATE,
        betas=_TOY_BETAS,
        eps=_TOY_EPSILON,
        weight_decay=_TOY_WEIGHT_DECAY,
    )
    initial_loss = math.nan
    final_loss = math.nan
    for step in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tokens,
            labels=tokens,
            memory_record_ids=record_ids,
            recurrent_steps=_TOY_RECURRENT_STEPS,
        )
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"non-finite C1 toy loss at width={width}, step={step}")
        loss_value = float(output.loss.detach().item())
        if step == 0:
            initial_loss = loss_value
        final_loss = loss_value
        output.loss.backward()
        optimizer.step()
    final_activations = _activation_snapshot(model, tokens, record_ids)
    return (
        WidthRun(
            width=width,
            head_dim=config.head_dim,
            unique_parameters=sum(parameter.numel() for parameter in model.parameters()),
            unique_decoder_blocks=(
                config.n_prelude_layers
                + config.n_core_blocks
                + config.n_coda_layers
            ),
            executed_decoder_block_passes=(
                config.n_prelude_layers
                + config.recurrent_steps * config.n_core_blocks
                + config.n_coda_layers
            ),
            initial_loss=initial_loss,
            final_training_loss=final_loss,
            activation_rms_at_init=initial_activations,
            activation_rms_after_steps=final_activations,
        ),
        attention_evidence,
    )


def _coordinate_ratio(values: tuple[float, ...]) -> tuple[float, bool]:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return math.inf, False
    if all(value == 0.0 for value in values):
        return 1.0, True
    minimum = min(values)
    if minimum == 0.0:
        return math.inf, False
    return max(values) / minimum, False


def _activation_coordinates(width_runs: tuple[WidthRun, ...]) -> tuple[ActivationCoordinate, ...]:
    if not width_runs:
        raise ValueError("at least one width run is required")
    phase_rows = (
        ("init", tuple(dict(run.activation_rms_at_init) for run in width_runs)),
        (
            "after_steps",
            tuple(dict(run.activation_rms_after_steps) for run in width_runs),
        ),
    )
    expected_names = set(phase_rows[0][1][0])
    for _phase, rows in phase_rows:
        if any(set(row) != expected_names for row in rows):
            raise RuntimeError("activation coordinate names differ across widths or phases")
    coordinates: list[ActivationCoordinate] = []
    for phase, rows in phase_rows:
        for name in sorted(expected_names):
            rms_by_width = tuple(
                (run.width, rows[index][name])
                for index, run in enumerate(width_runs)
            )
            ratio, all_zero = _coordinate_ratio(
                tuple(value for _width, value in rms_by_width)
            )
            coordinates.append(
                ActivationCoordinate(
                    module_name=name,
                    phase=phase,
                    rms_by_width=rms_by_width,
                    maximum_to_minimum_ratio=ratio,
                    all_zero=all_zero,
                    passed=ratio < C1_WIDTH_DRIFT_LIMIT,
                )
            )
    return tuple(coordinates)


def run_preflight_c1(
    *,
    widths: tuple[int, ...] = C1_CPU_WIDTHS,
    training_steps: int = C1_TRAINING_STEPS,
) -> C1PreflightReceipt:
    """Run the deterministic CPU slice and return a fail-closed C1 receipt."""

    if not widths or len(set(widths)) != len(widths):
        raise ValueError("widths must be a non-empty unique tuple")
    if any(width not in C1_CPU_WIDTHS for width in widths):
        raise ValueError("this runner is CPU-scoped to d in {64,128,256}")
    if type(training_steps) is not int or training_steps < 1:
        raise ValueError("training_steps must be a positive integer")
    width_runs: list[WidthRun] = []
    attention_evidence: list[AttentionScaleEvidence] = []
    for width in widths:
        run, evidence = _run_width(width, training_steps=training_steps)
        width_runs.append(run)
        attention_evidence.extend(evidence)
    runs = tuple(width_runs)
    coordinates = _activation_coordinates(runs)
    failed_coordinates = tuple(
        (coordinate.phase, coordinate.module_name)
        for coordinate in coordinates
        if not coordinate.passed
    )
    attention_passed = all(item.passed for item in attention_evidence)
    activation_passed = not failed_coordinates
    passed = attention_passed and activation_passed
    return C1PreflightReceipt(
        program_sha256=PREFLIGHT_PROGRAM_SHA256,
        ratification_sha256=PREFLIGHT_RATIFICATION_SHA256,
        attention_scale_authority=ATTENTION_SCALE_AUTHORITY,
        attention_scale_authority_bytes=ATTENTION_SCALE_AUTHORITY_BYTES,
        attention_scale_authority_sha256=ATTENTION_SCALE_AUTHORITY_SHA256,
        attention_scale_authority_drive_id=ATTENTION_SCALE_AUTHORITY_DRIVE_ID,
        cpu_widths=widths,
        deferred_gpu_widths=C1_DEFERRED_GPU_WIDTHS,
        width_drift_limit=C1_WIDTH_DRIFT_LIMIT,
        training_steps=training_steps,
        chassis=(
            "materialized_AblationLM_d_variable_heads8_kv4_ff11d_over4_"
            "prelude4_core2_coda4_K4_lanes2xd_over8"
        ),
        toy_protocol=(
            "cpu_fp32_synthetic_fixed_token_batch_b2_s8_v128; AdamW(lr=3e-4,"
            "betas=(0.9,0.95),eps=1e-8,weight_decay=0); "
            "diagnostic_only_not_S2_calibration"
        ),
        not_materialized_integrations=PF1_NOT_MATERIALIZED_INTEGRATIONS,
        attention_scale_evidence=tuple(attention_evidence),
        width_runs=runs,
        activation_coordinates=coordinates,
        failed_activation_coordinates=failed_coordinates,
        attention_scale_passed=attention_passed,
        activation_coordinate_passed=activation_passed,
        passed=passed,
        catch_number=None if passed else C1_CATCH_NUMBER,
        disposition=(
            "c1_cpu_pass_widest_gpu_deferred"
            if passed
            else "catch_28_return_to_strategy_no_model_patch"
        ),
    )


def main() -> None:
    print(json.dumps(run_preflight_c1().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTENTION_SCALE_AUTHORITY",
    "ATTENTION_SCALE_AUTHORITY_BYTES",
    "ATTENTION_SCALE_AUTHORITY_DRIVE_ID",
    "ATTENTION_SCALE_AUTHORITY_SHA256",
    "ActivationCoordinate",
    "AttentionScaleEvidence",
    "C1CoordinateCatch",
    "C1PreflightReceipt",
    "C1_CPU_WIDTHS",
    "C1_DEFERRED_GPU_WIDTHS",
    "C1_TRAINING_STEPS",
    "C1_WIDTH_DRIFT_LIMIT",
    "PREFLIGHT_PROGRAM_SHA256",
    "PREFLIGHT_RATIFICATION_SHA256",
    "WidthRun",
    "run_preflight_c1",
]
