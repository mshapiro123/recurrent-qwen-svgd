"""CPU-valid portion of WEFT-1 PRE-FLIGHT C2 loop-precision measurement.

The governing C2 row asks for a K=8 bf16/full-toy comparison against fp32
masters.  This module deliberately measures only the graph that exists today.
The learned rotor carrier, per-band callosum, and loop sidecar are not integrated,
so this receipt cannot decide rotor-carrier accumulation or claim that the full
WEFT-1 toy chassis passed.

The ratified ``1e-2 relative`` literal is preserved in the receipt.  It is not
turned into a pass/fail gate because the authority does not bind the relative
metric, denominator, tensor population, gradient target, or visit aggregation.
Those missing literals are returned as a catch instead of being inferred here.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch

from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.model import AblationLM


C2_AUTHORITY = "docs/STRATEGY_PREFLIGHT_PROGRAM_20260902.md#3-C2"
C2_AUTHORITY_SHA256 = (
    "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
)
C2_AUTHORITY_BYTES = 15_575
C2_RATIFICATION_AUTHORITY = "docs/STRATEGY_PREFLIGHT_RATIFICATION_20260902.md"
C2_RATIFICATION_SHA256 = (
    "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
)
C2_RATIFICATION_BYTES = 2_233
C2_RATIFIED_VISITS = 8
C2_REGISTERED_RELATIVE_THRESHOLD = 1e-2
C2_ROOT_SEED = 20_260_902
C2_GRADIENT_PARAMETER = "core_blocks.0.attention.q_proj.weight"

C2_CURRENT_INTEGRATED_MODULES = (
    "modified_hadamard_expert_bank",
    "static_kv_recurrent_core",
    "anchored_reentry_bridge",
    "position_aligned_scratch",
    "two_lane_birkhoff_scratch_carrier",
    "causal_token_engram",
    "read_only_long_term_memory",
)
C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS = (
    "full_width_bicameral_recurrent_block",
    "integrated_learned_rotor_carrier",
    "per_band_callosum",
    "conditional_loop_sidecar",
    "final_post_loop_bridge_out",
)
C2_REQUIRED_METRIC_RULINGS = (
    "bind the fixed toy weight state or synthetic-update protocol and input/loss panel",
    "bind whether 1e-2 gates vector relative-L2, scalar norm drift, cosine loss, or another metric",
    "bind the denominator or zero-reference policy for every relative metric",
    "bind whether the gate is per visit, maximum over visits, or final K=8 only",
    "bind the gradient loss, parameter/tensor population, and aggregation",
    "bind the production bf16 backend; CPU autocast is not GPU evaluator identity",
)


@dataclass(frozen=True)
class TensorDrift:
    """Candidate descriptive metrics; none is silently promoted to the C2 gate."""

    reference_dtype: str
    bf16_compute_dtype: str
    reference_l2: float
    bf16_compute_l2: float
    relative_l2_error: float
    relative_norm_drift: float
    cosine_similarity: float


@dataclass(frozen=True)
class C2VisitDrift:
    visit: int
    hidden: TensorDrift
    scratch_lanes: TensorDrift
    logits: TensorDrift
    gradient: TensorDrift
    fp32_loss: float
    bf16_compute_loss: float
    relative_loss_drift: float


@dataclass(frozen=True)
class DeferredC2Cell:
    cell: str
    status: str
    reason: str


@dataclass(frozen=True)
class C2DriftSummary:
    max_hidden_relative_l2: float
    max_hidden_relative_l2_visit: int
    max_scratch_lane_relative_l2: float
    max_scratch_lane_relative_l2_visit: int
    max_logit_relative_l2: float
    max_logit_relative_l2_visit: int
    max_gradient_relative_l2: float
    max_gradient_relative_l2_visit: int
    max_relative_loss_drift: float
    max_relative_loss_drift_visit: int
    candidate_relative_l2_crossings_of_registered_literal: tuple[tuple[str, int], ...]
    crossing_semantics: str


@dataclass(frozen=True)
class C2PreflightReceipt:
    authority: str
    authority_sha256: str
    authority_bytes: int
    ratification_authority: str
    ratification_sha256: str
    ratification_bytes: int
    authority_byte_verified: bool
    measurement_status: str
    current_composition: str
    current_integrated_modules: tuple[str, ...]
    representative_missing_full_toy_integrations: tuple[str, ...]
    full_weft1_toy_step_claim: bool
    carrier_accumulation_decision: str
    weight_state: str
    root_seed: int
    config_identity_sha256: str
    input_panel_sha256: str
    initial_model_state_sha256: str
    training_performed: bool
    checkpoint_used: bool
    reference_policy: str
    bf16_policy: str
    per_visit_readout_definition: str
    gradient_definition: str
    visits: int
    block_split: tuple[int, int, int]
    d_model: int
    scratch_shape: tuple[int, int]
    model_parameters: int
    gradient_parameter: str
    trace_matches_main_forward_fp32: bool
    per_visit: tuple[C2VisitDrift, ...]
    summary: C2DriftSummary
    registered_relative_threshold: float
    threshold_source_ratified: bool
    threshold_metric_binding_status: str
    threshold_applied: bool
    threshold_passed: None
    required_metric_rulings: tuple[str, ...]
    deferred_gpu_cells: tuple[DeferredC2Cell, ...]
    cpu_runtime: str
    torch_version: str
    a100_hours: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _VisitTrace:
    hidden: torch.Tensor
    lanes: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor
    gradient: torch.Tensor


def _verify_authority_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        (
            root / "docs" / "STRATEGY_PREFLIGHT_PROGRAM_20260902.md",
            C2_AUTHORITY_BYTES,
            C2_AUTHORITY_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_RATIFICATION_20260902.md",
            C2_RATIFICATION_BYTES,
            C2_RATIFICATION_SHA256,
        ),
    )
    for path, expected_bytes, expected_sha256 in expected:
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"C2 authority drift at {path.name}: bytes={len(payload)}, "
                f"sha256={actual_sha256}"
            )


def c2_current_toy_config() -> AblationLMConfig:
    """Return the d=64, 4/2/4, K=8 materialized-current toy composition."""

    return AblationLMConfig(
        vocab_size=128,
        d_model=64,
        n_heads=8,
        n_kv_heads=4,
        d_ff=176,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        use_recurrence=True,
        recurrent_steps=C2_RATIFIED_VISITS,
        max_recurrent_steps=C2_RATIFIED_VISITS,
        use_static_kv_core=True,
        max_sequence_length=16,
        initialization_seed=C2_ROOT_SEED,
        run_seed=C2_ROOT_SEED,
        use_front_hadamard_experts=True,
        hadamard_experts=8,
        hadamard_seed=C2_ROOT_SEED,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        scratch_width=8,
        use_engram=True,
        engram_hashes_per_order=2,
        engram_table_size=127,
        engram_row_dim=4,
        engram_hash_seed=C2_ROOT_SEED,
        use_long_term_memory=True,
        long_term_memory_slots=16,
        long_term_memory_width=8,
        jet_plane_probe_seed=C2_ROOT_SEED,
    )


def build_c2_current_toy_model() -> AblationLM:
    """Build current integrated modules only; this never fabricates absent arms."""

    config = c2_current_toy_config()
    generator = torch.Generator(device="cpu").manual_seed(C2_ROOT_SEED + 1)
    memory = ReadOnlyLatentMemory(
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
    return AblationLM(config, long_term_memory=memory)


def c2_fixed_batch(config: AblationLMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a synthetic fixed token panel and leave-one-record-out IDs."""

    if config.vocab_size < 17 or config.long_term_memory_slots < 2:
        raise ValueError("C2 fixed batch requires vocab>=17 and at least two memory records")
    tokens = torch.tensor(
        (
            (1, 3, 5, 7, 9, 11, 13, 15),
            (2, 4, 6, 8, 10, 12, 14, 16),
        ),
        dtype=torch.long,
    )
    record_ids = torch.tensor(((0,) * 8, (1,) * 8), dtype=torch.long)
    return tokens, record_ids


def _named_tensor_sha256(items: tuple[tuple[str, torch.Tensor], ...]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _config_identity_sha256(config: AblationLMConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def _precision_context(policy: str) -> Iterator[None]:
    if policy == "cpu_fp32_reference":
        yield
        return
    if policy == "cpu_fp32_master_bf16_autocast":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
        return
    raise ValueError(f"unsupported C2 precision policy: {policy}")


def _gradient_parameter(model: AblationLM) -> torch.nn.Parameter:
    parameters = dict(model.named_parameters())
    if C2_GRADIENT_PARAMETER not in parameters:
        raise RuntimeError(f"missing C2 gradient parameter {C2_GRADIENT_PARAMETER}")
    parameter = parameters[C2_GRADIENT_PARAMETER]
    if not parameter.requires_grad:
        raise RuntimeError("C2 gradient parameter is not trainable")
    return parameter


def _trace_visits(
    model: AblationLM,
    tokens: torch.Tensor,
    record_ids: torch.Tensor,
    *,
    policy: str,
) -> tuple[_VisitTrace, ...]:
    """Replay the current main graph and attach a read-only full readout each visit."""

    config = model.config
    if not config.use_recurrence or config.recurrent_steps != C2_RATIFIED_VISITS:
        raise ValueError("C2 trace requires the registered K=8 recurrent configuration")
    if not config.use_static_kv_core:
        raise ValueError("C2 current-composition trace expects the materialized static-K/V arm")
    if model.scratch is None or model.long_term_memory is None:
        raise ValueError("C2 current-composition trace requires materialized scratch and memory")
    parameter = _gradient_parameter(model)
    model.zero_grad(set_to_none=True)

    with _precision_context(policy):
        hidden = model.token_embedding(tokens)
        if model.front_hadamard is not None:
            hidden = model.front_hadamard(hidden)
        for index, block in enumerate(model.prelude_blocks):
            hidden = block(hidden)
            if index == 0 and model.engram is not None:
                hidden, _ = model.engram(hidden, tokens, document_ids=None, enabled=True)
        prelude = hidden
        lanes = model.scratch.initialize(prelude)
        core_kv_cache = model._project_core_kv(prelude, position_ids=None)
        position_ids = core_kv_cache[0].position_ids
        alpha = config.recurrence_scale(C2_RATIFIED_VISITS)

        states: list[torch.Tensor] = []
        lane_states: list[torch.Tensor] = []
        logits_by_visit: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for step_index in range(C2_RATIFIED_VISITS):
            if config.static_kv_midpoint_refresh and step_index == C2_RATIFIED_VISITS // 2:
                core_kv_cache = model._project_core_kv(hidden, position_ids=position_ids)
                position_ids = core_kv_cache[0].position_ids
            hidden, lanes = model._run_recurrent_visit(
                hidden,
                prelude=prelude,
                lanes=lanes,
                core_kv_cache=core_kv_cache,
                step_index=step_index,
                alpha=alpha,
                attention_mask=None,
                position_ids=position_ids,
                document_ids=None,
            )
            assert lanes is not None
            readout, _ = model.long_term_memory(hidden, record_ids=record_ids)
            for block in model.coda_blocks:
                readout = block(readout, position_ids=position_ids)
            logits = model.lm_head(model.final_norm(readout))
            loss = model._language_model_loss(logits, tokens, None, None)
            states.append(hidden)
            lane_states.append(lanes)
            logits_by_visit.append(logits)
            losses.append(loss)

        gradients = tuple(
            torch.autograd.grad(
                loss,
                parameter,
                retain_graph=visit_index + 1 < C2_RATIFIED_VISITS,
                allow_unused=False,
            )[0]
            for visit_index, loss in enumerate(losses)
        )

    return tuple(
        _VisitTrace(
            hidden=hidden_state.detach().clone(),
            lanes=lane_state.detach().clone(),
            logits=logits.detach().clone(),
            loss=loss.detach().clone(),
            gradient=gradient.detach().clone(),
        )
        for hidden_state, lane_state, logits, loss, gradient in zip(
            states,
            lane_states,
            logits_by_visit,
            losses,
            gradients,
            strict=True,
        )
    )


def _tensor_drift(reference: torch.Tensor, observed: torch.Tensor) -> TensorDrift:
    if reference.shape != observed.shape:
        raise ValueError("C2 drift tensors must have identical shapes")
    reference64 = reference.detach().double().reshape(-1)
    observed64 = observed.detach().double().reshape(-1)
    if not bool(torch.isfinite(reference64).all()) or not bool(torch.isfinite(observed64).all()):
        raise ValueError("C2 drift tensors must be finite")
    reference_l2 = torch.linalg.vector_norm(reference64)
    observed_l2 = torch.linalg.vector_norm(observed64)
    if reference_l2.item() == 0.0 or observed_l2.item() == 0.0:
        raise ValueError("C2 relative/cosine metrics require nonzero reference and observed tensors")
    difference = torch.linalg.vector_norm(observed64 - reference64)
    cosine = torch.dot(reference64, observed64) / (reference_l2 * observed_l2)
    return TensorDrift(
        reference_dtype=str(reference.dtype).removeprefix("torch."),
        bf16_compute_dtype=str(observed.dtype).removeprefix("torch."),
        reference_l2=float(reference_l2),
        bf16_compute_l2=float(observed_l2),
        relative_l2_error=float(difference / reference_l2),
        relative_norm_drift=float((observed_l2 - reference_l2).abs() / reference_l2),
        cosine_similarity=float(cosine),
    )


def _relative_scalar_drift(reference: torch.Tensor, observed: torch.Tensor) -> float:
    reference_value = float(reference.double())
    observed_value = float(observed.double())
    if not math.isfinite(reference_value) or not math.isfinite(observed_value):
        raise ValueError("C2 loss values must be finite")
    if reference_value == 0.0:
        raise ValueError("C2 relative loss drift requires a nonzero fp32 loss")
    return abs(observed_value - reference_value) / abs(reference_value)


def _summarize_drift(per_visit: tuple[C2VisitDrift, ...]) -> C2DriftSummary:
    if len(per_visit) != C2_RATIFIED_VISITS:
        raise ValueError("C2 summary requires exactly eight visit measurements")

    def maximum(selector: str) -> tuple[float, int]:
        values = tuple(
            (
                getattr(visit, selector).relative_l2_error
                if selector != "relative_loss_drift"
                else visit.relative_loss_drift
            )
            for visit in per_visit
        )
        index = max(range(len(values)), key=values.__getitem__)
        return values[index], per_visit[index].visit

    hidden, hidden_visit = maximum("hidden")
    lanes, lanes_visit = maximum("scratch_lanes")
    logits, logits_visit = maximum("logits")
    gradient, gradient_visit = maximum("gradient")
    loss, loss_visit = maximum("relative_loss_drift")
    crossings = tuple(
        (name, visit.visit)
        for visit in per_visit
        for name, value in (
            ("hidden", visit.hidden.relative_l2_error),
            ("scratch_lanes", visit.scratch_lanes.relative_l2_error),
            ("logits", visit.logits.relative_l2_error),
            ("gradient", visit.gradient.relative_l2_error),
            ("loss", visit.relative_loss_drift),
        )
        if value > C2_REGISTERED_RELATIVE_THRESHOLD
    )
    return C2DriftSummary(
        max_hidden_relative_l2=hidden,
        max_hidden_relative_l2_visit=hidden_visit,
        max_scratch_lane_relative_l2=lanes,
        max_scratch_lane_relative_l2_visit=lanes_visit,
        max_logit_relative_l2=logits,
        max_logit_relative_l2_visit=logits_visit,
        max_gradient_relative_l2=gradient,
        max_gradient_relative_l2_visit=gradient_visit,
        max_relative_loss_drift=loss,
        max_relative_loss_drift_visit=loss_visit,
        candidate_relative_l2_crossings_of_registered_literal=crossings,
        crossing_semantics=(
            "descriptive_only_until_strategy_binds_relative_l2_and_tensor_population"
        ),
    )


def run_preflight_c2() -> C2PreflightReceipt:
    """Measure CPU fp32-master/bf16-autocast drift without minting missing gates."""

    _verify_authority_bytes()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = build_c2_current_toy_model().cpu().eval()
        tokens, record_ids = c2_fixed_batch(model.config)
        config_identity_sha256 = _config_identity_sha256(model.config)
        input_panel_sha256 = _named_tensor_sha256(
            (("tokens", tokens), ("record_ids", record_ids))
        )
        initial_model_state_sha256 = _named_tensor_sha256(
            tuple((name, value) for name, value in model.state_dict().items())
        )
        fp32 = _trace_visits(model, tokens, record_ids, policy="cpu_fp32_reference")
        bf16 = _trace_visits(
            model,
            tokens,
            record_ids,
            policy="cpu_fp32_master_bf16_autocast",
        )

        with torch.no_grad():
            ordinary = model(tokens, memory_record_ids=record_ids)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    trace_matches = torch.equal(fp32[-1].logits, ordinary.logits)
    if not trace_matches:
        raise RuntimeError("C2 visit trace does not reproduce the current fp32 main graph")

    per_visit = tuple(
        C2VisitDrift(
            visit=index,
            hidden=_tensor_drift(reference.hidden, observed.hidden),
            scratch_lanes=_tensor_drift(reference.lanes, observed.lanes),
            logits=_tensor_drift(reference.logits, observed.logits),
            gradient=_tensor_drift(reference.gradient, observed.gradient),
            fp32_loss=float(reference.loss),
            bf16_compute_loss=float(observed.loss),
            relative_loss_drift=_relative_scalar_drift(reference.loss, observed.loss),
        )
        for index, (reference, observed) in enumerate(zip(fp32, bf16, strict=True), start=1)
    )
    config = model.config
    return C2PreflightReceipt(
        authority=C2_AUTHORITY,
        authority_sha256=C2_AUTHORITY_SHA256,
        authority_bytes=C2_AUTHORITY_BYTES,
        ratification_authority=C2_RATIFICATION_AUTHORITY,
        ratification_sha256=C2_RATIFICATION_SHA256,
        ratification_bytes=C2_RATIFICATION_BYTES,
        authority_byte_verified=True,
        measurement_status=(
            "cpu_current_integrated_composition_measured_"
            "full_weft1_and_threshold_gate_deferred"
        ),
        current_composition="ablation_lm_materialized_current_4_2_4_k8",
        current_integrated_modules=C2_CURRENT_INTEGRATED_MODULES,
        representative_missing_full_toy_integrations=(
            C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS
        ),
        full_weft1_toy_step_claim=False,
        carrier_accumulation_decision=(
            "deferred_integrated_learned_rotor_carrier_absent; "
            "two_lane_birkhoff_scratch_carrier_is_not_a_rotor"
        ),
        weight_state="deterministic_current_code_initialization_not_a_learned_checkpoint",
        root_seed=C2_ROOT_SEED,
        config_identity_sha256=config_identity_sha256,
        input_panel_sha256=input_panel_sha256,
        initial_model_state_sha256=initial_model_state_sha256,
        training_performed=False,
        checkpoint_used=False,
        reference_policy="cpu_fp32_parameters_and_compute",
        bf16_policy="cpu_fp32_master_parameters_with_bf16_autocast_compute",
        per_visit_readout_definition=(
            "current_read_only_long_term_memory_then_four_coda_blocks_then_tied_lm_head"
        ),
        gradient_definition=(
            "per_visit_shifted_cross_entropy_gradient_of_"
            "core_blocks.0.attention.q_proj.weight"
        ),
        visits=C2_RATIFIED_VISITS,
        block_split=(
            config.n_prelude_layers,
            config.n_core_blocks,
            config.n_coda_layers,
        ),
        d_model=config.d_model,
        scratch_shape=(config.scratch_lanes, config.scratch_width),
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        gradient_parameter=C2_GRADIENT_PARAMETER,
        trace_matches_main_forward_fp32=trace_matches,
        per_visit=per_visit,
        summary=_summarize_drift(per_visit),
        registered_relative_threshold=C2_REGISTERED_RELATIVE_THRESHOLD,
        threshold_source_ratified=True,
        threshold_metric_binding_status=(
            "underspecified_metric_population_denominator_gradient_and_visit_aggregation"
        ),
        threshold_applied=False,
        threshold_passed=None,
        required_metric_rulings=C2_REQUIRED_METRIC_RULINGS,
        deferred_gpu_cells=(
            DeferredC2Cell(
                cell="cuda_fp32_vs_bf16_forward_k8",
                status="deferred",
                reason="requires the separately metered deterministic GPU backend",
            ),
            DeferredC2Cell(
                cell="cuda_fp32_vs_bf16_backward_k8",
                status="deferred",
                reason="CPU autocast is not GPU backward evaluator identity",
            ),
            DeferredC2Cell(
                cell="cuda_same_seed_bit_replay",
                status="deferred",
                reason="belongs to the GPU-deterministic C2/C3 runtime cells",
            ),
        ),
        cpu_runtime=platform.platform(),
        torch_version=torch.__version__,
    )


def main() -> None:
    print(json.dumps(run_preflight_c2().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
