"""CPU-valid portion of the WEFT-1 PRE-FLIGHT C2 precision gate.

The governing C2 row asks for a K=8 bf16/full-toy comparison against fp32
masters.  This module deliberately measures only the graph that exists today.
The learned rotor carrier, per-band callosum, and loop sidecar are not integrated,
so this receipt cannot decide rotor-carrier accumulation or claim that the full
WEFT-1 toy chassis passed.

PF-2.2 binds vector-relative L2 per tensor and per visit, with terminal K=8 as
the decision value.  State/logits use a 1e-2 ceiling; lanes and gradients use a
5e-2 ceiling.  Gradients cover both the concatenated trainable-parameter vector
and the worst tensor in every parameter-owning module.  PF-2 expressly binds
those thresholds after seeing the original diagnostic values, which remains a
first-class disclosure in the receipt. PF-3.2 classifies an exact-zero reference
through the structural eligibility matrix: a valid ineligible zero is reported
and excluded, while an eligible zero or a precision mismatch fails the gate.
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
C2_PF2_AUTHORITY = "docs/STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md#2-PF-2.2"
C2_PF2_AUTHORITY_SHA256 = (
    "be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05"
)
C2_PF2_AUTHORITY_BYTES = 13_097
C2_PF3_AUTHORITY = "docs/STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md#2-PF-3.2"
C2_PF3_AUTHORITY_SHA256 = (
    "7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef"
)
C2_PF3_AUTHORITY_BYTES = 14_632
C2_RATIFIED_VISITS = 8
C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD = 1e-2
C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD = 5e-2
C2_ROOT_SEED = 20_260_902
C2_REENTRY_VISIT_ZERO_INELIGIBILITY_REASON = (
    "reentry_bridge_is_not_executed_on_recurrent_visit_zero"
)

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


class C2GateCatch(RuntimeError):
    """Raised when a caller attempts to promote a non-passing C2 receipt."""


@dataclass(frozen=True)
class TensorDrift:
    """PF-2.2 gate metric plus retained descriptive norm/cosine diagnostics."""

    reference_dtype: str
    bf16_compute_dtype: str
    reference_l2: float
    bf16_compute_l2: float
    relative_l2_error: float
    relative_norm_drift: float
    cosine_similarity: float | None


@dataclass(frozen=True)
class ModuleGradientWorstTensor:
    module_name: str
    parameter_name: str | None
    parameter_tensor_count: int
    defined_tensor_count: int
    excluded_zero_reference_parameter_names: tuple[str, ...]
    failed_zero_reference_parameter_names: tuple[str, ...]
    drift: TensorDrift | None
    complete: bool

    def __post_init__(self) -> None:
        classified = (
            self.defined_tensor_count
            + len(self.excluded_zero_reference_parameter_names)
            + len(self.failed_zero_reference_parameter_names)
        )
        if self.parameter_tensor_count < 1 or classified != self.parameter_tensor_count:
            raise ValueError("C2 module gradient tensor counts are inconsistent")
        if set(self.excluded_zero_reference_parameter_names).intersection(
            self.failed_zero_reference_parameter_names
        ):
            raise ValueError("C2 module zero-reference classes must be disjoint")
        if (self.drift is None) is not (self.defined_tensor_count == 0):
            raise ValueError("C2 module gradient drift presence is inconsistent")
        if (self.parameter_name is None) is not (self.drift is None):
            raise ValueError("C2 module worst parameter identity is inconsistent")
        if self.complete is not (not self.failed_zero_reference_parameter_names):
            raise ValueError("C2 module gradient completeness is inconsistent")


@dataclass(frozen=True)
class ZeroReferenceGradientCell:
    visit: int
    module_name: str
    parameter_name: str
    structurally_eligible: bool
    structural_reason: str
    disposition: str
    excluded_from_relative_error_population: bool
    fp32_autograd_connected: bool
    bf16_autograd_connected: bool
    fp32_l2: float
    bf16_compute_l2: float
    passed: bool

    def __post_init__(self) -> None:
        if self.visit < 1 or not self.module_name or not self.parameter_name:
            raise ValueError("C2 zero-reference cell identity is invalid")
        if self.fp32_l2 != 0.0:
            raise ValueError("C2 zero-reference cell requires an exact-zero fp32 norm")
        if not math.isfinite(self.bf16_compute_l2) or self.bf16_compute_l2 < 0.0:
            raise ValueError("C2 zero-reference bf16 norm must be finite and non-negative")
        expected_eligible, expected_reason = _gradient_structural_eligibility(
            self.parameter_name,
            visit=self.visit,
        )
        if self.structurally_eligible is not expected_eligible:
            raise ValueError("C2 zero-reference structural eligibility is inconsistent")
        if expected_eligible:
            expected_disposition = "fail (eligible zero reference)"
            expected_excluded = False
            expected_passed = False
        elif self.bf16_compute_l2 != 0.0:
            expected_reason = C2_REENTRY_VISIT_ZERO_INELIGIBILITY_REASON
            expected_disposition = "fail (ineligible precision mismatch)"
            expected_excluded = False
            expected_passed = False
        elif self.fp32_autograd_connected or self.bf16_autograd_connected:
            expected_reason = C2_REENTRY_VISIT_ZERO_INELIGIBILITY_REASON
            expected_disposition = "fail (ineligible autograd connectivity mismatch)"
            expected_excluded = False
            expected_passed = False
        else:
            expected_reason = C2_REENTRY_VISIT_ZERO_INELIGIBILITY_REASON
            expected_disposition = "ineligible (zero reference)"
            expected_excluded = True
            expected_passed = True
        if self.structural_reason != expected_reason:
            raise ValueError("C2 zero-reference structural reason is inconsistent")
        if (
            self.disposition != expected_disposition
            or self.excluded_from_relative_error_population is not expected_excluded
            or self.passed is not expected_passed
        ):
            raise ValueError("C2 zero-reference disposition is inconsistent")


@dataclass(frozen=True)
class GradientVisitDrift:
    full_parameter_vector: TensorDrift
    per_module_worst_tensors: tuple[ModuleGradientWorstTensor, ...]
    worst_module_name: str | None
    worst_parameter_name: str | None
    worst_tensor: TensorDrift | None
    zero_reference_cells: tuple[ZeroReferenceGradientCell, ...]
    complete: bool
    trainable_parameter_tensors: int
    trainable_parameter_elements: int

    def __post_init__(self) -> None:
        expected_complete = all(cell.passed for cell in self.zero_reference_cells)
        if self.complete is not expected_complete:
            raise ValueError("C2 visit gradient completeness is inconsistent")
        if self.trainable_parameter_tensors < 1 or self.trainable_parameter_elements < 1:
            raise ValueError("C2 visit gradient population must be non-empty")
        classified_tensors = sum(
            item.parameter_tensor_count for item in self.per_module_worst_tensors
        )
        if classified_tensors != self.trainable_parameter_tensors:
            raise ValueError("C2 visit gradient module population is inconsistent")
        defined = tuple(
            item for item in self.per_module_worst_tensors if item.drift is not None
        )
        expected_worst = (
            max(defined, key=_defined_module_worst_relative_l2) if defined else None
        )
        expected_name = None if expected_worst is None else expected_worst.module_name
        expected_parameter = (
            None if expected_worst is None else expected_worst.parameter_name
        )
        expected_drift = None if expected_worst is None else expected_worst.drift
        if (
            self.worst_module_name != expected_name
            or self.worst_parameter_name != expected_parameter
            or self.worst_tensor != expected_drift
        ):
            raise ValueError("C2 visit worst module gradient is inconsistent")


@dataclass(frozen=True)
class C2VisitDrift:
    visit: int
    hidden: TensorDrift
    scratch_lanes: TensorDrift
    logits: TensorDrift
    gradient: GradientVisitDrift
    fp32_loss: float
    bf16_compute_loss: float
    relative_loss_drift: float


@dataclass(frozen=True)
class DeferredC2Cell:
    cell: str
    status: str
    reason: str


@dataclass(frozen=True)
class ModuleGradientMaximum:
    module_name: str
    parameter_name: str | None
    max_relative_l2: float | None
    max_relative_l2_visit: int | None
    excluded_zero_reference_visits: tuple[int, ...]
    failed_zero_reference_visits: tuple[int, ...]
    complete: bool

    def __post_init__(self) -> None:
        if self.complete is not (not self.failed_zero_reference_visits):
            raise ValueError("C2 module maximum completeness is inconsistent")
        if (self.max_relative_l2 is None) is not (
            self.max_relative_l2_visit is None
        ):
            raise ValueError("C2 module maximum value and visit are inconsistent")
        if (self.max_relative_l2 is None) is not (self.parameter_name is None):
            raise ValueError("C2 module maximum parameter identity is inconsistent")


@dataclass(frozen=True)
class C2DriftSummary:
    max_hidden_relative_l2: float
    max_hidden_relative_l2_visit: int
    max_scratch_lane_relative_l2: float
    max_scratch_lane_relative_l2_visit: int
    max_logit_relative_l2: float
    max_logit_relative_l2_visit: int
    max_full_gradient_relative_l2: float
    max_full_gradient_relative_l2_visit: int
    max_worst_module_gradient_relative_l2: float | None
    max_worst_module_gradient_relative_l2_visit: int | None
    max_worst_module_gradient_module: str | None
    max_worst_module_gradient_parameter: str | None
    per_module_gradient_maxima: tuple[ModuleGradientMaximum, ...]
    zero_reference_cells: tuple[ZeroReferenceGradientCell, ...]
    zero_reference_failures: tuple[ZeroReferenceGradientCell, ...]
    gradient_maxima_complete: bool
    max_relative_loss_drift: float
    max_relative_loss_drift_visit: int


@dataclass(frozen=True)
class C2TerminalGateDecision:
    visit: int
    metric: str
    hidden_relative_l2: float
    scratch_lane_relative_l2: float
    logit_relative_l2: float
    full_gradient_relative_l2: float
    worst_module_gradient_relative_l2: float | None
    worst_module_gradient_module: str | None
    worst_module_gradient_parameter: str | None
    state_threshold: float
    logit_threshold: float
    lane_threshold: float
    gradient_threshold: float
    hidden_passed: bool
    scratch_lanes_passed: bool
    logits_passed: bool
    full_gradient_passed: bool
    gradient_population_complete: bool
    every_module_worst_gradient_passed: bool
    passed: bool


@dataclass(frozen=True)
class C2PreflightReceipt:
    authority: str
    authority_sha256: str
    authority_bytes: int
    ratification_authority: str
    ratification_sha256: str
    ratification_bytes: int
    pf2_authority: str
    pf2_authority_sha256: str
    pf2_authority_bytes: int
    pf3_authority: str
    pf3_authority_sha256: str
    pf3_authority_bytes: int
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
    gradient_population: str
    relative_l2_denominator_policy: str
    trace_matches_main_forward_fp32: bool
    per_visit: tuple[C2VisitDrift, ...]
    summary: C2DriftSummary
    terminal_gate: C2TerminalGateDecision
    state_logit_relative_l2_threshold: float
    lane_gradient_relative_l2_threshold: float
    threshold_source_authority: str
    thresholds_bound_after_data: bool
    thresholds_preregistered: bool
    threshold_binding_disclosure: str
    threshold_metric_binding_status: str
    threshold_applied: bool
    threshold_passed: bool
    catch_number: int | None
    catch_reason: str | None
    deferred_gpu_cells: tuple[DeferredC2Cell, ...]
    cpu_runtime: str
    torch_version: str
    a100_hours: float = 0.0

    def __post_init__(self) -> None:
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        if not self.authority_byte_verified:
            raise ValueError("C2 authority_byte_verified must be true")
        if (
            self.authority != C2_AUTHORITY
            or self.authority_sha256 != C2_AUTHORITY_SHA256
            or self.authority_bytes != C2_AUTHORITY_BYTES
            or self.ratification_authority != C2_RATIFICATION_AUTHORITY
            or self.ratification_sha256 != C2_RATIFICATION_SHA256
            or self.ratification_bytes != C2_RATIFICATION_BYTES
            or self.pf2_authority != C2_PF2_AUTHORITY
            or self.pf2_authority_sha256 != C2_PF2_AUTHORITY_SHA256
            or self.pf2_authority_bytes != C2_PF2_AUTHORITY_BYTES
            or self.pf3_authority_sha256 != C2_PF3_AUTHORITY_SHA256
            or self.pf3_authority_bytes != C2_PF3_AUTHORITY_BYTES
            or self.pf3_authority != C2_PF3_AUTHORITY
        ):
            raise ValueError("C2 authority metadata is inconsistent")
        expected_deferred_gpu_cells = (
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
        )
        provenance_and_scope_bound = (
            self.current_composition == "ablation_lm_materialized_current_4_2_4_k8"
            and self.current_integrated_modules == C2_CURRENT_INTEGRATED_MODULES
            and self.representative_missing_full_toy_integrations
            == C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS
            and not self.full_weft1_toy_step_claim
            and self.carrier_accumulation_decision
            == (
                "deferred_integrated_learned_rotor_carrier_absent; "
                "two_lane_birkhoff_scratch_carrier_is_not_a_rotor"
            )
            and self.weight_state
            == "deterministic_current_code_initialization_not_a_learned_checkpoint"
            and self.root_seed == C2_ROOT_SEED
            and self.config_identity_sha256
            == "57ef01bbaa4aa89ea18b098cbb51ece021fc0849f9b958435f7f534f2f6d8e28"
            and self.input_panel_sha256
            == "1bce335742f00f14c7e9abd88396776dc4d473fb955f60a923868618bf1d15f6"
            and self.initial_model_state_sha256
            == "8420c11aa7746c2191206cb061bd9bcbecf217fe264e0fc24cdb1761ee2dc2fb"
            and not self.training_performed
            and not self.checkpoint_used
            and self.reference_policy == "cpu_fp32_parameters_and_compute"
            and self.bf16_policy
            == "cpu_fp32_master_parameters_with_bf16_autocast_compute"
            and self.per_visit_readout_definition
            == "current_read_only_long_term_memory_then_four_coda_blocks_then_tied_lm_head"
            and self.gradient_definition
            == (
                "per_visit_shifted_cross_entropy_gradients_of_all_trainable_"
                "named_parameters; concatenated_full_vector_plus_each_leaf_"
                "parameter_module_worst_tensor; PF-3.2_structurally_ineligible_"
                "exact_zero_parameters_are_reported_and_excluded_from_relative_error"
            )
            and self.block_split == (4, 2, 4)
            and self.d_model == 64
            and self.scratch_shape == (2, 8)
            and self.model_parameters == 484_795
            and self.gradient_population
            == "all_requires_grad_named_parameters_in_model_named_parameters_order"
            and self.relative_l2_denominator_policy
            == (
                "relative_error_only_where_fp32_l2_is_nonzero; PF-3.2_valid_"
                "structurally_ineligible_exact_zero_cells_are_reported_and_excluded; "
                "eligible_zero_or_precision_mismatch_fails"
            )
            and self.trace_matches_main_forward_fp32
            and self.deferred_gpu_cells == expected_deferred_gpu_cells
            and self.cpu_runtime == platform.platform()
            and self.torch_version == torch.__version__
            and self.a100_hours == 0.0
        )
        if not provenance_and_scope_bound:
            raise ValueError("C2 topology, provenance, or execution scope is inconsistent")
        if (
            self.state_logit_relative_l2_threshold
            != C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD
            or self.lane_gradient_relative_l2_threshold
            != C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD
            or self.threshold_source_authority != C2_PF2_AUTHORITY
            or not self.threshold_applied
            or not self.thresholds_bound_after_data
            or self.thresholds_preregistered
            or self.threshold_binding_disclosure
            != (
                "PF-2.2_bound_thresholds_after_observing_the_original_C2_"
                "diagnostic_values; future_runs_inherit_them"
            )
        ):
            raise ValueError("C2 PF-2.2 threshold binding is inconsistent")
        if self.threshold_metric_binding_status != (
            "bound_by_PF-2.2_vector_relative_l2_per_tensor_per_visit_"
            "terminal_K8_decision_and_PF-3.2_zero_reference_eligibility"
        ):
            raise ValueError("C2 PF-3.2 metric binding status is inconsistent")
        if (
            self.visits != C2_RATIFIED_VISITS
            or len(self.per_visit) != C2_RATIFIED_VISITS
            or tuple(item.visit for item in self.per_visit)
            != tuple(range(1, C2_RATIFIED_VISITS + 1))
        ):
            raise ValueError("C2 visit population is inconsistent")

        expected_summary = _summarize_drift(self.per_visit)
        if self.summary != expected_summary:
            raise ValueError("C2 summary is inconsistent with per-visit measurements")
        expected_terminal = _terminal_gate_decision(self.per_visit)
        if self.terminal_gate != expected_terminal:
            raise ValueError("C2 terminal gate is inconsistent with per-visit measurements")

        expected_passed = expected_terminal.passed and expected_summary.gradient_maxima_complete
        expected_catch = None
        expected_reason = None
        if self.threshold_passed is not expected_passed:
            raise ValueError("C2 threshold_passed is inconsistent with the complete gate")
        if self.catch_number != expected_catch or self.catch_reason != expected_reason:
            raise ValueError("C2 catch disposition is inconsistent with the complete gate")
        expected_status = _measurement_status(
            terminal_gate=expected_terminal,
            complete_gate_passed=expected_passed,
        )
        if self.measurement_status != expected_status:
            raise ValueError("C2 measurement_status is inconsistent with the complete gate")

    def require_passed(self) -> None:
        self._validate_invariants()
        if self.threshold_passed:
            return
        if self.catch_number is not None:
            raise C2GateCatch(f"CATCH #{self.catch_number}: {self.catch_reason}")
        raise C2GateCatch("C2 PF-2.2 complete gate did not pass")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _VisitTrace:
    hidden: torch.Tensor
    lanes: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor
    gradients: tuple["_GradientTraceTensor", ...]


@dataclass(frozen=True)
class _GradientTraceTensor:
    parameter_name: str
    value: torch.Tensor
    autograd_connected: bool


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
        (
            root / "docs" / "STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md",
            C2_PF2_AUTHORITY_BYTES,
            C2_PF2_AUTHORITY_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md",
            C2_PF3_AUTHORITY_BYTES,
            C2_PF3_AUTHORITY_SHA256,
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


def _gradient_parameters(
    model: AblationLM,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("C2 gradient population has no trainable parameters")
    return parameters


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
    gradient_parameters = _gradient_parameters(model)
    parameter_values = tuple(parameter for _name, parameter in gradient_parameters)
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

        gradients_by_visit: list[tuple[_GradientTraceTensor, ...]] = []
        for visit_index, loss in enumerate(losses):
            gradients = torch.autograd.grad(
                loss,
                parameter_values,
                retain_graph=visit_index + 1 < C2_RATIFIED_VISITS,
                allow_unused=True,
            )
            gradients_by_visit.append(
                tuple(
                    _GradientTraceTensor(
                        parameter_name=name,
                        value=(
                            torch.zeros_like(_parameter)
                            if gradient is None
                            else gradient.detach().clone()
                        ),
                        autograd_connected=gradient is not None,
                    )
                    for (name, _parameter), gradient in zip(
                        gradient_parameters,
                        gradients,
                        strict=True,
                    )
                )
            )

    return tuple(
        _VisitTrace(
            hidden=hidden_state.detach().clone(),
            lanes=lane_state.detach().clone(),
            logits=logits.detach().clone(),
            loss=loss.detach().clone(),
            gradients=gradients,
        )
        for hidden_state, lane_state, logits, loss, gradients in zip(
            states,
            lane_states,
            logits_by_visit,
            losses,
            gradients_by_visit,
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
    if reference_l2.item() == 0.0:
        raise ValueError(
            "C2 vector-relative L2 requires a nonzero fp32 reference tensor; "
            "PF-2.2 binds no zero-denominator fallback"
        )
    difference = torch.linalg.vector_norm(observed64 - reference64)
    cosine = (
        None
        if observed_l2.item() == 0.0
        else float(torch.dot(reference64, observed64) / (reference_l2 * observed_l2))
    )
    return TensorDrift(
        reference_dtype=str(reference.dtype).removeprefix("torch."),
        bf16_compute_dtype=str(observed.dtype).removeprefix("torch."),
        reference_l2=float(reference_l2),
        bf16_compute_l2=float(observed_l2),
        relative_l2_error=float(difference / reference_l2),
        relative_norm_drift=float((observed_l2 - reference_l2).abs() / reference_l2),
        cosine_similarity=cosine,
    )


def _parameter_module_name(parameter_name: str) -> str:
    module_name, separator, _leaf_name = parameter_name.rpartition(".")
    return module_name if separator else "<root>"


def _defined_module_worst_relative_l2(item: ModuleGradientWorstTensor) -> float:
    if item.drift is None:
        raise ValueError("C2 internal error: expected a defined module gradient")
    return item.drift.relative_l2_error


def _defined_module_maximum_relative_l2(item: ModuleGradientMaximum) -> float:
    if item.max_relative_l2 is None:
        raise ValueError("C2 internal error: expected a defined module maximum")
    return item.max_relative_l2


def _gradient_structural_eligibility(
    parameter_name: str,
    *,
    visit: int,
) -> tuple[bool, str]:
    """Return the PF-1.5/PF-3.2 eligibility of one gradient tensor.

    Visits are one-indexed in the C2 receipt. The re-entry bridge is skipped by
    construction at recurrent visit zero, which is receipt visit one. Every
    materialized parameter outside that exact module is eligible at every C2
    visit; future module exceptions must be added to the registered matrix
    rather than inferred from an observed zero.
    """

    if visit < 1:
        raise ValueError("C2 structural eligibility requires a positive visit")
    if parameter_name == "reentry_bridge" or parameter_name.startswith(
        "reentry_bridge."
    ):
        if visit == 1:
            return False, C2_REENTRY_VISIT_ZERO_INELIGIBILITY_REASON
    return True, "parameter_is_structurally_eligible_at_this_visit"


def _gradient_drift(
    reference: tuple[_GradientTraceTensor, ...],
    observed: tuple[_GradientTraceTensor, ...],
    *,
    visit: int,
) -> GradientVisitDrift:
    reference_names = tuple(item.parameter_name for item in reference)
    observed_names = tuple(item.parameter_name for item in observed)
    if reference_names != observed_names:
        raise ValueError("C2 fp32 and bf16 gradient populations differ")
    if not reference:
        raise ValueError("C2 gradient population must not be empty")

    per_tensor: list[tuple[str, str, TensorDrift | None, ZeroReferenceGradientCell | None]] = []
    zero_reference_cells: list[ZeroReferenceGradientCell] = []
    for reference_item, observed_item in zip(reference, observed, strict=True):
        module_name = _parameter_module_name(reference_item.parameter_name)
        fp32_l2 = float(
            torch.linalg.vector_norm(reference_item.value.detach().double().reshape(-1))
        )
        bf16_l2 = float(
            torch.linalg.vector_norm(observed_item.value.detach().double().reshape(-1))
        )
        if fp32_l2 == 0.0:
            eligible, structural_reason = _gradient_structural_eligibility(
                reference_item.parameter_name,
                visit=visit,
            )
            if eligible:
                disposition = "fail (eligible zero reference)"
                excluded = False
                passed = False
            elif bf16_l2 != 0.0:
                disposition = "fail (ineligible precision mismatch)"
                excluded = False
                passed = False
            elif reference_item.autograd_connected or observed_item.autograd_connected:
                disposition = "fail (ineligible autograd connectivity mismatch)"
                excluded = False
                passed = False
            else:
                disposition = "ineligible (zero reference)"
                excluded = True
                passed = True
            zero_cell = ZeroReferenceGradientCell(
                visit=visit,
                module_name=module_name,
                parameter_name=reference_item.parameter_name,
                structurally_eligible=eligible,
                structural_reason=structural_reason,
                disposition=disposition,
                excluded_from_relative_error_population=excluded,
                fp32_autograd_connected=reference_item.autograd_connected,
                bf16_autograd_connected=observed_item.autograd_connected,
                fp32_l2=fp32_l2,
                bf16_compute_l2=bf16_l2,
                passed=passed,
            )
            per_tensor.append(
                (module_name, reference_item.parameter_name, None, zero_cell)
            )
            zero_reference_cells.append(zero_cell)
        else:
            per_tensor.append(
                (
                    module_name,
                    reference_item.parameter_name,
                    _tensor_drift(reference_item.value, observed_item.value),
                    None,
                )
            )
    modules = tuple(
        dict.fromkeys(module_name for module_name, _name, _drift, _zero in per_tensor)
    )
    per_module_worst: list[ModuleGradientWorstTensor] = []
    for module_name in modules:
        module_tensors = tuple(
            (parameter_name, drift, zero_cell)
            for tensor_module, parameter_name, drift, zero_cell in per_tensor
            if tensor_module == module_name
        )
        defined_tensors = tuple(
            (parameter_name, drift)
            for parameter_name, drift, _zero_cell in module_tensors
            if drift is not None
        )
        excluded_names = tuple(
            parameter_name
            for parameter_name, _drift, zero_cell in module_tensors
            if zero_cell is not None
            and zero_cell.excluded_from_relative_error_population
        )
        failed_names = tuple(
            parameter_name
            for parameter_name, _drift, zero_cell in module_tensors
            if zero_cell is not None and not zero_cell.passed
        )
        if defined_tensors:
            worst_parameter, worst_drift = max(
                defined_tensors,
                key=lambda item: item[1].relative_l2_error,
            )
        else:
            worst_parameter, worst_drift = None, None
        per_module_worst.append(
            ModuleGradientWorstTensor(
                module_name=module_name,
                parameter_name=worst_parameter,
                parameter_tensor_count=len(module_tensors),
                defined_tensor_count=len(defined_tensors),
                excluded_zero_reference_parameter_names=excluded_names,
                failed_zero_reference_parameter_names=failed_names,
                drift=worst_drift,
                complete=not failed_names,
            )
        )

    defined_module_worst = tuple(
        item for item in per_module_worst if item.drift is not None
    )
    worst_module = (
        max(
            defined_module_worst,
            key=_defined_module_worst_relative_l2,
        )
        if defined_module_worst
        else None
    )
    reference_vector = torch.cat(
        tuple(item.value.detach().reshape(-1) for item in reference)
    )
    observed_vector = torch.cat(
        tuple(item.value.detach().reshape(-1) for item in observed)
    )
    return GradientVisitDrift(
        full_parameter_vector=_tensor_drift(reference_vector, observed_vector),
        per_module_worst_tensors=tuple(per_module_worst),
        worst_module_name=None if worst_module is None else worst_module.module_name,
        worst_parameter_name=(
            None if worst_module is None else worst_module.parameter_name
        ),
        worst_tensor=None if worst_module is None else worst_module.drift,
        zero_reference_cells=tuple(zero_reference_cells),
        complete=all(cell.passed for cell in zero_reference_cells),
        trainable_parameter_tensors=len(reference),
        trainable_parameter_elements=reference_vector.numel(),
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
    loss, loss_visit = maximum("relative_loss_drift")
    full_gradient_values = tuple(
        visit.gradient.full_parameter_vector.relative_l2_error
        for visit in per_visit
    )
    full_gradient_index = max(
        range(len(full_gradient_values)),
        key=full_gradient_values.__getitem__,
    )
    full_gradient = full_gradient_values[full_gradient_index]
    full_gradient_visit = per_visit[full_gradient_index].visit

    module_names = tuple(
        item.module_name
        for item in per_visit[0].gradient.per_module_worst_tensors
    )
    if any(
        tuple(item.module_name for item in visit.gradient.per_module_worst_tensors)
        != module_names
        for visit in per_visit[1:]
    ):
        raise ValueError("C2 per-module gradient population differs across visits")
    module_maxima: list[ModuleGradientMaximum] = []
    for module_index, module_name in enumerate(module_names):
        candidates = tuple(
            visit.gradient.per_module_worst_tensors[module_index]
            for visit in per_visit
        )
        defined_candidate_indices = tuple(
            index for index, item in enumerate(candidates) if item.drift is not None
        )
        excluded_visits = tuple(
            per_visit[index].visit
            for index, item in enumerate(candidates)
            if item.excluded_zero_reference_parameter_names
        )
        failed_visits = tuple(
            per_visit[index].visit
            for index, item in enumerate(candidates)
            if item.failed_zero_reference_parameter_names
        )
        maximum_index = (
            max(
                defined_candidate_indices,
                key=lambda index: _defined_module_worst_relative_l2(
                    candidates[index]
                ),
            )
            if defined_candidate_indices
            else None
        )
        maximum_item = None if maximum_index is None else candidates[maximum_index]
        module_maxima.append(
            ModuleGradientMaximum(
                module_name=module_name,
                parameter_name=(
                    None if maximum_item is None else maximum_item.parameter_name
                ),
                max_relative_l2=(
                    None
                    if maximum_item is None or maximum_item.drift is None
                    else maximum_item.drift.relative_l2_error
                ),
                max_relative_l2_visit=(
                    None if maximum_index is None else per_visit[maximum_index].visit
                ),
                excluded_zero_reference_visits=excluded_visits,
                failed_zero_reference_visits=failed_visits,
                complete=not failed_visits,
            )
        )
    defined_module_maxima = tuple(
        item for item in module_maxima if item.max_relative_l2 is not None
    )
    worst_module_maximum = (
        max(
            defined_module_maxima,
            key=_defined_module_maximum_relative_l2,
        )
        if defined_module_maxima
        else None
    )
    zero_reference_cells = tuple(
        cell
        for visit in per_visit
        for cell in visit.gradient.zero_reference_cells
    )
    zero_reference_failures = tuple(
        cell for cell in zero_reference_cells if not cell.passed
    )
    return C2DriftSummary(
        max_hidden_relative_l2=hidden,
        max_hidden_relative_l2_visit=hidden_visit,
        max_scratch_lane_relative_l2=lanes,
        max_scratch_lane_relative_l2_visit=lanes_visit,
        max_logit_relative_l2=logits,
        max_logit_relative_l2_visit=logits_visit,
        max_full_gradient_relative_l2=full_gradient,
        max_full_gradient_relative_l2_visit=full_gradient_visit,
        max_worst_module_gradient_relative_l2=(
            None if worst_module_maximum is None else worst_module_maximum.max_relative_l2
        ),
        max_worst_module_gradient_relative_l2_visit=(
            None
            if worst_module_maximum is None
            else worst_module_maximum.max_relative_l2_visit
        ),
        max_worst_module_gradient_module=(
            None if worst_module_maximum is None else worst_module_maximum.module_name
        ),
        max_worst_module_gradient_parameter=(
            None
            if worst_module_maximum is None
            else worst_module_maximum.parameter_name
        ),
        per_module_gradient_maxima=tuple(module_maxima),
        zero_reference_cells=zero_reference_cells,
        zero_reference_failures=zero_reference_failures,
        gradient_maxima_complete=not zero_reference_failures,
        max_relative_loss_drift=loss,
        max_relative_loss_drift_visit=loss_visit,
    )


def _terminal_gate_decision(
    per_visit: tuple[C2VisitDrift, ...],
) -> C2TerminalGateDecision:
    if len(per_visit) != C2_RATIFIED_VISITS or per_visit[-1].visit != C2_RATIFIED_VISITS:
        raise ValueError("C2 terminal decision requires the complete K=8 visit trace")
    terminal = per_visit[-1]
    hidden = terminal.hidden.relative_l2_error
    lanes = terminal.scratch_lanes.relative_l2_error
    logits = terminal.logits.relative_l2_error
    full_gradient = terminal.gradient.full_parameter_vector.relative_l2_error
    worst_module_gradient = (
        None
        if terminal.gradient.worst_tensor is None
        else terminal.gradient.worst_tensor.relative_l2_error
    )
    hidden_passed = hidden <= C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD
    lanes_passed = lanes <= C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD
    logits_passed = logits <= C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD
    full_gradient_passed = full_gradient <= C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD
    module_gradients_passed = terminal.gradient.complete and all(
        item.complete
        and (
            item.drift is None
            or item.drift.relative_l2_error
            <= C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD
        )
        for item in terminal.gradient.per_module_worst_tensors
    )
    return C2TerminalGateDecision(
        visit=terminal.visit,
        metric="vector_relative_l2_per_tensor",
        hidden_relative_l2=hidden,
        scratch_lane_relative_l2=lanes,
        logit_relative_l2=logits,
        full_gradient_relative_l2=full_gradient,
        worst_module_gradient_relative_l2=worst_module_gradient,
        worst_module_gradient_module=terminal.gradient.worst_module_name,
        worst_module_gradient_parameter=terminal.gradient.worst_parameter_name,
        state_threshold=C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD,
        logit_threshold=C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD,
        lane_threshold=C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD,
        gradient_threshold=C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD,
        hidden_passed=hidden_passed,
        scratch_lanes_passed=lanes_passed,
        logits_passed=logits_passed,
        full_gradient_passed=full_gradient_passed,
        gradient_population_complete=terminal.gradient.complete,
        every_module_worst_gradient_passed=module_gradients_passed,
        passed=(
            hidden_passed
            and lanes_passed
            and logits_passed
            and full_gradient_passed
            and module_gradients_passed
        ),
    )


def _measurement_status(
    *,
    terminal_gate: C2TerminalGateDecision,
    complete_gate_passed: bool,
) -> str:
    if complete_gate_passed:
        prefix = "cpu_current_integrated_composition_pf3_2_complete_gate_passed_"
    else:
        prefix = "cpu_current_integrated_composition_pf3_2_terminal_gate_failed_"
    terminal = "terminal_k8_passed_" if terminal_gate.passed else "terminal_k8_failed_"
    return prefix + terminal + "full_weft1_and_carrier_deferred"


def run_preflight_c2() -> C2PreflightReceipt:
    """Apply PF-2.2/PF-3.2 to the pinned CPU composition."""

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
            gradient=_gradient_drift(
                reference.gradients,
                observed.gradients,
                visit=index,
            ),
            fp32_loss=float(reference.loss),
            bf16_compute_loss=float(observed.loss),
            relative_loss_drift=_relative_scalar_drift(reference.loss, observed.loss),
        )
        for index, (reference, observed) in enumerate(zip(fp32, bf16, strict=True), start=1)
    )
    config = model.config
    summary = _summarize_drift(per_visit)
    terminal_gate = _terminal_gate_decision(per_visit)
    complete_gate_passed = terminal_gate.passed and summary.gradient_maxima_complete
    catch_number = None
    catch_reason = None
    return C2PreflightReceipt(
        authority=C2_AUTHORITY,
        authority_sha256=C2_AUTHORITY_SHA256,
        authority_bytes=C2_AUTHORITY_BYTES,
        ratification_authority=C2_RATIFICATION_AUTHORITY,
        ratification_sha256=C2_RATIFICATION_SHA256,
        ratification_bytes=C2_RATIFICATION_BYTES,
        pf2_authority=C2_PF2_AUTHORITY,
        pf2_authority_sha256=C2_PF2_AUTHORITY_SHA256,
        pf2_authority_bytes=C2_PF2_AUTHORITY_BYTES,
        pf3_authority=C2_PF3_AUTHORITY,
        pf3_authority_sha256=C2_PF3_AUTHORITY_SHA256,
        pf3_authority_bytes=C2_PF3_AUTHORITY_BYTES,
        authority_byte_verified=True,
        measurement_status=_measurement_status(
            terminal_gate=terminal_gate,
            complete_gate_passed=complete_gate_passed,
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
            "per_visit_shifted_cross_entropy_gradients_of_all_trainable_"
            "named_parameters; concatenated_full_vector_plus_each_leaf_"
            "parameter_module_worst_tensor; PF-3.2_structurally_ineligible_"
            "exact_zero_parameters_are_reported_and_excluded_from_relative_error"
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
        gradient_population=(
            "all_requires_grad_named_parameters_in_model_named_parameters_order"
        ),
        relative_l2_denominator_policy=(
            "relative_error_only_where_fp32_l2_is_nonzero; PF-3.2_valid_"
            "structurally_ineligible_exact_zero_cells_are_reported_and_excluded; "
            "eligible_zero_or_precision_mismatch_fails"
        ),
        trace_matches_main_forward_fp32=trace_matches,
        per_visit=per_visit,
        summary=summary,
        terminal_gate=terminal_gate,
        state_logit_relative_l2_threshold=(
            C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD
        ),
        lane_gradient_relative_l2_threshold=(
            C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD
        ),
        threshold_source_authority=C2_PF2_AUTHORITY,
        thresholds_bound_after_data=True,
        thresholds_preregistered=False,
        threshold_binding_disclosure=(
            "PF-2.2_bound_thresholds_after_observing_the_original_C2_"
            "diagnostic_values; future_runs_inherit_them"
        ),
        threshold_metric_binding_status=(
            "bound_by_PF-2.2_vector_relative_l2_per_tensor_per_visit_"
            "terminal_K8_decision_and_PF-3.2_zero_reference_eligibility"
        ),
        threshold_applied=True,
        threshold_passed=complete_gate_passed,
        catch_number=catch_number,
        catch_reason=catch_reason,
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
