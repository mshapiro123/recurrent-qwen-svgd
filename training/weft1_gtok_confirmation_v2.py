"""Physical equal-FLOP confirmation and vocabulary-freeze mint for G-TOK v2.

This module is intentionally downstream of a factory-validated base matrix.  It
does four things, in order:

1. price the 20-percent guard on target rung B for every vocabulary arm;
2. select the registered top-two pair and prove that the exact common FLOP
   budget is reachable at an optimizer-step boundary for all four rows;
3. run a fresh-model, checkpoint-free confirmation campaign whose cumulative
   meter extends the base campaign; and
4. mint ``V`` only from ``validate_compute_confirmation_v2``'s
   ``GREEN_NO_REVERSAL`` result.

The production entry point has no executor-injection surface.  Synthetic
executors terminate in :class:`DryRunComputeConfirmationV2`, which deliberately
has neither a validated confirmation nor a vocabulary-freeze field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

import torch
from tokenizers import Tokenizer

from training.weft1_gtok_campaign_v2 import (
    BaseCampaignResultV2,
    CampaignStopArtifactV2,
    GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
    GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
    GTOK_GOVERNED_TRAINING_SEEDS_V2,
    GTOK_MICROBATCH_SEQUENCES_V2,
    GTokCampaignV2Error,
    TokenizerExecutionArmV2,
    _event_ledger_sha256,
    _exclusive_write,
    _execute_with_lifecycle_v2,
    _load_persisted_attempts_v2,
    _next_physical_attempt_id_v2,
    _persist_attempt,
    _projection_from_measurement_v2,
    _write_stop,
    build_preflight_projection_v2,
    recover_orphaned_lifecycle_attempts_v2,
    validate_lifecycle_ledger_v2,
    validate_sqlite_event_ledger_v2,
)
from training.weft1_gtok_code_closure_v2 import (
    GTokCodeClosureReceiptV2,
    validate_gtok_code_closure_v2,
)
from training.weft1_gtok_contract import (
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    canonical_json_bytes,
)
from training.weft1_gtok_training_v2 import (
    AnalyticUnsupportedFlopRowV2,
    CalibrationMeasurementV2,
    CompleteFlopLedgerV2,
    GTokCampaignTripwireV2,
    GTokRunWatchdogV2,
    GTokTrainingV2Error,
    PackedBatchV2,
    PhysicalShapeFlopReceiptV2,
    ProfilerOperatorFlopRowV2,
    TrainingDocumentV2,
    TrainingPlanV2,
    V4CorpusSourceV2,
    _PhysicalFlopAccountantV2,
    _elapsed_microseconds,
    _execute_optimizer_step_v2,
    _update_training_plan_digest_v2,
    build_flat_a1_adamw_v2,
    build_gtok_proxy_model_v2,
    calibrate_arm_v2,
    evaluate_heldout_v2,
    iter_packed_global_batches_v2,
    measure_output_surface_performance_v2,
    require_production_a100_v2,
)
from training.weft1_gtok_v2_contract import (
    ArmCalibrationProjectionV2,
    CampaignComputeReceiptV2,
    ComputeAttemptReceiptV2,
    ComputeConfirmationRunV2,
    GTOK_PER_RUN_WATCHDOG_MULTIPLIER,
    GTOK_TRIPWIRE_A100_MICROSECONDS,
    GTokSelectionReceiptV2,
    GTokV2Stop,
    PreflightProjectionReceiptV2,
    RuntimeTripwireSnapshotV2,
    VocabExtBasisV2,
    ValidatedComputeConfirmationV2,
    ValidatedGTokMatrixV2,
    VocabularyAdmissibilityReceiptV2,
    VocabularyFreezeArtifactV2,
    gtok_v2_bound_sha256,
    mint_vocabulary_freeze_v2,
    select_vocabulary_v2,
    validate_compute_confirmation_v2,
    validate_selection_receipt_v2,
)
from training.weft1_strict_io import (
    StrictJsonError,
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


RUNG_B_ANCHOR_VOCAB_SIZE_V2 = 32_768
RUNG_B_ANCHOR_PARAMETER_COUNT_V2 = 305_800_000
RUNG_B_MODEL_WIDTH_V2 = 1_024
CONFIRMATION_HELDOUT_EVALUATIONS_V2 = 3
CONFIRMATION_BINDING_V2 = {
    "admissibility": (
        "V*1024/(305800000+(V-32768)*1024)<=1/5"
    ),
    "base_flop_source": "completed_physical_base_lifecycle_flop_ledgers",
    "budget": "minimum_base_measured_flops_across_top_two_by_two_seeds",
    "budget_boundary": "exact_optimizer_step_prefix_or_stop_before_calibration",
    "calibration": (
        "fresh_model_20_warmup_plus_80_measured_plus_one_H_plus_output_surface"
    ),
    "confirmation_evaluations": "three_H_passes_at_nearest_distinct_thirds",
    "gpu_provenance": "physical_uuid_per_attempt_not_runtime_identity",
    "model_state": "fresh_per_attempt_no_checkpoint_or_optimizer_resume",
    "offline_network": (
        "stable_policy_identity_plus_physical_parent_launch_receipt_per_attempt"
    ),
    "runtime": "same_training_runtime_and_code_closure_as_base_matrix",
    "selection": "A2_R7_target_rung_B_then_top_two_by_agreed_seed_order",
}
CONFIRMATION_BINDING_SHA256_V2 = hashlib.sha256(
    canonical_json_bytes(CONFIRMATION_BINDING_V2)
).hexdigest()
_HEX = frozenset("0123456789abcdef")


class GTokConfirmationV2Error(RuntimeError):
    """Physical confirmation evidence is absent, inconsistent, or unsafe."""


@dataclass(frozen=True)
class BaseStepFlopV2:
    optimizer_step: int
    batch_rows: int
    sequence_length: int
    optimizer_phase: str
    measured_flops: int

    def __post_init__(self) -> None:
        for name in (
            "optimizer_step",
            "batch_rows",
            "sequence_length",
            "measured_flops",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        expected = "initial" if self.optimizer_step == 1 else "steady"
        if self.optimizer_phase != expected:
            raise ValueError("base step optimizer phase is inconsistent")


@dataclass(frozen=True)
class BaseRunFlopEvidenceV2:
    vocab_size: int
    seed: int
    base_run_receipt_sha256: str
    base_compute_attempt_id: str
    flop_ledger_sha256: str
    steps: tuple[BaseStepFlopV2, ...]
    measured_flops: int

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("base FLOP evidence uses an unregistered vocabulary")
        if type(self.seed) is not int:
            raise TypeError("base FLOP evidence seed must be exact")
        for name in ("base_run_receipt_sha256", "flop_ledger_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if not isinstance(self.base_compute_attempt_id, str) or not self.base_compute_attempt_id:
            raise ValueError("base compute attempt identity must be nonempty")
        if not self.steps or tuple(row.optimizer_step for row in self.steps) != tuple(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("base FLOP steps must be complete and contiguous")
        if self.measured_flops != sum(row.measured_flops for row in self.steps):
            raise ValueError("base run FLOPs differ from its reconstructed steps")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_base_run_flop_evidence", self)


@dataclass(frozen=True)
class ConfirmationReachabilityRowV2:
    vocab_size: int
    seed: int
    base_flop_evidence_sha256: str
    common_flop_budget: int
    reached_optimizer_steps: int | None
    reached_compute_token_slots: int | None
    nearest_lower_flops: int
    nearest_upper_flops: int | None
    exact_reachable: bool

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS or type(self.seed) is not int:
            raise ValueError("reachability row key is invalid")
        if len(self.base_flop_evidence_sha256) != 64:
            raise ValueError("reachability row lacks base FLOP evidence")
        for name in ("common_flop_budget", "nearest_lower_flops"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if self.nearest_upper_flops is not None and (
            type(self.nearest_upper_flops) is not int
            or self.nearest_upper_flops < self.common_flop_budget
        ):
            raise ValueError("nearest upper FLOPs are invalid")
        if self.exact_reachable:
            if (
                type(self.reached_optimizer_steps) is not int
                or self.reached_optimizer_steps < 100
                or type(self.reached_compute_token_slots) is not int
                or self.reached_compute_token_slots < 1
                or self.nearest_lower_flops != self.common_flop_budget
            ):
                raise ValueError(
                    "reachable confirmation must bind >=100 exact steps and token slots"
                )
        elif self.reached_optimizer_steps is not None or self.reached_compute_token_slots is not None:
            raise ValueError("unreachable confirmation may not claim a physical prefix")


@dataclass(frozen=True)
class ConfirmationReachabilityReceiptV2:
    matrix_receipt_sha256: str
    selection_receipt_sha256: str
    pair: tuple[int, int]
    seeds: tuple[int, int]
    common_flop_budget: int
    rows: tuple[ConfirmationReachabilityRowV2, ...]
    all_exact_reachable: bool
    binding_sha256: str = CONFIRMATION_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        for value in (self.matrix_receipt_sha256, self.selection_receipt_sha256):
            if len(value) != 64:
                raise ValueError("reachability receipt join must be SHA-256")
        if len(self.pair) != 2 or self.pair[0] == self.pair[1]:
            raise ValueError("reachability receipt requires two distinct arms")
        if (
            not isinstance(self.seeds, tuple)
            or len(self.seeds) != GTOK_SEED_COUNT
            or any(type(seed) is not int for seed in self.seeds)
            or len(set(self.seeds)) != GTOK_SEED_COUNT
        ):
            raise ValueError("reachability receipt requires two distinct exact seeds")
        expected = {(vocab, seed) for vocab in self.pair for seed in self.seeds}
        if {(row.vocab_size, row.seed) for row in self.rows} != expected:
            raise ValueError("reachability receipt does not cover pair by both seeds")
        if len(self.rows) != len(expected):
            raise ValueError("reachability receipt contains duplicate rows")
        if any(row.common_flop_budget != self.common_flop_budget for row in self.rows):
            raise ValueError("reachability rows use different common budgets")
        if self.all_exact_reachable is not all(row.exact_reachable for row in self.rows):
            raise ValueError("reachability aggregate differs from its physical rows")
        if self.binding_sha256 != CONFIRMATION_BINDING_SHA256_V2:
            raise ValueError("confirmation reachability binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_reachability", self)


@dataclass(frozen=True)
class ConfirmationExecutionPlanV2:
    vocab_size: int
    seed: int
    initialization_seed: int
    data_order_seed: int
    data_order_sha256: str
    common_flop_budget: int
    base_flop_evidence_sha256: str
    training_plan: TrainingPlanV2

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("confirmation plan vocabulary is unregistered")
        for name in ("seed", "initialization_seed", "data_order_seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name in ("data_order_sha256", "base_flop_evidence_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if type(self.common_flop_budget) is not int or self.common_flop_budget < 1:
            raise ValueError("confirmation plan FLOP budget must be positive")
        if self.training_plan.optimizer_steps < 100:
            raise ValueError("confirmation plan is too short for literal calibration")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_execution_plan", self)


@dataclass(frozen=True)
class ConfirmationPhysicalMeasurementV2:
    run: ComputeConfirmationRunV2
    execution_plan_sha256: str
    base_flop_evidence_sha256: str
    training_plan_sha256: str
    heldout_evaluation_steps: tuple[int, int, int]
    physical_flop_ledger_sha256: str
    physical_optimizer_steps: int
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        for name in (
            "execution_plan_sha256",
            "base_flop_evidence_sha256",
            "training_plan_sha256",
            "physical_flop_ledger_sha256",
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if tuple(sorted(set(self.heldout_evaluation_steps))) != self.heldout_evaluation_steps:
            raise ValueError("confirmation H evaluation steps must be distinct and ascending")
        if self.heldout_evaluation_steps[-1] != self.physical_optimizer_steps:
            raise ValueError("confirmation terminal H pass must follow the exact final step")
        if self.run.measured_flops != self.run.common_flop_budget:
            raise ValueError("physical confirmation missed its exact FLOP budget")
        if self.training_runtime_receipt_sha256 != self.run.training_runtime_receipt_sha256:
            raise ValueError("physical confirmation runtime join drifted")
        if self.code_closure_receipt_sha256 != self.run.code_closure_receipt_sha256:
            raise ValueError("physical confirmation code join drifted")
        if self.checkpoint_retained is not False:
            raise ValueError("physical confirmation may not retain checkpoints")


@dataclass(frozen=True)
class ComputeConfirmationCampaignResultV2:
    selection: GTokSelectionReceiptV2
    reachability: ConfirmationReachabilityReceiptV2
    preflight: PreflightProjectionReceiptV2
    compute: CampaignComputeReceiptV2
    runs: tuple[ComputeConfirmationRunV2, ...]
    confirmation: ValidatedComputeConfirmationV2
    vocab_ext_basis: VocabExtBasisV2
    vocabulary_freeze: VocabularyFreezeArtifactV2
    execution_plans: tuple[ConfirmationExecutionPlanV2, ...]
    offline_network_policy_sha256: str
    gpu_uuid_provenance_by_attempt: tuple[tuple[str, str], ...]
    offline_network_receipt_sha256_by_attempt: tuple[tuple[str, str], ...]
    binding_sha256: str = CONFIRMATION_BINDING_SHA256_V2


@dataclass(frozen=True)
class DryRunComputeConfirmationV2:
    """Non-authoritative sink exposing identities, never factory-ready values."""

    selection_receipt_sha256: str
    reachability_receipt_sha256: str
    preflight_receipt_sha256: str
    compute_receipt_sha256: str
    run_receipt_sha256s: tuple[str, ...]
    authority_status: str = "NON_AUTHORITATIVE_INJECTED_CONFIRMATION_EXECUTORS"
    binding_sha256: str = CONFIRMATION_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        values = (
            self.selection_receipt_sha256,
            self.reachability_receipt_sha256,
            self.preflight_receipt_sha256,
            self.compute_receipt_sha256,
            *self.run_receipt_sha256s,
        )
        if not self.run_receipt_sha256s or any(
            len(value) != 64 or any(character not in _HEX for character in value)
            for value in values
        ):
            raise ValueError("dry-run evidence identities must be lowercase SHA-256")


class ConfirmationCalibrationExecutorV2(Protocol):
    def __call__(
        self,
        *,
        vocab_size: int,
        tokenizer: Tokenizer,
        plan: TrainingPlanV2,
        initialization_seed: int,
        run_seed: int,
        document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    ) -> CalibrationMeasurementV2: ...


class ConfirmationFullRunExecutorV2(Protocol):
    def __call__(
        self,
        *,
        execution_plan: ConfirmationExecutionPlanV2,
        tokenizer: Tokenizer,
        base_run_receipt_sha256: str,
        compute_attempt_id: str,
        watchdog_limit_a100_microseconds: int,
        prior_campaign_a100_microseconds: int,
        gpu_uuid_provenance: str | None,
        document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    ) -> ConfirmationPhysicalMeasurementV2: ...


def build_rung_b_admissibility_v2() -> tuple[VocabularyAdmissibilityReceiptV2, ...]:
    """Price the tied vocabulary matrix at the exact rung-B width."""

    rows = tuple(
        VocabularyAdmissibilityReceiptV2(
            vocab_size=vocab_size,
            vocabulary_parameter_count=vocab_size * RUNG_B_MODEL_WIDTH_V2,
            target_parameter_count=(
                RUNG_B_ANCHOR_PARAMETER_COUNT_V2
                + (vocab_size - RUNG_B_ANCHOR_VOCAB_SIZE_V2)
                * RUNG_B_MODEL_WIDTH_V2
            ),
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )
    if any(not row.admissible for row in rows):
        raise GTokV2Stop("a registered G-TOK arm exceeds the rung-B vocabulary cap")
    return rows


def _parse_flop_ledger_v2(payload: Mapping[str, Any]) -> CompleteFlopLedgerV2:
    try:
        shapes = tuple(
            PhysicalShapeFlopReceiptV2(
                batch_rows=int(row["batch_rows"]),
                sequence_length=int(row["sequence_length"]),
                optimizer_phase=str(row["optimizer_phase"]),
                occurrences=int(row["occurrences"]),
                profiler_rows=tuple(
                    ProfilerOperatorFlopRowV2(**operator)
                    for operator in row["profiler_rows"]
                ),
                unsupported_rows=tuple(
                    AnalyticUnsupportedFlopRowV2(**operator)
                    for operator in row["unsupported_rows"]
                ),
                zero_flop_profiler_operators=tuple(
                    row["zero_flop_profiler_operators"]
                ),
            )
            for row in payload["shapes"]
        )
        return CompleteFlopLedgerV2(
            shapes=shapes,
            optimizer_steps=int(payload["optimizer_steps"]),
            compute_token_slots=int(payload["compute_token_slots"]),
            profiler_with_flops=payload.get("profiler_with_flops", True),
            flop_binding_sha256=str(payload["flop_binding_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GTokConfirmationV2Error("base physical FLOP ledger is invalid") from error


def _expand_flop_ledger_v2(ledger: CompleteFlopLedgerV2) -> tuple[BaseStepFlopV2, ...]:
    initial = tuple(row for row in ledger.shapes if row.optimizer_phase == "initial")
    steady = tuple(row for row in ledger.shapes if row.optimizer_phase == "steady")
    if len(initial) != 1 or initial[0].occurrences != 1 or not steady:
        raise GTokConfirmationV2Error("base FLOP ledger cannot reconstruct physical step order")
    if len(steady) > 2:
        raise GTokConfirmationV2Error("base FLOP ledger has an unregistered shape schedule")
    if len(steady) == 2:
        ordered_steady = tuple(
            sorted(steady, key=lambda row: (row.batch_rows, row.sequence_length), reverse=True)
        )
        if ordered_steady[-1].occurrences != 1:
            raise GTokConfirmationV2Error("only the terminal global batch may be partial")
    else:
        ordered_steady = steady
    rows: list[BaseStepFlopV2] = []

    def append_shape(shape: PhysicalShapeFlopReceiptV2, count: int) -> None:
        per_step = shape.profiler_flops_per_occurrence + shape.unsupported_flops_per_occurrence
        for _ in range(count):
            rows.append(
                BaseStepFlopV2(
                    optimizer_step=len(rows) + 1,
                    batch_rows=shape.batch_rows,
                    sequence_length=shape.sequence_length,
                    optimizer_phase="initial" if not rows else "steady",
                    measured_flops=per_step,
                )
            )

    append_shape(initial[0], 1)
    for shape in ordered_steady:
        append_shape(shape, shape.occurrences)
    if len(rows) != ledger.optimizer_steps or sum(row.measured_flops for row in rows) != ledger.measured_flops:
        raise GTokConfirmationV2Error("expanded base FLOP ledger is not complete")
    return tuple(rows)


def load_base_run_flop_evidence_v2(
    *,
    base_campaign_root: Path,
    matrix: ValidatedGTokMatrixV2,
) -> tuple[BaseRunFlopEvidenceV2, ...]:
    """Re-open profiler ledgers from completed physical base lifecycle rows."""

    if not isinstance(matrix, ValidatedGTokMatrixV2):
        raise TypeError("base FLOP evidence requires a validated G-TOK matrix")
    physical_ledger_sha256 = validate_sqlite_event_ledger_v2(
        base_campaign_root,
        matrix.compute.attempts,
    )
    if physical_ledger_sha256 != matrix.compute.event_ledger_sha256:
        raise GTokConfirmationV2Error(
            "base physical attempt ledger differs from the validated matrix"
        )
    events = validate_lifecycle_ledger_v2(base_campaign_root)
    completed = tuple(
        event
        for event in events
        if event.scope == "base_screen"
        and event.kind == "full_run"
        and event.phase == "TERMINAL"
        and event.terminal_status == "completed"
    )
    by_attempt = {run.compute_attempt_id: run for run in matrix.runs}
    rows: list[BaseRunFlopEvidenceV2] = []
    seen: set[str] = set()
    for event in completed:
        base_run = by_attempt.get(event.attempt_id)
        if base_run is None:
            continue
        if event.attempt_id in seen:
            raise GTokConfirmationV2Error("base physical run completed twice")
        payload = event.completion_payload
        if not isinstance(payload, Mapping):
            raise GTokConfirmationV2Error("base completion omits physical measurement")
        raw_run = payload.get("run")
        raw_ledger = payload.get("flop_ledger")
        if not isinstance(raw_run, Mapping) or not isinstance(raw_ledger, Mapping):
            raise GTokConfirmationV2Error("base completion omits its run/FLOP join")
        if (
            raw_run.get("vocab_size") != base_run.vocab_size
            or raw_run.get("seed") != base_run.seed
            or raw_run.get("compute_attempt_id") != base_run.compute_attempt_id
            or raw_run.get("measured_flops") != base_run.measured_flops
        ):
            raise GTokConfirmationV2Error("base physical completion differs from matrix row")
        ledger = _parse_flop_ledger_v2(raw_ledger)
        if ledger.measured_flops != base_run.measured_flops:
            raise GTokConfirmationV2Error("base matrix FLOPs differ from profiler ledger")
        rows.append(
            BaseRunFlopEvidenceV2(
                vocab_size=base_run.vocab_size,
                seed=base_run.seed,
                base_run_receipt_sha256=base_run.receipt_sha256,
                base_compute_attempt_id=base_run.compute_attempt_id,
                flop_ledger_sha256=ledger.receipt_sha256,
                steps=_expand_flop_ledger_v2(ledger),
                measured_flops=ledger.measured_flops,
            )
        )
        seen.add(event.attempt_id)
    if seen != set(by_attempt):
        raise GTokConfirmationV2Error("base campaign lacks a physical profiler ledger for every row")
    return tuple(sorted(rows, key=lambda row: (row.vocab_size, row.seed)))


def build_confirmation_reachability_v2(
    *,
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
    base_flop_evidence: tuple[BaseRunFlopEvidenceV2, ...],
) -> ConfirmationReachabilityReceiptV2:
    validate_selection_receipt_v2(selection, matrix=matrix)
    pair = selection.compute_confirmation_pair
    expected = {(vocab, seed) for vocab in pair for seed in matrix.seeds}
    selected = tuple(
        row for row in base_flop_evidence if (row.vocab_size, row.seed) in expected
    )
    if len(selected) != len(expected) or {
        (row.vocab_size, row.seed) for row in selected
    } != expected:
        raise GTokConfirmationV2Error("top-two physical FLOP evidence is incomplete")
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    if any(
        row.base_run_receipt_sha256
        != base_by_key[(row.vocab_size, row.seed)].receipt_sha256
        or row.measured_flops != base_by_key[(row.vocab_size, row.seed)].measured_flops
        for row in selected
    ):
        raise GTokConfirmationV2Error("top-two FLOP evidence differs from base matrix")
    budget = min(row.measured_flops for row in selected)
    rows: list[ConfirmationReachabilityRowV2] = []
    for evidence in sorted(selected, key=lambda row: (row.vocab_size, row.seed)):
        cumulative = 0
        reached: int | None = None
        lower = 0
        upper: int | None = None
        token_slots = 0
        for step in evidence.steps:
            next_value = cumulative + step.measured_flops
            if next_value <= budget:
                cumulative = next_value
                lower = cumulative
                token_slots += step.batch_rows * step.sequence_length
                if cumulative == budget:
                    reached = step.optimizer_step
                    break
            else:
                upper = next_value
                break
        qualified_reached = reached if reached is not None and reached >= 100 else None
        rows.append(
            ConfirmationReachabilityRowV2(
                vocab_size=evidence.vocab_size,
                seed=evidence.seed,
                base_flop_evidence_sha256=evidence.receipt_sha256,
                common_flop_budget=budget,
                reached_optimizer_steps=qualified_reached,
                reached_compute_token_slots=(
                    token_slots if qualified_reached is not None else None
                ),
                nearest_lower_flops=lower,
                nearest_upper_flops=upper,
                exact_reachable=qualified_reached is not None,
            )
        )
    return ConfirmationReachabilityReceiptV2(
        matrix_receipt_sha256=matrix.receipt_sha256,
        selection_receipt_sha256=selection.receipt_sha256,
        pair=pair,
        seeds=matrix.seeds,
        common_flop_budget=budget,
        rows=tuple(rows),
        all_exact_reachable=all(row.exact_reachable for row in rows),
    )


def build_confirmation_prefix_plan_v2(
    *,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    tokenizer: Tokenizer,
    base_evidence: BaseRunFlopEvidenceV2,
    optimizer_steps: int,
) -> TrainingPlanV2:
    """Scan the exact physical prefix and bind its tensors and terminal row."""

    if optimizer_steps < 100 or optimizer_steps > len(base_evidence.steps):
        raise GTokConfirmationV2Error("confirmation prefix step count is invalid")
    digest = hashlib.sha256()
    slots = predictions = raw_bytes = documents = 0
    observed = 0
    for step, batch in enumerate(
        iter_packed_global_batches_v2(document_factory(), tokenizer=tokenizer), start=1
    ):
        expected = base_evidence.steps[step - 1]
        if (
            batch.input_ids.shape[0] != expected.batch_rows
            or batch.input_ids.shape[1] != expected.sequence_length
        ):
            raise GTokConfirmationV2Error("confirmation prefix shape differs from base profiler row")
        _update_training_plan_digest_v2(digest, batch)
        slots += batch.input_ids.numel()
        predictions += batch.valid_prediction_count
        raw_bytes += batch.completed_raw_bytes
        documents += batch.completed_document_count
        observed = step
        if step == optimizer_steps:
            break
    if observed != optimizer_steps or min(predictions, raw_bytes, documents) < 1:
        raise GTokConfirmationV2Error("confirmation physical prefix is incomplete")
    return TrainingPlanV2(
        optimizer_steps=optimizer_steps,
        compute_token_slots=slots,
        valid_prediction_count=predictions,
        realized_raw_bytes=raw_bytes,
        document_count=documents,
        packed_stream_sha256=digest.hexdigest(),
    )


def _heldout_evaluation_steps_v2(optimizer_steps: int) -> tuple[int, int, int]:
    if optimizer_steps < 3:
        raise GTokConfirmationV2Error("confirmation needs three distinct H evaluation points")
    return (
        max(1, math.ceil(optimizer_steps / 3)),
        max(2, math.ceil(2 * optimizer_steps / 3)),
        optimizer_steps,
    )


def _execute_physical_confirmation_run_v2(
    *,
    execution_plan: ConfirmationExecutionPlanV2,
    tokenizer: Tokenizer,
    tokenizer_receipt_sha256: str,
    corpus_heldout_stream_sha256: str,
    corpus_heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]],
    base_run_receipt_sha256: str,
    compute_attempt_id: str,
    watchdog_limit_a100_microseconds: int,
    prior_campaign_a100_microseconds: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    gpu_uuid_provenance: str,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    device: torch.device,
    microbatch_sequences: int,
) -> ConfirmationPhysicalMeasurementV2:
    require_production_a100_v2(device)
    model = build_gtok_proxy_model_v2(
        vocab_size=execution_plan.vocab_size,
        initialization_seed=execution_plan.initialization_seed,
        run_seed=execution_plan.seed,
    )
    try:
        optimizer = build_flat_a1_adamw_v2(model)
        model.to(device)
        model.train()
        accountant = _PhysicalFlopAccountantV2(model, device=device)
        evaluation_steps = _heldout_evaluation_steps_v2(
            execution_plan.training_plan.optimizer_steps
        )
        terminal_strata: tuple[StratumNllReceipt, ...] | None = None
        raw_bytes = documents = predictions = slots = steps = 0
        digest = hashlib.sha256()
        start_ns = time.perf_counter_ns()

        def checked_elapsed() -> int:
            elapsed = _elapsed_microseconds(start_ns, device)
            if elapsed > watchdog_limit_a100_microseconds:
                raise GTokRunWatchdogV2(elapsed)
            if prior_campaign_a100_microseconds + elapsed > GTOK_TRIPWIRE_A100_MICROSECONDS:
                raise GTokCampaignTripwireV2(prior_campaign_a100_microseconds + elapsed)
            return elapsed

        for step, batch in enumerate(
            iter_packed_global_batches_v2(document_factory(), tokenizer=tokenizer), start=1
        ):
            if step > execution_plan.training_plan.optimizer_steps:
                break
            accountant.execute(
                batch=batch,
                step=step,
                operation=lambda batch=batch, step=step: _execute_optimizer_step_v2(
                    model,
                    optimizer,
                    batch=batch,
                    step=step,
                    plan=execution_plan.training_plan,
                    device=device,
                    microbatch_sequences=microbatch_sequences,
                ),
            )
            _update_training_plan_digest_v2(digest, batch)
            raw_bytes += batch.completed_raw_bytes
            documents += batch.completed_document_count
            predictions += batch.valid_prediction_count
            slots += batch.input_ids.numel()
            steps = step
            checked_elapsed()
            if step in evaluation_steps:
                strata = evaluate_heldout_v2(
                    model,
                    tokenizer=tokenizer,
                    heldout_factory=corpus_heldout_factory,
                    device=device,
                    microbatch_sequences=microbatch_sequences,
                )
                checked_elapsed()
                if step == evaluation_steps[-1]:
                    terminal_strata = strata
        # Calibration prices one physical output-surface benchmark per run.
        # Execute that governed work here even though the confirmation decision
        # itself consumes only terminal held-out BPB.
        measure_output_surface_performance_v2(model, device=device)
        checked_elapsed()
        plan = execution_plan.training_plan
        if (
            steps != plan.optimizer_steps
            or slots != plan.compute_token_slots
            or predictions != plan.valid_prediction_count
            or raw_bytes != plan.realized_raw_bytes
            or documents != plan.document_count
            or digest.hexdigest() != plan.packed_stream_sha256
            or terminal_strata is None
        ):
            raise GTokTrainingV2Error("confirmation execution differs from its physical prefix plan")
        ledger = accountant.finalize(plan)
        if ledger.measured_flops != execution_plan.common_flop_budget:
            raise GTokV2Stop(
                "physical profiler could not reproduce the exact common FLOP budget"
            )
        elapsed = checked_elapsed()
        run = ComputeConfirmationRunV2(
            vocab_size=execution_plan.vocab_size,
            seed=execution_plan.seed,
            base_run_receipt_sha256=base_run_receipt_sha256,
            compute_attempt_id=compute_attempt_id,
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=ledger.measured_flops,
            heldout_stream_sha256=corpus_heldout_stream_sha256,
            strata=terminal_strata,
            measured_a100_microseconds=elapsed,
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
            gpu_uuid_provenance=gpu_uuid_provenance,
        )
        return ConfirmationPhysicalMeasurementV2(
            run=run,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=plan.receipt_sha256,
            heldout_evaluation_steps=evaluation_steps,
            physical_flop_ledger_sha256=ledger.receipt_sha256,
            physical_optimizer_steps=steps,
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _basis_from_selection_v2(
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
) -> VocabExtBasisV2:
    tokenizer = next(
        row for row in matrix.tokenizers if row.vocab_size == selection.selected_vocab_size
    )
    return VocabExtBasisV2(
        vocab_size=tokenizer.vocab_size,
        tokenizer_json_sha256=tokenizer.tokenizer_json_sha256,
        merges_sha256=tokenizer.merges_sha256,
        token_inventory_sha256=tokenizer.token_inventory_sha256,
        reserved_inventory_sha256=tokenizer.reserved_inventory_sha256,
        pretokenizer_regex_sha256=tokenizer.pretokenizer_regex_sha256,
        full_corpus_manifest_sha256=matrix.corpus.full_corpus_manifest_sha256,
        screen_submanifest_sha256=matrix.corpus.screen_submanifest_sha256,
    )


def _confirmation_attempt_id(kind: str, vocab_size: int, seed: int | None = None) -> str:
    suffix = "arm" if seed is None else str(seed)
    return f"confirmation-{kind}-v{vocab_size}-s{suffix}"


def _validate_confirmation_inputs_v2(
    *,
    base: BaseCampaignResultV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    offline_network_receipt_sha256: str | None,
    offline_network_policy_sha256: str | None,
    gpu_uuid_provenance: str | None,
    authoritative: bool,
) -> None:
    if not isinstance(base, BaseCampaignResultV2) or not isinstance(base.matrix, ValidatedGTokMatrixV2):
        raise TypeError("confirmation requires an authoritative validated base campaign")
    matrix = base.matrix
    if (
        base.preflight.receipt_sha256 != matrix.compute.preflight.receipt_sha256
        or base.compute.receipt_sha256 != matrix.compute.receipt_sha256
        or tuple(row.receipt_sha256 for row in base.runs)
        != tuple(row.receipt_sha256 for row in matrix.runs)
        or base.training_runtime_receipt_sha256
        != matrix.training_runtime_receipt_sha256
        or base.code_closure_receipt_sha256 != matrix.code_closure_receipt_sha256
    ):
        raise GTokConfirmationV2Error(
            "confirmation base result differs from its validated matrix"
        )
    if (
        training_runtime_receipt_sha256 != matrix.training_runtime_receipt_sha256
        or code_closure_receipt_sha256 != matrix.code_closure_receipt_sha256
    ):
        raise GTokConfirmationV2Error("confirmation runtime/code differs from base matrix")
    if offline_network_policy_sha256 != base.offline_network_receipt_sha256:
        raise GTokConfirmationV2Error(
            "confirmation offline-network policy differs from base campaign"
        )
    if authoritative and (
        not isinstance(offline_network_receipt_sha256, str)
        or len(offline_network_receipt_sha256) != 64
        or any(
            character not in _HEX
            for character in offline_network_receipt_sha256
        )
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires a physical offline launch receipt"
        )
    if authoritative and (
        not isinstance(gpu_uuid_provenance, str)
        or not gpu_uuid_provenance.startswith("GPU-")
        or len(gpu_uuid_provenance) <= 4
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires current NVIDIA GPU provenance"
        )
    if (
        source.physical_d6_evidence_sha256 != matrix.corpus.d6_physical_evidence_sha256
        or source.training_raw_bytes != matrix.corpus.training_realized_bytes
        or source.heldout_raw_bytes_by_stratum
        != matrix.corpus.heldout_denominator_signature
    ):
        raise GTokConfirmationV2Error("confirmation physical corpus differs from base matrix")
    if tuple(row.receipt for row in tokenizer_arms) != matrix.tokenizers:
        raise GTokConfirmationV2Error("confirmation tokenizer panel differs from base matrix")
    if tuple(row[0] for row in source.training_order_receipts) != matrix.seeds:
        raise GTokConfirmationV2Error("confirmation physical training seeds drifted")
    base_data_seeds = {
        seed: {run.data_order_seed for run in matrix.runs if run.seed == seed}
        for seed in matrix.seeds
    }
    base_data_orders = {
        seed: {run.data_order_sha256 for run in matrix.runs if run.seed == seed}
        for seed in matrix.seeds
    }
    if any(len(values) != 1 for values in base_data_seeds.values()) or tuple(
        row[1] for row in source.training_order_receipts
    ) != tuple(next(iter(base_data_seeds[seed])) for seed in matrix.seeds):
        raise GTokConfirmationV2Error("confirmation physical data-order seeds drifted")
    if any(len(values) != 1 for values in base_data_orders.values()) or tuple(
        row[2] for row in source.training_order_receipts
    ) != tuple(next(iter(base_data_orders[seed])) for seed in matrix.seeds):
        raise GTokConfirmationV2Error("confirmation physical data-order identities drifted")


def _write_receipt_v2(path: Path, schema: str, value: Any) -> str:
    payload = {
        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
        "payload": asdict(value),
        "receipt_sha256": value.receipt_sha256,
        "schema": schema,
    }
    return _write_or_validate_v2(path, payload)


def _write_or_validate_v2(path: Path, value: Mapping[str, Any]) -> str:
    """Create one canonical receipt or prove the existing receipt is identical."""

    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        try:
            raw, _stored = load_canonical_json_snapshot(path)
        except (OSError, StrictJsonError, TypeError, ValueError) as error:
            raise GTokConfirmationV2Error(
                f"stored confirmation receipt is unreadable: {path.name}"
            ) from error
        if raw != expected:
            raise GTokConfirmationV2Error(
                f"stored confirmation receipt differs on resume: {path.name}"
            )
        return hashlib.sha256(raw).hexdigest()
    return _exclusive_write(path, value)


def _calibration_from_lifecycle_v2(event: Any) -> CalibrationMeasurementV2:
    if (
        event.scope != "confirmation"
        or event.kind != "calibration"
        or event.phase != "TERMINAL"
        or event.terminal_status != "completed"
        or not isinstance(event.completion_payload, Mapping)
    ):
        raise GTokConfirmationV2Error("calibration lifecycle has no completed evidence")
    try:
        measurement = CalibrationMeasurementV2(**event.completion_payload)
        if event.charged_a100_microseconds != measurement.charged_a100_microseconds:
            measurement = replace(
                measurement,
                charged_a100_microseconds=event.charged_a100_microseconds,
            )
        return measurement
    except (TypeError, ValueError) as error:
        raise GTokConfirmationV2Error(
            "completed calibration lifecycle payload is invalid"
        ) from error


def _confirmation_measurement_from_lifecycle_v2(
    event: Any,
) -> ConfirmationPhysicalMeasurementV2:
    if (
        event.scope != "confirmation"
        or event.kind != "full_run"
        or event.phase != "TERMINAL"
        or event.terminal_status != "completed"
        or not isinstance(event.completion_payload, Mapping)
    ):
        raise GTokConfirmationV2Error("confirmation lifecycle has no completed evidence")
    payload = event.completion_payload
    raw_run = payload.get("run")
    if not isinstance(raw_run, Mapping):
        raise GTokConfirmationV2Error("completed confirmation omits its run receipt")
    raw_strata = raw_run.get("strata")
    if not isinstance(raw_strata, (list, tuple)):
        raise GTokConfirmationV2Error("completed confirmation omits held-out strata")
    try:
        run = ComputeConfirmationRunV2(
            **{
                key: value
                for key, value in raw_run.items()
                if key not in ("strata", "measured_a100_microseconds")
            },
            strata=tuple(StratumNllReceipt(**row) for row in raw_strata),
            measured_a100_microseconds=event.charged_a100_microseconds,
        )
        return ConfirmationPhysicalMeasurementV2(
            **{
                key: value
                for key, value in payload.items()
                if key not in ("run", "heldout_evaluation_steps")
            },
            run=run,
            heldout_evaluation_steps=tuple(payload["heldout_evaluation_steps"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GTokConfirmationV2Error(
            "completed confirmation lifecycle payload is invalid"
        ) from error


def _terminal_events_for_logical_v2(
    events: tuple[Any, ...],
    *,
    logical_attempt_id: str,
) -> tuple[Any, ...]:
    terminal = tuple(
        event
        for event in events
        if event.logical_attempt_id == logical_attempt_id and event.phase == "TERMINAL"
    )
    completed = tuple(event for event in terminal if event.terminal_status == "completed")
    if len(completed) > 1:
        raise GTokConfirmationV2Error("logical confirmation attempt completed twice")
    return terminal


def _lifecycle_gpu_rows_v2(
    events: tuple[Any, ...],
    *,
    authoritative: bool,
) -> tuple[tuple[str, str], ...]:
    """Prove one stable physical GPU identity within each attempt."""

    values_by_attempt: dict[str, set[str | None]] = {}
    latest_by_attempt: dict[str, Any] = {}
    for event in events:
        values_by_attempt.setdefault(event.attempt_id, set()).add(
            event.gpu_uuid_provenance
        )
        latest_by_attempt[event.attempt_id] = event
    if any(len(values) != 1 for values in values_by_attempt.values()):
        raise GTokConfirmationV2Error(
            "confirmation lifecycle changed GPU identity within one attempt"
        )
    if authoritative and any(
        next(iter(values)) is None for values in values_by_attempt.values()
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation lifecycle lacks per-attempt GPU provenance"
        )
    return tuple(
        (attempt_id, str(next(iter(values_by_attempt[attempt_id]))))
        for attempt_id in sorted(latest_by_attempt)
        if next(iter(values_by_attempt[attempt_id])) is not None
    )


def _lifecycle_offline_rows_v2(
    events: tuple[Any, ...],
    *,
    authoritative: bool,
) -> tuple[tuple[str, str], ...]:
    """Prove one stable physical offline-launch receipt within each attempt."""

    values_by_attempt: dict[str, set[str | None]] = {}
    for event in events:
        values_by_attempt.setdefault(event.attempt_id, set()).add(
            event.offline_network_launch_receipt_sha256
        )
    if any(len(values) != 1 for values in values_by_attempt.values()):
        raise GTokConfirmationV2Error(
            "confirmation lifecycle changed offline identity within one attempt"
        )
    if authoritative and any(
        next(iter(values)) is None for values in values_by_attempt.values()
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation lifecycle lacks per-attempt offline provenance"
        )
    return tuple(
        (attempt_id, str(next(iter(values_by_attempt[attempt_id]))))
        for attempt_id in sorted(values_by_attempt)
        if next(iter(values_by_attempt[attempt_id])) is not None
    )


def _reconcile_terminal_attempts_v2(
    *,
    root: Path,
    attempts: list[ComputeAttemptReceiptV2],
    terminal_events: tuple[Any, ...],
    projection: ArmCalibrationProjectionV2,
    kind: str,
    vocab_size: int,
    seed: int | None,
) -> None:
    existing = {row.attempt_id: row for row in attempts}
    for event in terminal_events:
        status = event.terminal_status
        if status not in ("completed", "failed", "preempted", "aborted_watchdog"):
            raise GTokConfirmationV2Error("terminal lifecycle status is unregistered")
        if kind == "calibration" and status == "aborted_watchdog":
            raise GTokConfirmationV2Error("calibration lifecycle claims a run watchdog")
        watchdog = 2 * projection.projected_run_a100_microseconds
        exceeded = event.charged_a100_microseconds > watchdog
        if kind == "full_run" and exceeded:
            if status != "aborted_watchdog":
                raise GTokConfirmationV2Error(
                    "over-watchdog lifecycle terminal lacks hard-abort status"
                )
            hard_abort = True
        else:
            if status == "aborted_watchdog":
                raise GTokConfirmationV2Error(
                    "watchdog terminal did not strictly exceed its projection"
                )
            hard_abort = False
        receipt = ComputeAttemptReceiptV2(
            attempt_id=event.attempt_id,
            scope="confirmation",
            kind=kind,
            vocab_size=vocab_size,
            seed=seed,
            consumed_a100_microseconds=event.charged_a100_microseconds,
            status=status,
            calibration_projection_sha256=projection.receipt_sha256,
            projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
            watchdog_limit_a100_microseconds=watchdog,
            hard_abort_issued=hard_abort,
        )
        prior = existing.get(event.attempt_id)
        if prior is not None:
            if prior != receipt:
                raise GTokConfirmationV2Error(
                    "persisted attempt differs from its lifecycle terminal"
                )
            continue
        attempts.append(receipt)
        _persist_attempt(root, len(attempts) - 1, receipt)
        existing[event.attempt_id] = receipt


def _confirmation_core_v2(
    *,
    base: BaseCampaignResultV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    base_flop_evidence: tuple[BaseRunFlopEvidenceV2, ...],
    output_root: Path,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    offline_network_receipt_sha256: str | None,
    offline_network_policy_sha256: str | None,
    gpu_uuid_provenance: str | None,
    calibration_executor: ConfirmationCalibrationExecutorV2,
    full_run_executor: ConfirmationFullRunExecutorV2,
    revalidate_code_closure: Callable[[], None],
    authoritative: bool,
) -> ComputeConfirmationCampaignResultV2 | DryRunComputeConfirmationV2:
    _validate_confirmation_inputs_v2(
        base=base,
        source=source,
        tokenizer_arms=tokenizer_arms,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
        offline_network_policy_sha256=offline_network_policy_sha256,
        gpu_uuid_provenance=gpu_uuid_provenance,
        authoritative=authoritative,
    )
    matrix = base.matrix
    root = assert_no_symlink_ancestors(output_root)
    resuming = root.exists()
    if resuming:
        if root.is_symlink():
            raise GTokConfirmationV2Error("confirmation output root may not be a symlink")
        root = root.resolve(strict=True)
        if (root / "campaign-stop.json").exists():
            raise GTokV2Stop(
                "a governed confirmation STOP exists; automatic resume is prohibited"
            )
        if (root / "campaign-events.sqlite3").exists() and not (
            root / "campaign-lifecycle.sqlite3"
        ).exists():
            raise GTokConfirmationV2Error(
                "confirmation attempt ledger exists without its lifecycle ledger"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
        root = root.resolve(strict=True)
    authority = {
        "authority_status": (
            "AUTHORITATIVE_PHYSICAL_DEFAULT_EXECUTORS"
            if authoritative
            else "NON_AUTHORITATIVE_INJECTED_CONFIRMATION_EXECUTORS"
        ),
        "base_compute_receipt_sha256": matrix.compute.receipt_sha256,
        "base_matrix_receipt_sha256": matrix.receipt_sha256,
        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
        "code_closure_receipt_sha256": code_closure_receipt_sha256,
        "offline_network_policy_sha256": offline_network_policy_sha256,
        "gpu_uuid_policy": "RECORDED_PER_ATTEMPT_NOT_RUNTIME_IDENTITY",
        "microbatch_sequences": base.microbatch_sequences,
        "gradient_accumulation_slices": 256 // base.microbatch_sequences,
        "training_runtime_receipt_sha256": training_runtime_receipt_sha256,
    }
    _write_or_validate_v2(root / "confirmation-authority.json", authority)
    if resuming and (root / "campaign-lifecycle.sqlite3").exists():
        validate_lifecycle_ledger_v2(root)
        recover_orphaned_lifecycle_attempts_v2(root)
        _lifecycle_gpu_rows_v2(
            validate_lifecycle_ledger_v2(root),
            authoritative=authoritative,
        )
        _lifecycle_offline_rows_v2(
            validate_lifecycle_ledger_v2(root),
            authoritative=authoritative,
        )
    try:
        selection = select_vocabulary_v2(
            matrix,
            admissibility=build_rung_b_admissibility_v2(),
        )
    except GTokV2Stop:
        _write_stop(
            root,
            reason="SELECTION_SEED_SPLIT_OR_ADMISSIBILITY_STOP",
            cumulative=matrix.compute.consumed_a100_microseconds,
            attempts=(),
            pending=(),
            running=(),
        )
        raise
    _write_receipt_v2(
        root / "selection-receipt.json",
        "weft1_gtok_v2_selection",
        selection,
    )
    if selection.selected_vocab_size not in selection.compute_confirmation_pair:
        _write_stop(
            root,
            reason="SELECTED_VOCAB_OUTSIDE_RAW_CONFIRMATION_PAIR",
            cumulative=matrix.compute.consumed_a100_microseconds,
            attempts=(),
            pending=tuple(
                _confirmation_attempt_id("run", vocab_size, seed)
                for vocab_size in selection.compute_confirmation_pair
                for seed in matrix.seeds
            ),
            running=(),
        )
        raise GTokV2Stop(
            "selected vocabulary is outside the unchanged raw confirmation pair; "
            "return to strategy"
        )
    reachability = build_confirmation_reachability_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=base_flop_evidence,
    )
    _write_receipt_v2(
        root / "exact-flop-reachability.json",
        "weft1_gtok_v2_confirmation_reachability",
        reachability,
    )
    if not reachability.all_exact_reachable:
        _write_stop(
            root,
            reason="EXACT_COMMON_FLOP_BUDGET_UNREACHABLE",
            cumulative=matrix.compute.consumed_a100_microseconds,
            attempts=(),
            pending=tuple(
                _confirmation_attempt_id("run", row.vocab_size, row.seed)
                for row in reachability.rows
            ),
            running=(),
        )
        raise GTokV2Stop(
            "exact common FLOP budget is unreachable for all four rows; return to strategy"
        )

    tokenizers = {row.receipt.vocab_size: row.load() for row in tokenizer_arms}
    evidence_by_key = {(row.vocab_size, row.seed): row for row in base_flop_evidence}
    reach_by_key = {(row.vocab_size, row.seed): row for row in reachability.rows}
    order_by_seed = {
        training_seed: (data_seed, order_sha)
        for training_seed, data_seed, order_sha in source.training_order_receipts
    }
    initializations = {
        seed: {run.initialization_seed for run in matrix.runs if run.seed == seed}
        for seed in matrix.seeds
    }
    if any(len(values) != 1 for values in initializations.values()):
        raise GTokConfirmationV2Error("base matrix initialization seed split by vocabulary")
    init_by_seed = {
        seed: next(iter(initializations[seed])) for seed in matrix.seeds
    }
    plans: list[ConfirmationExecutionPlanV2] = []
    for vocab_size in selection.compute_confirmation_pair:
        for seed in matrix.seeds:
            evidence = evidence_by_key[(vocab_size, seed)]
            reach = reach_by_key[(vocab_size, seed)]
            assert reach.reached_optimizer_steps is not None
            data_seed, order_sha = order_by_seed[seed]
            plan = build_confirmation_prefix_plan_v2(
                document_factory=lambda seed=seed: source.training_documents(seed),
                tokenizer=tokenizers[vocab_size],
                base_evidence=evidence,
                optimizer_steps=reach.reached_optimizer_steps,
            )
            if plan.compute_token_slots != reach.reached_compute_token_slots:
                raise GTokConfirmationV2Error("physical prefix token slots differ from reachability")
            plans.append(
                ConfirmationExecutionPlanV2(
                    vocab_size=vocab_size,
                    seed=seed,
                    initialization_seed=init_by_seed[seed],
                    data_order_seed=data_seed,
                    data_order_sha256=order_sha,
                    common_flop_budget=reachability.common_flop_budget,
                    base_flop_evidence_sha256=evidence.receipt_sha256,
                    training_plan=plan,
                )
            )
    plans_tuple = tuple(plans)
    for vocab_size in selection.compute_confirmation_pair:
        arm_plans = tuple(row.training_plan for row in plans_tuple if row.vocab_size == vocab_size)
        if len({(row.optimizer_steps, row.compute_token_slots) for row in arm_plans}) != 1:
            _write_stop(
                root,
                reason="SEED_SPECIFIC_CONFIRMATION_PREFIX_SHAPE_SPLIT",
                cumulative=matrix.compute.consumed_a100_microseconds,
                attempts=(),
                pending=(),
                running=(),
            )
            raise GTokV2Stop("confirmation prefix differs by seed; return to strategy")
    revalidate_code_closure()

    attempts: list[ComputeAttemptReceiptV2] = list(
        _load_persisted_attempts_v2(root)
    )
    if any(row.scope != "confirmation" for row in attempts):
        raise GTokConfirmationV2Error(
            "confirmation attempt ledger contains another compute scope"
        )
    projections: list[ArmCalibrationProjectionV2] = []
    cumulative = matrix.compute.consumed_a100_microseconds + sum(
        row.consumed_a100_microseconds for row in attempts
    )
    for vocab_size in tuple(sorted(selection.compute_confirmation_pair)):
        revalidate_code_closure()
        plan = next(row.training_plan for row in plans_tuple if row.vocab_size == vocab_size)
        logical_attempt_id = _confirmation_attempt_id("calibration", vocab_size)
        lifecycle = (
            validate_lifecycle_ledger_v2(root)
            if (root / "campaign-lifecycle.sqlite3").exists()
            else ()
        )
        terminal_events = _terminal_events_for_logical_v2(
            lifecycle,
            logical_attempt_id=logical_attempt_id,
        )
        completed = tuple(
            row for row in terminal_events if row.terminal_status == "completed"
        )
        if completed:
            completed_event = completed[0]
            measurement = _calibration_from_lifecycle_v2(completed_event)
            attempt_id = completed_event.attempt_id
            charged = completed_event.charged_a100_microseconds
        else:
            attempt_id = _next_physical_attempt_id_v2(
                logical_attempt_id,
                lifecycle,
            )
            try:
                measurement, charged = _execute_with_lifecycle_v2(
                    root=root,
                    logical_attempt_id=logical_attempt_id,
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="calibration",
                    operation=lambda vocab_size=vocab_size, plan=plan: calibration_executor(
                        vocab_size=vocab_size,
                        tokenizer=tokenizers[vocab_size],
                        plan=plan,
                        initialization_seed=init_by_seed[matrix.seeds[0]],
                        run_seed=matrix.seeds[0],
                        document_factory=lambda: source.training_documents(matrix.seeds[0]),
                    ),
                    success_charge=lambda row: row.charged_a100_microseconds,
                    success_payload=lambda row: asdict(row),
                    gpu_uuid_provenance=gpu_uuid_provenance,
                    offline_network_launch_receipt_sha256=(
                        offline_network_receipt_sha256
                    ),
                )
                if charged != measurement.charged_a100_microseconds:
                    measurement = replace(
                        measurement,
                        charged_a100_microseconds=charged,
                    )
            except BaseException as error:
                charged = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                cumulative += charged
                lifecycle = validate_lifecycle_ledger_v2(root)
                _exclusive_write(
                    root / f"calibration-failure-v{vocab_size}.json",
                    {
                        "attempt_id": attempt_id,
                        "charged_a100_microseconds": charged,
                        "error_type": type(error).__name__,
                        "lifecycle_event_receipt_sha256s": tuple(
                            row.receipt_sha256
                            for row in lifecycle
                            if row.attempt_id == attempt_id
                        ),
                        "schema": "weft1_gtok_v2_confirmation_calibration_failure",
                    },
                )
                _write_stop(
                    root,
                    reason=f"CONFIRMATION_CALIBRATION_FAILED:{type(error).__name__}",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=(),
                    running=(),
                )
                raise
        if measurement.planned_tokens_per_run != plan.compute_token_slots:
            raise GTokConfirmationV2Error(
                "confirmation calibration priced a different prefix"
            )
        projection = _projection_from_measurement_v2(
            scope="confirmation",
            vocab_size=vocab_size,
            attempt_id=attempt_id,
            measurement=measurement,
            charged_a100_microseconds=charged,
        )
        projections.append(projection)
        terminal_events = _terminal_events_for_logical_v2(
            validate_lifecycle_ledger_v2(root),
            logical_attempt_id=logical_attempt_id,
        )
        _reconcile_terminal_attempts_v2(
            root=root,
            attempts=attempts,
            terminal_events=terminal_events,
            projection=projection,
            kind="calibration",
            vocab_size=vocab_size,
            seed=None,
        )
        cumulative = matrix.compute.consumed_a100_microseconds + sum(
            row.consumed_a100_microseconds for row in attempts
        )
        if cumulative > GTOK_TRIPWIRE_A100_MICROSECONDS:
            _write_stop(
                root,
                reason="CUMULATIVE_TRIPWIRE_DURING_CONFIRMATION_CALIBRATION",
                cumulative=cumulative,
                attempts=tuple(attempts),
                pending=(),
                running=(),
            )
            raise GTokV2Stop("confirmation calibration crossed 12 A100-hours")

    try:
        selected_calibration_ids = {
            row.calibration_attempt_id for row in projections
        }
        recovered_calibration_a100_microseconds = sum(
            row.consumed_a100_microseconds
            for row in attempts
            if row.kind == "calibration"
            and row.attempt_id not in selected_calibration_ids
        )
        preflight = build_preflight_projection_v2(
            tuple(projections),
            prior_campaign_a100_microseconds=matrix.compute.consumed_a100_microseconds,
            prior_event_ledger_sha256=matrix.compute.event_ledger_sha256,
            scope="confirmation",
            recovered_attempt_a100_microseconds=(
                recovered_calibration_a100_microseconds
            ),
        )
    except GTokV2Stop:
        _write_stop(
            root,
            reason="PROJECTED_CONFIRMATION_SCOPE_EXCEEDS_12_A100_HOURS",
            cumulative=cumulative,
            attempts=tuple(attempts),
            pending=(),
            running=(),
        )
        raise
    _write_receipt_v2(
        root / "confirmation-preflight.json",
        "weft1_gtok_v2_preflight_projection",
        preflight,
    )
    projection_by_vocab = {row.vocab_size: row for row in projections}
    base_run_by_key = {(row.vocab_size, row.seed): row for row in matrix.runs}
    runs: list[ComputeConfirmationRunV2] = []

    # A process can die after a durable full-run START but before the in-memory
    # attempt list is updated.  Once calibration projections exist, reconcile
    # every recovered full-run terminal before authorizing any retry.  This is
    # what makes orphan work part of both the 12-hour meter and the relaunch
    # projection instead of charging it only after the retry has completed.
    lifecycle = validate_lifecycle_ledger_v2(root)
    completed_logical_attempts: set[str] = set()
    for execution_plan in plans_tuple:
        logical_attempt_id = _confirmation_attempt_id(
            "run", execution_plan.vocab_size, execution_plan.seed
        )
        terminal_events = _terminal_events_for_logical_v2(
            lifecycle,
            logical_attempt_id=logical_attempt_id,
        )
        _reconcile_terminal_attempts_v2(
            root=root,
            attempts=attempts,
            terminal_events=terminal_events,
            projection=projection_by_vocab[execution_plan.vocab_size],
            kind="full_run",
            vocab_size=execution_plan.vocab_size,
            seed=execution_plan.seed,
        )
        if any(row.terminal_status == "completed" for row in terminal_events):
            completed_logical_attempts.add(logical_attempt_id)
    cumulative = matrix.compute.consumed_a100_microseconds + sum(
        row.consumed_a100_microseconds for row in attempts
    )
    pending = tuple(
        _confirmation_attempt_id("run", row.vocab_size, row.seed)
        for row in plans_tuple
        if _confirmation_attempt_id("run", row.vocab_size, row.seed)
        not in completed_logical_attempts
    )
    remaining_projection = sum(
        projection_by_vocab[row.vocab_size].projected_run_a100_microseconds
        for row in plans_tuple
        if _confirmation_attempt_id("run", row.vocab_size, row.seed)
        not in completed_logical_attempts
    )
    if cumulative + remaining_projection > GTOK_TRIPWIRE_A100_MICROSECONDS:
        _write_stop(
            root,
            reason="RESUME_PROJECTED_CONFIRMATION_EXCEEDS_12_A100_HOURS",
            cumulative=cumulative,
            attempts=tuple(attempts),
            pending=pending,
            running=(),
        )
        raise GTokV2Stop(
            "recovered confirmation work leaves no governed 12-hour budget for retries"
        )
    for execution_plan in plans_tuple:
        revalidate_code_closure()
        projection = projection_by_vocab[execution_plan.vocab_size]
        logical_attempt_id = _confirmation_attempt_id(
            "run", execution_plan.vocab_size, execution_plan.seed
        )
        pending = tuple(row for row in pending if row != logical_attempt_id)
        lifecycle = validate_lifecycle_ledger_v2(root)
        terminal_events = _terminal_events_for_logical_v2(
            lifecycle,
            logical_attempt_id=logical_attempt_id,
        )
        completed = tuple(
            row for row in terminal_events if row.terminal_status == "completed"
        )
        if completed:
            completed_event = completed[0]
            measurement = _confirmation_measurement_from_lifecycle_v2(
                completed_event
            )
            attempt_id = completed_event.attempt_id
        else:
            attempt_id = _next_physical_attempt_id_v2(
                logical_attempt_id,
                lifecycle,
            )
            try:
                measurement, charged = _execute_with_lifecycle_v2(
                    root=root,
                    logical_attempt_id=logical_attempt_id,
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="full_run",
                    operation=lambda execution_plan=execution_plan, attempt_id=attempt_id: full_run_executor(
                        execution_plan=execution_plan,
                        tokenizer=tokenizers[execution_plan.vocab_size],
                        base_run_receipt_sha256=base_run_by_key[
                            (execution_plan.vocab_size, execution_plan.seed)
                        ].receipt_sha256,
                        compute_attempt_id=attempt_id,
                        watchdog_limit_a100_microseconds=(
                            GTOK_PER_RUN_WATCHDOG_MULTIPLIER
                            * projection.projected_run_a100_microseconds
                        ),
                        prior_campaign_a100_microseconds=cumulative,
                        gpu_uuid_provenance=gpu_uuid_provenance,
                        document_factory=lambda execution_plan=execution_plan: source.training_documents(
                            execution_plan.seed
                        ),
                    ),
                    success_charge=lambda row: row.run.measured_a100_microseconds,
                    success_payload=lambda row: asdict(row),
                    gpu_uuid_provenance=gpu_uuid_provenance,
                    offline_network_launch_receipt_sha256=(
                        offline_network_receipt_sha256
                    ),
                )
                if charged != measurement.run.measured_a100_microseconds:
                    measurement = replace(
                        measurement,
                        run=replace(
                            measurement.run,
                            measured_a100_microseconds=charged,
                        ),
                    )
            except (GTokRunWatchdogV2, GTokCampaignTripwireV2) as error:
                consumed = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                watchdog = 2 * projection.projected_run_a100_microseconds
                exceeded = consumed > watchdog
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="full_run",
                    vocab_size=execution_plan.vocab_size,
                    seed=execution_plan.seed,
                    consumed_a100_microseconds=consumed,
                    status="aborted_watchdog" if exceeded else "preempted",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                    watchdog_limit_a100_microseconds=watchdog,
                    hard_abort_issued=exceeded,
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += consumed
                _write_stop(
                    root,
                    reason=(
                        "PER_RUN_WATCHDOG_STRICTLY_ABOVE_2X"
                        if exceeded
                        else "CUMULATIVE_TRIPWIRE_STRICTLY_ABOVE_12_A100_HOURS"
                    ),
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(attempt_id,),
                )
                raise GTokV2Stop("confirmation hard-aborted; return to strategy") from error
            except BaseException as error:
                consumed = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                watchdog = 2 * projection.projected_run_a100_microseconds
                exceeded = consumed > watchdog
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="full_run",
                    vocab_size=execution_plan.vocab_size,
                    seed=execution_plan.seed,
                    consumed_a100_microseconds=consumed,
                    status="aborted_watchdog" if exceeded else "failed",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                    watchdog_limit_a100_microseconds=watchdog,
                    hard_abort_issued=exceeded,
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += consumed
                _write_stop(
                    root,
                    reason=(
                        "PER_RUN_WATCHDOG_STRICTLY_ABOVE_2X"
                        if exceeded
                        else f"CONFIRMATION_RUN_FAILED:{type(error).__name__}"
                    ),
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(),
                )
                raise
        run = measurement.run
        watchdog = 2 * projection.projected_run_a100_microseconds
        if run.measured_a100_microseconds > watchdog:
            attempt = ComputeAttemptReceiptV2(
                attempt_id=attempt_id,
                scope="confirmation",
                kind="full_run",
                vocab_size=execution_plan.vocab_size,
                seed=execution_plan.seed,
                consumed_a100_microseconds=run.measured_a100_microseconds,
                status="aborted_watchdog",
                calibration_projection_sha256=projection.receipt_sha256,
                projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                watchdog_limit_a100_microseconds=watchdog,
                hard_abort_issued=True,
            )
            attempts.append(attempt)
            _persist_attempt(root, len(attempts) - 1, attempt)
            cumulative += run.measured_a100_microseconds
            _write_stop(
                root,
                reason="PER_RUN_WATCHDOG_STRICTLY_ABOVE_2X",
                cumulative=cumulative,
                attempts=tuple(attempts),
                pending=pending,
                running=(),
            )
            raise GTokV2Stop("confirmation lifecycle charge crossed the 2x watchdog")
        if (
            run.training_runtime_receipt_sha256 != training_runtime_receipt_sha256
            or run.code_closure_receipt_sha256 != code_closure_receipt_sha256
            or measurement.execution_plan_sha256 != execution_plan.receipt_sha256
            or measurement.base_flop_evidence_sha256
            != execution_plan.base_flop_evidence_sha256
        ):
            attempt = ComputeAttemptReceiptV2(
                attempt_id=attempt_id,
                scope="confirmation",
                kind="full_run",
                vocab_size=execution_plan.vocab_size,
                seed=execution_plan.seed,
                consumed_a100_microseconds=run.measured_a100_microseconds,
                status="failed",
                calibration_projection_sha256=projection.receipt_sha256,
                projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                watchdog_limit_a100_microseconds=watchdog,
            )
            attempts.append(attempt)
            _persist_attempt(root, len(attempts) - 1, attempt)
            cumulative += run.measured_a100_microseconds
            _write_stop(
                root,
                reason="CONFIRMATION_MEASUREMENT_PROVENANCE_DRIFT",
                cumulative=cumulative,
                attempts=tuple(attempts),
                pending=pending,
                running=(),
            )
            raise GTokConfirmationV2Error("confirmation measurement provenance drifted")
        attempt_events = tuple(
            row
            for row in validate_lifecycle_ledger_v2(root)
            if row.attempt_id == attempt_id
        )
        completed_attempt_events = tuple(
            row
            for row in attempt_events
            if row.phase == "TERMINAL" and row.terminal_status == "completed"
        )
        if len(completed_attempt_events) != 1 or (
            run.gpu_uuid_provenance
            != completed_attempt_events[0].gpu_uuid_provenance
        ):
            raise GTokConfirmationV2Error(
                "confirmation run GPU differs from its physical lifecycle attempt"
            )
        terminal_events = _terminal_events_for_logical_v2(
            validate_lifecycle_ledger_v2(root),
            logical_attempt_id=logical_attempt_id,
        )
        _reconcile_terminal_attempts_v2(
            root=root,
            attempts=attempts,
            terminal_events=terminal_events,
            projection=projection,
            kind="full_run",
            vocab_size=run.vocab_size,
            seed=run.seed,
        )
        cumulative = matrix.compute.consumed_a100_microseconds + sum(
            row.consumed_a100_microseconds for row in attempts
        )
        if cumulative > GTOK_TRIPWIRE_A100_MICROSECONDS:
            _write_stop(
                root,
                reason="CUMULATIVE_TRIPWIRE_STRICTLY_ABOVE_12_A100_HOURS",
                cumulative=cumulative,
                attempts=tuple(attempts),
                pending=pending,
                running=(),
            )
            raise GTokV2Stop("confirmation crossed 12 A100-hours; return to strategy")
        runs.append(run)

    final_lifecycle = validate_lifecycle_ledger_v2(root)
    latest_by_attempt: dict[str, Any] = {}
    for event in final_lifecycle:
        latest_by_attempt[event.attempt_id] = event
    if {row.attempt_id for row in attempts} != set(latest_by_attempt) or any(
        row.phase != "TERMINAL" for row in latest_by_attempt.values()
    ):
        raise GTokConfirmationV2Error(
            "confirmation attempt ledger differs from durable lifecycle terminals"
        )
    terminal_gpu_rows = _lifecycle_gpu_rows_v2(
        final_lifecycle,
        authoritative=authoritative,
    )
    terminal_offline_rows = _lifecycle_offline_rows_v2(
        final_lifecycle,
        authoritative=authoritative,
    )
    attempt_tuple = tuple(attempts)
    ledger_sha = validate_sqlite_event_ledger_v2(root, attempt_tuple)
    runtime = RuntimeTripwireSnapshotV2(
        event_ledger_sha256=ledger_sha,
        cumulative_a100_microseconds=cumulative,
        pending_attempt_ids=(),
        running_attempt_ids=(),
        hard_abort_attempt_ids=(),
        hard_abort_and_report=False,
        return_to_strategy=False,
    )
    compute = CampaignComputeReceiptV2(
        scope="confirmation",
        predecessor_campaign_sha256=matrix.compute.receipt_sha256,
        preflight=preflight,
        attempts=attempt_tuple,
        event_ledger_sha256=ledger_sha,
        consumed_a100_microseconds=cumulative,
        selected_run_a100_microseconds=sum(row.measured_a100_microseconds for row in runs),
        runtime_snapshot=runtime,
        all_attempts_accounted=True,
    )
    runs_tuple = tuple(runs)
    if not authoritative:
        dry = DryRunComputeConfirmationV2(
            selection_receipt_sha256=selection.receipt_sha256,
            reachability_receipt_sha256=reachability.receipt_sha256,
            preflight_receipt_sha256=preflight.receipt_sha256,
            compute_receipt_sha256=compute.receipt_sha256,
            run_receipt_sha256s=tuple(row.receipt_sha256 for row in runs_tuple),
        )
        _write_or_validate_v2(
            root / "non-authoritative-confirmation-dry-run.json",
            {
                "authority_status": dry.authority_status,
                "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                "compute_receipt_sha256": dry.compute_receipt_sha256,
                "runs": dry.run_receipt_sha256s,
                "schema": "weft1_gtok_v2_non_authoritative_confirmation_dry_run",
            },
        )
        return dry

    revalidate_code_closure()
    try:
        confirmation = validate_compute_confirmation_v2(
            runs_tuple,
            matrix=matrix,
            selection=selection,
            compute=compute,
        )
    except GTokV2Stop:
        _write_stop(
            root,
            reason="COMPUTE_CONFIRMATION_REVERSAL_OR_TIE",
            cumulative=cumulative,
            attempts=attempt_tuple,
            pending=(),
            running=(),
        )
        raise
    if confirmation.status != "GREEN_NO_REVERSAL":
        raise GTokConfirmationV2Error("confirmation validator did not return GREEN_NO_REVERSAL")
    basis = _basis_from_selection_v2(matrix, selection)
    freeze = mint_vocabulary_freeze_v2(
        matrix=matrix,
        selection=selection,
        confirmation=confirmation,
        basis=basis,
    )
    result = ComputeConfirmationCampaignResultV2(
        selection=selection,
        reachability=reachability,
        preflight=preflight,
        compute=compute,
        runs=runs_tuple,
        confirmation=confirmation,
        vocab_ext_basis=basis,
        vocabulary_freeze=freeze,
        execution_plans=plans_tuple,
        offline_network_policy_sha256=str(offline_network_policy_sha256),
        gpu_uuid_provenance_by_attempt=terminal_gpu_rows,
        offline_network_receipt_sha256_by_attempt=terminal_offline_rows,
    )
    _write_or_validate_v2(
        root / "vocabulary-freeze.json",
        {
            "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
            "compute": asdict(compute),
            "compute_receipt_sha256": compute.receipt_sha256,
            "confirmation": asdict(confirmation),
            "confirmation_receipt_sha256": confirmation.receipt_sha256,
            "gpu_uuid_provenance_by_attempt": terminal_gpu_rows,
            "offline_network_policy_sha256": offline_network_policy_sha256,
            "offline_network_receipt_sha256_by_attempt": terminal_offline_rows,
            "selection_receipt_sha256": selection.receipt_sha256,
            "vocab_ext_basis": asdict(basis),
            "vocab_ext_basis_sha256": basis.receipt_sha256,
            "vocabulary_freeze": asdict(freeze),
            "vocabulary_freeze_receipt_sha256": freeze.receipt_sha256,
            "schema": "weft1_gtok_v2_vocabulary_freeze",
        },
    )
    return result


def run_compute_confirmation_and_freeze_v2(
    *,
    base: BaseCampaignResultV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    base_campaign_root: Path,
    output_root: Path,
    device: torch.device,
    microbatch_sequences: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
    gpu_uuid_provenance: str,
    code_closure_receipt: GTokCodeClosureReceiptV2,
    repository_root: Path,
) -> ComputeConfirmationCampaignResultV2:
    """Run the only authoritative physical confirmation + V mint path.

    There are deliberately no executor arguments on this surface.
    """

    if (
        microbatch_sequences != GTOK_MICROBATCH_SEQUENCES_V2
        or base.microbatch_sequences != GTOK_MICROBATCH_SEQUENCES_V2
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires microbatch_sequences=8"
        )

    if code_closure_receipt.receipt_sha256 != code_closure_receipt_sha256:
        raise GTokConfirmationV2Error("code-closure payload differs from its identity")
    if (
        len(offline_network_receipt_sha256) != 64
        or any(character not in _HEX for character in offline_network_receipt_sha256)
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires a physical parent-probed offline receipt"
        )
    if (
        len(offline_network_policy_sha256) != 64
        or any(character not in _HEX for character in offline_network_policy_sha256)
        or offline_network_policy_sha256 != base.offline_network_receipt_sha256
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires the base offline policy identity"
        )
    if base.matrix.seeds != GTOK_GOVERNED_TRAINING_SEEDS_V2:
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires both A2 governed training seeds"
        )
    if tuple(row[1] for row in source.training_order_receipts) != (
        GTOK_GOVERNED_DATA_ORDER_SEEDS_V2
    ):
        raise GTokConfirmationV2Error(
            "authoritative confirmation requires both A2 governed data-order seeds"
        )

    def revalidate() -> None:
        validate_gtok_code_closure_v2(
            code_closure_receipt,
            repository_root=repository_root,
        )

    revalidate()
    evidence = load_base_run_flop_evidence_v2(
        base_campaign_root=base_campaign_root,
        matrix=base.matrix,
    )

    def calibration(**kwargs: Any) -> CalibrationMeasurementV2:
        model = build_gtok_proxy_model_v2(
            vocab_size=kwargs["vocab_size"],
            initialization_seed=kwargs["initialization_seed"],
            run_seed=kwargs["run_seed"],
        )
        try:
            return calibrate_arm_v2(
                model=model,
                tokenizer=kwargs["tokenizer"],
                document_factory=kwargs["document_factory"],
                plan=kwargs["plan"],
                device=device,
                microbatch_sequences=microbatch_sequences,
                heldout_factory=source.heldout_documents,
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    receipt_by_vocab = {row.receipt.vocab_size: row.receipt for row in tokenizer_arms}

    def full(**kwargs: Any) -> ConfirmationPhysicalMeasurementV2:
        execution_plan = kwargs["execution_plan"]
        return _execute_physical_confirmation_run_v2(
            execution_plan=execution_plan,
            tokenizer=kwargs["tokenizer"],
            tokenizer_receipt_sha256=receipt_by_vocab[
                execution_plan.vocab_size
            ].receipt_sha256,
            corpus_heldout_stream_sha256=base.matrix.corpus.heldout_stream_sha256,
            corpus_heldout_factory=source.heldout_documents,
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            watchdog_limit_a100_microseconds=kwargs[
                "watchdog_limit_a100_microseconds"
            ],
            prior_campaign_a100_microseconds=kwargs[
                "prior_campaign_a100_microseconds"
            ],
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
            gpu_uuid_provenance=gpu_uuid_provenance,
            document_factory=kwargs["document_factory"],
            device=device,
            microbatch_sequences=microbatch_sequences,
        )

    result = _confirmation_core_v2(
        base=base,
        source=source,
        tokenizer_arms=tokenizer_arms,
        base_flop_evidence=evidence,
        output_root=output_root,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
        offline_network_policy_sha256=offline_network_policy_sha256,
        gpu_uuid_provenance=gpu_uuid_provenance,
        calibration_executor=calibration,
        full_run_executor=full,
        revalidate_code_closure=revalidate,
        authoritative=True,
    )
    if not isinstance(result, ComputeConfirmationCampaignResultV2):
        raise AssertionError("authoritative confirmation returned a dry-run value")
    return result


def run_compute_confirmation_dry_run_v2(
    *,
    base: BaseCampaignResultV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    base_flop_evidence: tuple[BaseRunFlopEvidenceV2, ...],
    output_root: Path,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    calibration_executor: ConfirmationCalibrationExecutorV2,
    full_run_executor: ConfirmationFullRunExecutorV2,
    gpu_uuid_provenance: str | None = None,
    offline_network_receipt_sha256: str | None = None,
) -> DryRunComputeConfirmationV2:
    """Exercise orchestration with injected executors without mint authority."""

    result = _confirmation_core_v2(
        base=base,
        source=source,
        tokenizer_arms=tokenizer_arms,
        base_flop_evidence=base_flop_evidence,
        output_root=output_root,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
        offline_network_policy_sha256=base.offline_network_receipt_sha256,
        gpu_uuid_provenance=gpu_uuid_provenance,
        calibration_executor=calibration_executor,
        full_run_executor=full_run_executor,
        revalidate_code_closure=lambda: None,
        authoritative=False,
    )
    if not isinstance(result, DryRunComputeConfirmationV2):
        raise AssertionError("injected confirmation crossed the authoritative mint boundary")
    return result


__all__ = [
    "BaseRunFlopEvidenceV2",
    "BaseStepFlopV2",
    "CONFIRMATION_BINDING_SHA256_V2",
    "ComputeConfirmationCampaignResultV2",
    "ConfirmationExecutionPlanV2",
    "ConfirmationPhysicalMeasurementV2",
    "ConfirmationReachabilityReceiptV2",
    "ConfirmationReachabilityRowV2",
    "DryRunComputeConfirmationV2",
    "GTokConfirmationV2Error",
    "RUNG_B_ANCHOR_PARAMETER_COUNT_V2",
    "RUNG_B_ANCHOR_VOCAB_SIZE_V2",
    "RUNG_B_MODEL_WIDTH_V2",
    "build_confirmation_prefix_plan_v2",
    "build_confirmation_reachability_v2",
    "build_rung_b_admissibility_v2",
    "load_base_run_flop_evidence_v2",
    "run_compute_confirmation_and_freeze_v2",
    "run_compute_confirmation_dry_run_v2",
]
