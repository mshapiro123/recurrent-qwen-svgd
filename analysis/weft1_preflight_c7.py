"""Fail-closed PRE-FLIGHT C7 staged receipt audit.

PF-2.4 authorizes a deterministic CPU toy emission for the four already-bound
G-TOK families. This module emits and independently validates those values;
it does not manufacture the absent sidecar or either pending Jacobian line.
Consequently stage 1 can be complete while the preserved complete C7 gate
remains incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from pathlib import Path

from models.ablation_lm.accounting import CompositionReceipt
from models.ablation_lm.certificates import LoopLipschitzReceipt
from training.weft1_gtok_confirmation_v2 import (
    BaseRunFlopEvidenceV2,
    BaseStepFlopV2,
    ConfirmationArmFlopPlanV2,
    ConfirmationBudgetReceiptV2,
    build_confirmation_budget_v2,
    build_rung_b_admissibility_v2,
    floor_arm_mean_flops_v2,
    precompute_byte_checkpoint_steps_v2,
)
from training.weft1_gtok_contract import (
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
)
from training.weft1_gtok_training_v2 import (
    ConfirmationTrainingPlanV2,
    TrainingDocumentV2,
    TrainingPlanV2,
)
from training.weft1_gtok_v2_contract import (
    A2FirstFitGroupReceiptV2,
    A2FirstFitScreenReceiptV2,
    ArmCalibrationProjectionV2,
    ArmTerminalStatisticsV2,
    BpbMilestoneReceiptV2,
    CampaignComputeReceiptV2,
    ComputeAttemptReceiptV2,
    FrozenScreenCorpusV2,
    GTOK_CALIBRATION_MAX_STEPS,
    GTokRunReceiptV2,
    GTokSelectionReceiptV2,
    PreflightProjectionReceiptV2,
    RuntimeTripwireSnapshotV2,
    TokenizerArmReceiptV2,
    ValidatedGTokMatrixV2,
    compute_event_ledger_sha256_v2,
    select_vocabulary_v2,
    validate_complete_gtok_matrix_v2,
    validate_selection_receipt_v2,
)


PREFLIGHT_PROGRAM_BYTES = 15_575
PREFLIGHT_PROGRAM_SHA256 = (
    "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
)
PREFLIGHT_RATIFICATION_BYTES = 2_233
PREFLIGHT_RATIFICATION_SHA256 = (
    "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
)
PF1_AUTHORITY_BYTES = 12_285
PF1_AUTHORITY_SHA256 = (
    "4e3186c432b57f71b9f32a444a269eec08557ca5181a6896b477078dbbb40861"
)
PF2_AUTHORITY_BYTES = 13_097
PF2_AUTHORITY_SHA256 = (
    "be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05"
)
C7_STAGE1_TOY_SHA256 = (
    "04b9c1515a3902c2963eb1e13e5bfa42ede144549f88a44366953b76a422abd6"
)
GTOK_SEMANTICS_AUTHORITIES = (
    (
        "STRATEGY_GTOK_CONFIRMATION_SEMANTICS_20260831.md",
        13_975,
        "2e42664d0062a119c9fadcb76bf227a91134914920116627f9244f650defe72d",
    ),
    (
        "STRATEGY_GTOK_SEMANTICS_AMENDMENT_S1_20260831.md",
        12_411,
        "c37c4be064fe447e01182acc11b1713239c761ddd50583a8299972b4b340bd2a",
    ),
    (
        "STRATEGY_GTOK_SEMANTICS_AMENDMENT_S2_20260831.md",
        6_638,
        "5420a4e57c080d09f5f924acc859a5579edd1ca1939c8bbdaf727e5afd55ac5e",
    ),
)

FULL_GLOBAL_BATCH_TOKENS = 256 * 2_048
TOY_SEEDS = (101, 202)
TOY_OPTIMIZER_STEPS = 400
RHO_DECIMAL_QUANTUM = Decimal("0.000001")
RHO_DECIMAL_SCALE = Decimal(1_000_000)
STAGE1_FAMILIES = (
    "rho_values",
    "consumption_fields",
    "integer_f_star",
    "checkpoint_step_indices",
)
RESOLVED_C7_STATUSES = frozenset(
    {
        "emitted_and_verified",
        "struck_by_pf2_4",
    }
)
EXPECTED_C7_LINE_STATUSES = (
    ("rho_values", "emitted_and_verified"),
    ("consumption_fields", "emitted_and_verified"),
    ("integer_f_star", "emitted_and_verified"),
    ("checkpoint_step_indices", "emitted_and_verified"),
    ("gate_rate_by_k", "stage_2_pending_sidecar"),
    ("realized_eta_lambda", "struck_by_pf2_4"),
    ("lambda_adapters", "stage_2_pending_catch_26_c_jac_1"),
    ("lambda_hat_core", "stage_2_pending_catch_26_c_jac_1"),
)

CONSUMPTION_FIELDS = (
    "stream_bytes",
    "stream_tokens",
    "stream_docs",
    "trained_bytes",
    "trained_tokens",
    "trained_docs_full",
    "dropped_bytes",
    "dropped_tokens",
    "dropped_docs",
)
BOUNDARY_FIELDS = ("boundary_doc_id", "boundary_doc_consumed_tokens")
TOY_BOUNDARY_FIELDS = BOUNDARY_FIELDS + ("boundary_doc_token_length",)


class C7SchemaIncomplete(RuntimeError):
    """Raised when a caller attempts to promote the staged audit to a pass."""


def _require_exact_int(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _float64_half_even_bpb_micros(value: float) -> int:
    """Round one already-computed binary64 value half-even to six decimals."""

    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError("rho must be a positive finite binary64 float")
    with localcontext() as context:
        context.prec = 50
        rounded = Decimal.from_float(value).quantize(
            RHO_DECIMAL_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
    return int(rounded * RHO_DECIMAL_SCALE)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _toy_sha256(label: str) -> str:
    return hashlib.sha256(f"weft1-c7-stage1:{label}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class C7BoundaryTokenRecord:
    """Concrete synthetic document token record backing the strict boundary."""

    document: TrainingDocumentV2
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document, TrainingDocumentV2):
            raise TypeError("boundary source must be a production training document")
        expected = (
            1,
            *(4 + value for value in self.document.raw_bytes),
            2,
            3,
        )
        if self.token_ids != expected:
            raise ValueError("boundary token record differs from the toy byte tokenizer")


@dataclass(frozen=True)
class C7ConsumptionEmission:
    """Populated PF-2.4 consumption family plus boundary-length evidence."""

    source_run: GTokRunReceiptV2
    optimizer_steps: int
    stream_bytes: int
    stream_tokens: int
    stream_docs: int
    trained_bytes: int
    trained_tokens: int
    trained_docs_full: int
    dropped_bytes: int
    dropped_tokens: int
    dropped_docs: int
    boundary_doc_id: str | None
    boundary_doc_consumed_tokens: int | None
    boundary_source: C7BoundaryTokenRecord | None
    boundary_doc_token_length: int | None = field(init=False)

    def __post_init__(self) -> None:
        if self.boundary_source is not None and not isinstance(
            self.boundary_source,
            C7BoundaryTokenRecord,
        ):
            raise TypeError("boundary source has the wrong type")
        derived_boundary_length = (
            None if self.boundary_source is None else len(self.boundary_source.token_ids)
        )
        object.__setattr__(
            self,
            "boundary_doc_token_length",
            derived_boundary_length,
        )
        if not isinstance(self.source_run, GTokRunReceiptV2):
            raise TypeError("consumption must come from a production G-TOK run receipt")
        _require_exact_int(self.optimizer_steps, "optimizer_steps", minimum=1)
        for name in CONSUMPTION_FIELDS:
            _require_exact_int(getattr(self, name), name)
            if getattr(self.source_run, name) != getattr(self, name):
                raise ValueError(f"{name} differs from its production source run")
        if (
            self.source_run.boundary_doc_id != self.boundary_doc_id
            or self.source_run.boundary_doc_consumed_tokens
            != self.boundary_doc_consumed_tokens
        ):
            raise ValueError("boundary fields differ from the production source run")
        if not self.source_run.has_exact_stream_accounting:
            raise ValueError("source run lacks exact stream accounting")
        if self.source_run.terminal.optimizer_step != self.optimizer_steps:
            raise ValueError("n differs from the source run terminal step")
        if self.trained_tokens != self.optimizer_steps * FULL_GLOBAL_BATCH_TOKENS:
            raise ValueError("trained_tokens must equal n * 524,288")
        if self.stream_tokens != self.trained_tokens + self.dropped_tokens:
            raise ValueError("stream token accounting does not close")
        if self.stream_bytes != self.trained_bytes + self.dropped_bytes:
            raise ValueError("stream byte accounting does not close")
        if not 0 <= self.dropped_tokens < FULL_GLOBAL_BATCH_TOKENS:
            raise ValueError("dropped_tokens must be a strict partial-batch suffix")
        if self.optimizer_steps != self.stream_tokens // FULL_GLOBAL_BATCH_TOKENS:
            raise ValueError("n must equal floor(stream_tokens / 524,288)")

        boundary = (
            self.boundary_doc_id,
            self.boundary_doc_consumed_tokens,
            self.boundary_doc_token_length,
        )
        if any(value is None for value in boundary) and not all(
            value is None for value in boundary
        ):
            raise ValueError("boundary identity, consumed tokens, and length travel together")
        has_boundary = self.boundary_doc_id is not None
        if has_boundary:
            assert self.boundary_doc_id is not None
            assert self.boundary_doc_consumed_tokens is not None
            assert self.boundary_doc_token_length is not None
            assert self.boundary_source is not None
            if self.boundary_doc_id != self.boundary_source.document.raw_content_id:
                raise ValueError("boundary ID differs from its concrete token record")
            if (
                len(self.boundary_doc_id) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in self.boundary_doc_id
                )
            ):
                raise ValueError("boundary document ID must be lowercase SHA-1")
            _require_exact_int(
                self.boundary_doc_consumed_tokens,
                "boundary_doc_consumed_tokens",
                minimum=1,
            )
            _require_exact_int(
                self.boundary_doc_token_length,
                "boundary_doc_token_length",
                minimum=1,
            )
            if self.boundary_doc_consumed_tokens >= self.boundary_doc_token_length:
                raise ValueError(
                    "boundary consumed-token count must be strictly below its token length"
                )
        if self.stream_docs != (
            self.trained_docs_full + self.dropped_docs + int(has_boundary)
        ):
            raise ValueError("stream document accounting does not close")


@dataclass(frozen=True)
class C7CheckpointSeries:
    """One plan's exact byte history and governed first-crossing indices."""

    cumulative_consumed_bytes: tuple[int, ...]
    checkpoint_steps: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative_consumed_bytes, tuple):
            raise TypeError("cumulative consumed bytes must be a tuple")
        if any(
            type(value) is not int or value < 0
            for value in self.cumulative_consumed_bytes
        ):
            raise ValueError("cumulative consumed bytes must be non-negative exact integers")
        if any(
            later < earlier
            for earlier, later in zip(
                self.cumulative_consumed_bytes,
                self.cumulative_consumed_bytes[1:],
            )
        ):
            raise ValueError("cumulative consumed bytes must be monotone")
        if (
            not isinstance(self.checkpoint_steps, tuple)
            or len(self.checkpoint_steps) != 3
            or any(type(step) is not int for step in self.checkpoint_steps)
        ):
            raise ValueError("checkpoint indices must be three exact integers")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint indices must be strictly increasing")
        if self.checkpoint_steps[2] != self.n:
            raise ValueError("the third checkpoint index must equal n")
        expected = precompute_byte_checkpoint_steps_v2(
            self.cumulative_consumed_bytes
        )
        if self.checkpoint_steps != expected:
            raise ValueError("checkpoint indices are not the governed first crossings")
        for step, numerator, denominator in zip(
            self.checkpoint_steps,
            (1, 1, 1),
            (4, 2, 1),
            strict=True,
        ):
            at_step = self.cumulative_consumed_bytes[step - 1]
            previous = (
                0 if step == 1 else self.cumulative_consumed_bytes[step - 2]
            )
            if denominator * at_step < numerator * self.total_bytes:
                raise ValueError("checkpoint is below its target byte fraction")
            if denominator * previous >= numerator * self.total_bytes:
                raise ValueError("checkpoint is not the first target-byte crossing")

    @property
    def n(self) -> int:
        return len(self.cumulative_consumed_bytes)

    @property
    def total_bytes(self) -> int:
        return self.cumulative_consumed_bytes[-1]


@dataclass(frozen=True)
class C7CheckpointEmission:
    """Separate base and fresh-confirmation pre-launch checkpoint evidence."""

    base_plan: TrainingPlanV2
    base_run: GTokRunReceiptV2
    base: C7CheckpointSeries
    confirmation_plan: ConfirmationTrainingPlanV2
    confirmation_budget_row: ConfirmationArmFlopPlanV2
    confirmation_boundary_source: C7BoundaryTokenRecord
    confirmation: C7CheckpointSeries

    def __post_init__(self) -> None:
        if not isinstance(self.base_plan, TrainingPlanV2):
            raise TypeError("base checkpoints require a production pre-launch plan")
        if not isinstance(self.base_run, GTokRunReceiptV2):
            raise TypeError("base checkpoints require a production G-TOK run receipt")
        if not isinstance(self.base, C7CheckpointSeries):
            raise TypeError("base checkpoint series has the wrong type")
        if not isinstance(self.confirmation_plan, ConfirmationTrainingPlanV2):
            raise TypeError("fresh checkpoints require a confirmation pre-launch plan")
        if not isinstance(
            self.confirmation_budget_row,
            ConfirmationArmFlopPlanV2,
        ):
            raise TypeError("fresh checkpoints require their confirmation budget row")
        if not isinstance(
            self.confirmation_boundary_source,
            C7BoundaryTokenRecord,
        ):
            raise TypeError("fresh checkpoints require their boundary token record")
        if not isinstance(self.confirmation, C7CheckpointSeries):
            raise TypeError("confirmation checkpoint series has the wrong type")

        if self.base_run.terminal.optimizer_step != self.base.n:
            raise ValueError("base checkpoint n differs from the source run")
        if self.base_plan.optimizer_steps != self.base.n:
            raise ValueError("base checkpoint n differs from the pre-launch plan")
        if self.base_plan.bpb_checkpoint_steps != self.base.checkpoint_steps:
            raise ValueError("base checkpoint indices differ from the pre-launch plan")
        source_steps = tuple(
            observation.optimizer_step for observation in self.base_run.observations
        )
        if self.base.checkpoint_steps != source_steps:
            raise ValueError("base checkpoint indices differ from the source run")
        if self.base_run.trained_bytes != self.base.total_bytes:
            raise ValueError("base checkpoint bytes differ from the source run")
        if self.base_plan.trained_bytes != self.base.total_bytes:
            raise ValueError("base checkpoint bytes differ from the pre-launch plan")
        for name in CONSUMPTION_FIELDS + BOUNDARY_FIELDS:
            if getattr(self.base_plan, name) != getattr(self.base_run, name):
                raise ValueError(
                    f"base pre-launch {name} differs from the production source run"
                )
        for observation, step in zip(
            self.base_run.observations,
            self.base.checkpoint_steps,
            strict=True,
        ):
            previous = (
                0
                if step == 1
                else self.base.cumulative_consumed_bytes[step - 2]
            )
            if (
                observation.training_raw_bytes
                != self.base.cumulative_consumed_bytes[step - 1]
                or observation.previous_training_raw_bytes != previous
            ):
                raise ValueError(
                    "base checkpoint bytes differ from the production source run"
                )

        if self.confirmation_budget_row.planned_optimizer_steps != self.confirmation.n:
            raise ValueError("confirmation budget row n differs from its byte history")
        if self.confirmation_plan.optimizer_steps != self.confirmation.n:
            raise ValueError("confirmation plan n differs from its byte history")
        if self.confirmation_budget_row.planned_optimizer_steps != (
            self.confirmation_plan.optimizer_steps
        ):
            raise ValueError("confirmation budget row n differs from its plan n")
        if self.confirmation_budget_row.byte_matched_optimizer_steps != self.base.n:
            raise ValueError("confirmation budget row is not byte-matched to base n")
        if (
            self.confirmation_plan.bpb_checkpoint_steps
            != self.confirmation.checkpoint_steps
        ):
            raise ValueError("confirmation indices differ from the pre-launch plan")
        if self.confirmation_plan.trained_bytes != self.confirmation.total_bytes:
            raise ValueError("confirmation bytes differ from the pre-launch plan")
        if self.confirmation_plan.boundary_doc_id != (
            self.confirmation_boundary_source.document.raw_content_id
        ):
            raise ValueError("confirmation boundary identity differs from its token record")
        consumed_tokens = self.confirmation_plan.boundary_doc_consumed_tokens
        if (
            type(consumed_tokens) is not int
            or not 0 < consumed_tokens < len(self.confirmation_boundary_source.token_ids)
        ):
            raise ValueError("confirmation boundary must be a strict token prefix")


@dataclass(frozen=True)
class C7Stage1ToyEmission:
    """Deterministic synthetic emission of all four PF-2.4 stage-1 families."""

    source_matrix: ValidatedGTokMatrixV2
    source_selection: GTokSelectionReceiptV2
    rho_rows: tuple[ArmTerminalStatisticsV2, ...]
    consumption: C7ConsumptionEmission
    base_flop_evidence: tuple[BaseRunFlopEvidenceV2, ...]
    confirmation_budget: ConfirmationBudgetReceiptV2
    checkpoints: C7CheckpointEmission
    validated_families: tuple[str, ...] = STAGE1_FAMILIES
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_matrix, ValidatedGTokMatrixV2):
            raise TypeError("stage 1 requires a factory-validated synthetic matrix")
        if not isinstance(self.source_selection, GTokSelectionReceiptV2):
            raise TypeError("stage 1 requires a production selection receipt")
        validate_selection_receipt_v2(
            self.source_selection,
            matrix=self.source_matrix,
        )
        if not isinstance(self.rho_rows, tuple) or any(
            not isinstance(row, ArmTerminalStatisticsV2) for row in self.rho_rows
        ):
            raise TypeError("rho rows must contain ArmTerminalStatisticsV2 values")
        if tuple(row.vocab_size for row in self.rho_rows) != GTOK_VOCABULARY_ARMS:
            raise ValueError("rho rows must follow the four registered vocabulary arms")
        if self.rho_rows != self.source_selection.arm_statistics:
            raise ValueError("rho rows differ from the production selection join")
        if len({row.seeds for row in self.rho_rows}) != 1:
            raise ValueError("rho rows must preserve one recorded per-seed order")
        for row in self.rho_rows:
            if row.seed_bpbs[0] == row.seed_bpbs[1]:
                raise ValueError("toy rho rows require distinguishable per-seed BPBs")
            mean = math.fsum(row.seed_bpbs) / 2.0
            if row.rho_bpb_micros != _float64_half_even_bpb_micros(mean):
                raise ValueError(
                    "rho must be float64-computed and half-even-rounded to six decimals"
                )

        if not isinstance(self.consumption, C7ConsumptionEmission):
            raise TypeError("consumption emission has the wrong type")
        matrix_run_hashes = {
            run.receipt_sha256 for run in self.source_matrix.runs
        }
        if self.consumption.source_run.receipt_sha256 not in matrix_run_hashes:
            raise ValueError("consumption source run is absent from the validated matrix")
        if (
            not isinstance(self.base_flop_evidence, tuple)
            or any(
                not isinstance(row, BaseRunFlopEvidenceV2)
                for row in self.base_flop_evidence
            )
        ):
            raise TypeError("F* evidence must contain production base-run FLOP receipts")
        if not isinstance(self.confirmation_budget, ConfirmationBudgetReceiptV2):
            raise TypeError("confirmation budget emission has the wrong type")
        if not isinstance(self.checkpoints, C7CheckpointEmission):
            raise TypeError("checkpoint emission has the wrong type")
        if self.validated_families != STAGE1_FAMILIES:
            raise ValueError("stage-1 family inventory drifted")
        if self.synthetic_only is not True:
            raise ValueError("the PF-2.4 stage-1 emitter must remain synthetic")

        expected_budget = build_confirmation_budget_v2(
            matrix=self.source_matrix,
            selection=self.source_selection,
            base_flop_evidence=self.base_flop_evidence,
        )
        if self.confirmation_budget != expected_budget:
            raise ValueError("F* differs from the production FLOP-evidence join")

        means = tuple(
            floor_arm_mean_flops_v2(*row.base_flops)
            for row in self.confirmation_budget.rows
        )
        if (
            type(self.confirmation_budget.target_flops) is not int
            or self.confirmation_budget.target_flops != min(means)
            or tuple(row.arm_mean_flops for row in self.confirmation_budget.rows)
            != means
        ):
            raise ValueError("F* must be the exact integer min of the pair's floor means")
        if self.checkpoints.base.n != self.consumption.optimizer_steps:
            raise ValueError("base checkpoint horizon must equal consumption n")
        if (
            self.checkpoints.base_run.receipt_sha256
            != self.consumption.source_run.receipt_sha256
        ):
            raise ValueError("base checkpoints and consumption use different source runs")
        if self.checkpoints.base.total_bytes != self.consumption.trained_bytes:
            raise ValueError("base checkpoint terminal bytes must equal trained bytes")
        if self.checkpoints.confirmation_budget_row not in self.confirmation_budget.rows:
            raise ValueError("confirmation checkpoint budget row is absent from its receipt")
        if self.checkpoints.confirmation_budget_row.vocab_size != (
            self.confirmation_budget.fresh_vocab_size
        ):
            raise ValueError("confirmation checkpoints must join the fresh budget row")
        if self.confirmation_budget.fresh_vocab_size != (
            self.source_selection.selected_vocab_size
        ):
            raise ValueError("fresh confirmation row differs from the selected vocabulary")
        if self.checkpoints.confirmation.n >= self.checkpoints.base.n:
            raise ValueError("toy fresh confirmation must exercise its shorter budget horizon")
        if self.checkpoints.confirmation.total_bytes >= self.checkpoints.base.total_bytes:
            raise ValueError("toy fresh confirmation must use its own shorter byte prefix")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self)


@dataclass(frozen=True)
class C7SchemaLine:
    name: str
    status: str
    source_type: str | None
    source_fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class C7SchemaAudit:
    program_authority_sha256: str
    ratification_authority_sha256: str
    pf1_authority_sha256: str
    pf2_authority_sha256: str
    gtok_semantics_authority_sha256s: tuple[str, ...]
    authority_byte_verified: bool
    stage: str
    stage1_emission: C7Stage1ToyEmission
    stage1_emission_sha256: str
    lines: tuple[C7SchemaLine, ...]
    complete: bool
    disposition: str
    a100_hours: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.program_authority_sha256 != PREFLIGHT_PROGRAM_SHA256
            or self.ratification_authority_sha256 != PREFLIGHT_RATIFICATION_SHA256
            or self.pf1_authority_sha256 != PF1_AUTHORITY_SHA256
            or self.pf2_authority_sha256 != PF2_AUTHORITY_SHA256
        ):
            raise ValueError("C7 audit authority identity drifted")
        if self.authority_byte_verified is not True:
            raise ValueError("C7 audit may not omit byte-exact authority verification")
        if self.gtok_semantics_authority_sha256s != tuple(
            authority[2] for authority in GTOK_SEMANTICS_AUTHORITIES
        ):
            raise ValueError("C7 G-TOK semantics authority chain drifted")
        if not isinstance(self.stage1_emission, C7Stage1ToyEmission):
            raise TypeError("C7 audit requires its validated stage-1 emission")
        if (
            self.stage1_emission_sha256 != self.stage1_emission.receipt_sha256
            or self.stage1_emission_sha256 != C7_STAGE1_TOY_SHA256
        ):
            raise ValueError("C7 stage-1 emission identity drifted")
        actual_statuses = tuple((line.name, line.status) for line in self.lines)
        if actual_statuses != EXPECTED_C7_LINE_STATUSES:
            raise ValueError("C7 line inventory or staged status drifted")
        expected_complete = all(
            line.status in RESOLVED_C7_STATUSES for line in self.lines
        )
        if self.complete is not expected_complete:
            raise ValueError("C7 complete flag differs from its line statuses")
        if self.stage != "stage_1_emitted_and_verified_stage_2_pending":
            raise ValueError("C7 staged disposition drifted")
        if self.disposition != (
            "complete_c7_gate_preserved_incomplete_stage_2_pending"
        ):
            raise ValueError("C7 complete-gate disposition drifted")
        if self.a100_hours != 0.0:
            raise ValueError("the synthetic CPU C7 audit may not claim GPU use")

    def to_dict(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self))

    def require_complete(self) -> None:
        blocked = tuple(
            (line.name, line.status)
            for line in self.lines
            if line.status not in RESOLVED_C7_STATUSES
        )
        if blocked:
            raise C7SchemaIncomplete(f"C7 schema remains fail-closed: {blocked}")


def _field_names(receipt_type: type[object]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(receipt_type))


def _require_fields(
    receipt_type: type[object],
    required: tuple[str, ...],
) -> tuple[str, ...]:
    available = _field_names(receipt_type)
    missing = tuple(name for name in required if name not in available)
    if missing:
        raise C7SchemaIncomplete(
            f"{receipt_type.__name__} is missing required C7 fields {missing}"
        )
    return required


def _verify_authority_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        (
            root / "docs" / "STRATEGY_PREFLIGHT_PROGRAM_20260902.md",
            PREFLIGHT_PROGRAM_BYTES,
            PREFLIGHT_PROGRAM_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_RATIFICATION_20260902.md",
            PREFLIGHT_RATIFICATION_BYTES,
            PREFLIGHT_RATIFICATION_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_AMENDMENT_PF1_20260902.md",
            PF1_AUTHORITY_BYTES,
            PF1_AUTHORITY_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md",
            PF2_AUTHORITY_BYTES,
            PF2_AUTHORITY_SHA256,
        ),
        *tuple(
            (root / "docs" / name, expected_bytes, expected_sha256)
            for name, expected_bytes, expected_sha256 in GTOK_SEMANTICS_AUTHORITIES
        ),
    )
    for path, expected_bytes, expected_sha256 in expected:
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"C7 authority drift at {path.name}: bytes={len(payload)}, "
                f"sha256={actual_sha256}"
            )


def _toy_first_fit_groups(
    stream: str,
) -> tuple[A2FirstFitGroupReceiptV2, ...]:
    targets = dict(
        GTOK_SCREEN_TRAIN_STRATUM_TARGETS
        if stream == "T"
        else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
    )
    return tuple(
        A2FirstFitGroupReceiptV2(
            stream=stream,
            stratum=stratum,
            target_bytes=targets[stratum],
            realized_bytes=targets[stratum] - 1,
            deficit_bytes=1,
            document_count=10,
            ordered_raw_content_ids_sha256=_toy_sha256(
                f"{stream}:{stratum}:order"
            ),
        )
        for stratum in GTOK_STRATA
    )


def _toy_corpus() -> FrozenScreenCorpusV2:
    first_fit = A2FirstFitScreenReceiptV2(
        groups=(*_toy_first_fit_groups("T"), *_toy_first_fit_groups("H")),
        training_framed_stream_sha256=_toy_sha256("training-stream"),
        heldout_framed_stream_sha256=_toy_sha256("heldout-stream"),
        document_overlap_count=0,
        cluster_overlap_count=0,
    )
    return FrozenScreenCorpusV2(
        full_corpus_manifest_sha256=_toy_sha256("full-corpus-manifest"),
        screen_submanifest_sha256=_toy_sha256("screen-submanifest"),
        d6_physical_evidence_sha256=_toy_sha256("physical-d6"),
        corpus_freeze_receipt_sha256=_toy_sha256("p-b-freeze"),
        d1_d6_gate_bundle_sha256=_toy_sha256("d1-d6"),
        decontamination_receipt_sha256=_toy_sha256("decontamination"),
        first_fit=first_fit,
    )


def _toy_tokenizers(
    corpus: FrozenScreenCorpusV2,
) -> tuple[TokenizerArmReceiptV2, ...]:
    return tuple(
        TokenizerArmReceiptV2(
            vocab_size=vocab_size,
            tokenizer_json_sha256=_toy_sha256(f"tokenizer-json:{vocab_size}"),
            merges_sha256=_toy_sha256(f"merges:{vocab_size}"),
            token_inventory_sha256=_toy_sha256(f"inventory:{vocab_size}"),
            reserved_inventory_sha256=_toy_sha256("reserved-inventory"),
            pretokenizer_regex_sha256=_toy_sha256("pretokenizer-regex"),
            fit_stream_sha256=corpus.training_stream_sha256,
            full_corpus_manifest_sha256=corpus.full_corpus_manifest_sha256,
            double_fit_receipt_sha256=_toy_sha256(f"double-fit:{vocab_size}"),
            byte_round_trip_receipt_sha256=_toy_sha256("byte-round-trip"),
            token_inventory_count=vocab_size,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )


def _toy_strata(
    corpus: FrozenScreenCorpusV2,
    bpb: float,
) -> tuple[StratumNllReceipt, ...]:
    counts = dict(corpus.heldout_denominator_signature)
    return tuple(
        StratumNllReceipt(
            stratum=stratum,
            nll_nats=bpb * math.log(2.0) * counts[stratum],
            raw_byte_count=counts[stratum],
        )
        for stratum in GTOK_STRATA
    )


def _toy_terminal_bpbs() -> dict[tuple[int, int], float]:
    upper_edge = float.fromhex("0x1.0000192a73711p+0")
    lower_edge = float.fromhex("0x1.000008637bd06p+0")
    return {
        (16_384, 101): 1.41,
        (16_384, 202): 1.39,
        (24_576, 101): 1.31,
        (24_576, 202): 1.29,
        (32_768, 101): upper_edge + 1e-7,
        (32_768, 202): upper_edge - 1e-7,
        (49_152, 101): lower_edge + 1e-7,
        (49_152, 202): lower_edge - 1e-7,
    }


def _toy_boundary_source(
    label: str = "base",
) -> C7BoundaryTokenRecord:
    text = f"c7-stage1-{label}-boundary-source!"
    raw_bytes = text.encode("utf-8")
    document = TrainingDocumentV2(
        raw_content_id=hashlib.sha1(  # noqa: S324 - governed raw-content ID
            raw_bytes
        ).hexdigest(),
        text=text,
        stratum="general",
    )
    return C7BoundaryTokenRecord(
        document=document,
        token_ids=(1, *(4 + value for value in raw_bytes), 2, 3),
    )


def _toy_cumulative_consumed_bytes(trained_bytes: int) -> tuple[int, ...]:
    increments = [
        10_000_000 + ((step * 7_919) % 2_001) - 1_000
        for step in range(1, TOY_OPTIMIZER_STEPS + 1)
    ]
    increments[-1] += trained_bytes - sum(increments)
    if increments[-1] < 1:
        raise C7SchemaIncomplete("toy byte-source terminal increment is not positive")
    cumulative: list[int] = []
    consumed = 0
    for increment in increments:
        consumed += increment
        cumulative.append(consumed)
    return tuple(cumulative)


def _toy_training_plan(
    corpus: FrozenScreenCorpusV2,
    cumulative_consumed_bytes: tuple[int, ...],
    boundary_source: C7BoundaryTokenRecord,
) -> TrainingPlanV2:
    trained_tokens = TOY_OPTIMIZER_STEPS * FULL_GLOBAL_BATCH_TOKENS
    trained_bytes = cumulative_consumed_bytes[-1]
    return TrainingPlanV2(
        optimizer_steps=TOY_OPTIMIZER_STEPS,
        compute_token_slots=trained_tokens,
        valid_prediction_count=trained_tokens - TOY_OPTIMIZER_STEPS,
        realized_raw_bytes=trained_bytes,
        document_count=10_000,
        packed_stream_sha256=_toy_sha256("packed-training-stream"),
        stream_bytes=corpus.training_realized_bytes,
        stream_tokens=trained_tokens + 17,
        stream_docs=10_002,
        trained_bytes=trained_bytes,
        trained_tokens=trained_tokens,
        trained_docs_full=10_000,
        dropped_bytes=corpus.training_realized_bytes - trained_bytes,
        dropped_tokens=17,
        dropped_docs=1,
        boundary_doc_id=boundary_source.document.raw_content_id,
        boundary_doc_consumed_tokens=11,
        bpb_checkpoint_steps=precompute_byte_checkpoint_steps_v2(
            cumulative_consumed_bytes
        ),
    )


def _toy_confirmation_training_plan(
    base_plan: TrainingPlanV2,
    cumulative_consumed_bytes: tuple[int, ...],
    budget_row: ConfirmationArmFlopPlanV2,
    boundary_source: C7BoundaryTokenRecord,
) -> ConfirmationTrainingPlanV2:
    base_stream = tuple(
        getattr(base_plan, name)
        for name in ("stream_bytes", "stream_tokens", "stream_docs")
    )
    if any(value is None for value in base_stream):
        raise C7SchemaIncomplete("base plan cannot source confirmation accounting")
    optimizer_steps = budget_row.planned_optimizer_steps
    if len(cumulative_consumed_bytes) != optimizer_steps:
        raise C7SchemaIncomplete(
            "confirmation byte history must have the budget row's exact horizon"
        )
    stream_bytes = base_plan.stream_bytes
    stream_tokens = base_plan.stream_tokens
    stream_docs = base_plan.stream_docs
    assert stream_bytes is not None
    assert stream_tokens is not None
    assert stream_docs is not None
    trained_tokens = optimizer_steps * FULL_GLOBAL_BATCH_TOKENS
    trained_bytes = cumulative_consumed_bytes[-1]
    trained_docs_full = 25 * optimizer_steps
    dropped_tokens = stream_tokens - trained_tokens
    dropped_bytes = stream_bytes - trained_bytes
    dropped_docs = stream_docs - trained_docs_full - 1
    if min(dropped_tokens, dropped_bytes, dropped_docs) < 0:
        raise C7SchemaIncomplete("confirmation prefix exceeds its independent stream")
    calibration_slots = 100 * FULL_GLOBAL_BATCH_TOKENS
    return ConfirmationTrainingPlanV2(
        confirmation_order_receipt_sha256=_toy_sha256("confirmation-order"),
        optimizer_steps=optimizer_steps,
        global_batch_sequences=256,
        sequence_length=2_048,
        compute_token_slots=trained_tokens,
        valid_prediction_count=trained_tokens - optimizer_steps,
        trained_bytes=trained_bytes,
        trained_tokens=trained_tokens,
        trained_docs_full=trained_docs_full,
        boundary_doc_id=boundary_source.document.raw_content_id,
        boundary_doc_consumed_tokens=11,
        stream_bytes=stream_bytes,
        stream_tokens=stream_tokens,
        stream_docs=stream_docs,
        dropped_bytes=dropped_bytes,
        dropped_tokens=dropped_tokens,
        dropped_docs=dropped_docs,
        packed_stream_sha256=_toy_sha256("confirmation-packed-training-stream"),
        calibration_prefix_compute_token_slots=calibration_slots,
        calibration_prefix_valid_prediction_count=calibration_slots - 100,
        calibration_prefix_realized_raw_bytes=cumulative_consumed_bytes[99],
        calibration_prefix_document_count=2_500,
        calibration_prefix_packed_stream_sha256=_toy_sha256(
            "confirmation-calibration-prefix"
        ),
        bpb_checkpoint_steps=precompute_byte_checkpoint_steps_v2(
            cumulative_consumed_bytes
        ),
    )


def _toy_observations(
    corpus: FrozenScreenCorpusV2,
    terminal_bpb: float,
    cumulative_consumed_bytes: tuple[int, ...],
) -> tuple[BpbMilestoneReceiptV2, ...]:
    checkpoint_steps = precompute_byte_checkpoint_steps_v2(
        cumulative_consumed_bytes
    )
    return tuple(
        BpbMilestoneReceiptV2(
            label=label,
            optimizer_step=step,
            previous_training_raw_bytes=(
                0 if step == 1 else cumulative_consumed_bytes[step - 2]
            ),
            training_raw_bytes=cumulative_consumed_bytes[step - 1],
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_toy_strata(corpus, terminal_bpb + offset),
        )
        for label, step, offset in zip(
            ("after_1b", "after_2b", "terminal_realized_T"),
            checkpoint_steps,
            (0.20, 0.10, 0.0),
            strict=True,
        )
    )


def _toy_measured_flops(vocab_size: int, seed: int) -> int:
    pair_values = {
        (32_768, 101): 2**60 + 1,
        (32_768, 202): 2**60 + 4,
        (49_152, 101): 2**60 - 1,
        (49_152, 202): 2**60 + 2,
    }
    return pair_values.get((vocab_size, seed), 2**60 + vocab_size + seed)


def _toy_runs(
    corpus: FrozenScreenCorpusV2,
    tokenizers: tuple[TokenizerArmReceiptV2, ...],
    cumulative_consumed_bytes: tuple[int, ...],
    training_plan: TrainingPlanV2,
    boundary_source: C7BoundaryTokenRecord,
) -> tuple[GTokRunReceiptV2, ...]:
    terminal_bpbs = _toy_terminal_bpbs()
    by_vocab = {item.vocab_size: item for item in tokenizers}
    accounting = tuple(
        getattr(training_plan, name) for name in CONSUMPTION_FIELDS
    )
    if any(value is None for value in accounting):
        raise C7SchemaIncomplete("pre-launch toy plan lacks exact accounting")
    return tuple(
        GTokRunReceiptV2(
            vocab_size=vocab_size,
            seed=seed,
            frozen_screen_corpus_sha256=corpus.receipt_sha256,
            tokenizer_receipt_sha256=by_vocab[vocab_size].receipt_sha256,
            initialization_recipe_sha256=_toy_sha256("initialization-recipe"),
            initialization_seed=10_000 + seed,
            shared_initial_state_sha256=_toy_sha256(f"shared-state:{seed}"),
            data_order_seed=20_000 + seed,
            data_order_sha256=_toy_sha256(f"data-order:{seed}"),
            training_runtime_receipt_sha256=_toy_sha256("training-runtime"),
            code_closure_receipt_sha256=_toy_sha256("code-closure"),
            compute_attempt_id=f"c7-toy-base-{vocab_size}-{seed}",
            measured_a100_microseconds=100_000_000,
            measured_flops=_toy_measured_flops(vocab_size, seed),
            optimizer=a1_flat_adamw_recipe(),
            observations=_toy_observations(
                corpus,
                terminal_bpbs[(vocab_size, seed)],
                cumulative_consumed_bytes,
            ),
            stream_bytes=training_plan.stream_bytes,
            stream_tokens=training_plan.stream_tokens,
            stream_docs=training_plan.stream_docs,
            trained_bytes=training_plan.trained_bytes,
            trained_tokens=training_plan.trained_tokens,
            trained_docs_full=training_plan.trained_docs_full,
            dropped_bytes=training_plan.dropped_bytes,
            dropped_tokens=training_plan.dropped_tokens,
            dropped_docs=training_plan.dropped_docs,
            boundary_doc_id=boundary_source.document.raw_content_id,
            boundary_doc_consumed_tokens=(
                training_plan.boundary_doc_consumed_tokens
            ),
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
        for seed in TOY_SEEDS
    )


def _toy_preflight() -> PreflightProjectionReceiptV2:
    calibrations = tuple(
        ArmCalibrationProjectionV2(
            scope="base_screen",
            vocab_size=vocab_size,
            calibration_attempt_id=f"c7-toy-calibration-{vocab_size}",
            calibration_steps=GTOK_CALIBRATION_MAX_STEPS,
            measured_tokens=80 * FULL_GLOBAL_BATCH_TOKENS,
            measured_a100_microseconds=10_000_000,
            planned_tokens_per_run=(
                TOY_OPTIMIZER_STEPS * FULL_GLOBAL_BATCH_TOKENS
            ),
            projected_run_a100_microseconds=50_000_000,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )
    return PreflightProjectionReceiptV2(
        scope="base_screen",
        prior_campaign_a100_microseconds=0,
        prior_event_ledger_sha256=None,
        calibrations=calibrations,
        projected_campaign_a100_microseconds=sum(
            item.projected_scope_a100_microseconds for item in calibrations
        ),
    )


def _toy_compute(
    preflight: PreflightProjectionReceiptV2,
    runs: tuple[GTokRunReceiptV2, ...],
) -> CampaignComputeReceiptV2:
    by_vocab = {item.vocab_size: item for item in preflight.calibrations}
    calibration_attempts = tuple(
        ComputeAttemptReceiptV2(
            attempt_id=item.calibration_attempt_id,
            scope="base_screen",
            kind="calibration",
            vocab_size=item.vocab_size,
            seed=None,
            consumed_a100_microseconds=item.measured_a100_microseconds,
            status="completed",
            calibration_projection_sha256=item.receipt_sha256,
            projected_run_a100_microseconds=item.projected_run_a100_microseconds,
            watchdog_limit_a100_microseconds=(
                2 * item.projected_run_a100_microseconds
            ),
        )
        for item in preflight.calibrations
    )
    run_attempts = tuple(
        ComputeAttemptReceiptV2(
            attempt_id=run.compute_attempt_id,
            scope="base_screen",
            kind="full_run",
            vocab_size=run.vocab_size,
            seed=run.seed,
            consumed_a100_microseconds=run.measured_a100_microseconds,
            status="completed",
            calibration_projection_sha256=by_vocab[
                run.vocab_size
            ].receipt_sha256,
            projected_run_a100_microseconds=by_vocab[
                run.vocab_size
            ].projected_run_a100_microseconds,
            watchdog_limit_a100_microseconds=(
                2 * by_vocab[run.vocab_size].projected_run_a100_microseconds
            ),
        )
        for run in runs
    )
    attempts = (*calibration_attempts, *run_attempts)
    event_ledger_sha256 = compute_event_ledger_sha256_v2(attempts)
    consumed = sum(item.consumed_a100_microseconds for item in attempts)
    snapshot = RuntimeTripwireSnapshotV2(
        event_ledger_sha256=event_ledger_sha256,
        cumulative_a100_microseconds=consumed,
        pending_attempt_ids=(),
        running_attempt_ids=(),
        hard_abort_attempt_ids=(),
        hard_abort_and_report=False,
        return_to_strategy=False,
    )
    return CampaignComputeReceiptV2(
        scope="base_screen",
        predecessor_campaign_sha256=None,
        preflight=preflight,
        attempts=attempts,
        event_ledger_sha256=event_ledger_sha256,
        consumed_a100_microseconds=consumed,
        selected_run_a100_microseconds=sum(
            run.measured_a100_microseconds for run in runs
        ),
        runtime_snapshot=snapshot,
        all_attempts_accounted=True,
    )


def _toy_matrix() -> tuple[
    ValidatedGTokMatrixV2,
    tuple[int, ...],
    TrainingPlanV2,
    C7BoundaryTokenRecord,
]:
    corpus = _toy_corpus()
    tokenizers = _toy_tokenizers(corpus)
    cumulative_consumed_bytes = _toy_cumulative_consumed_bytes(
        corpus.training_realized_bytes - 1_337
    )
    boundary_source = _toy_boundary_source()
    training_plan = _toy_training_plan(
        corpus,
        cumulative_consumed_bytes,
        boundary_source,
    )
    runs = _toy_runs(
        corpus,
        tokenizers,
        cumulative_consumed_bytes,
        training_plan,
        boundary_source,
    )
    preflight = _toy_preflight()
    matrix = validate_complete_gtok_matrix_v2(
        runs,
        corpus=corpus,
        tokenizers=tokenizers,
        compute=_toy_compute(preflight, runs),
    )
    return (
        matrix,
        cumulative_consumed_bytes,
        training_plan,
        boundary_source,
    )


def _toy_base_flop_evidence(
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
) -> tuple[BaseRunFlopEvidenceV2, ...]:
    by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    rows: list[BaseRunFlopEvidenceV2] = []
    for vocab_size in selection.compute_confirmation_pair:
        for seed in matrix.seeds:
            run = by_key[(vocab_size, seed)]
            steps = tuple(
                BaseStepFlopV2(
                    optimizer_step=step,
                    batch_rows=256,
                    sequence_length=2_048,
                    optimizer_phase="initial" if step == 1 else "steady",
                    measured_flops=(
                        1
                        if step < TOY_OPTIMIZER_STEPS
                        else run.measured_flops - TOY_OPTIMIZER_STEPS + 1
                    ),
                )
                for step in range(1, TOY_OPTIMIZER_STEPS + 1)
            )
            rows.append(
                BaseRunFlopEvidenceV2(
                    vocab_size=vocab_size,
                    seed=seed,
                    base_run_receipt_sha256=run.receipt_sha256,
                    base_compute_attempt_id=run.compute_attempt_id,
                    flop_ledger_sha256=_toy_sha256(
                        f"flop-ledger:{vocab_size}:{seed}"
                    ),
                    steps=steps,
                    measured_flops=run.measured_flops,
                )
            )
    return tuple(rows)


def _toy_consumption(
    run: GTokRunReceiptV2,
    boundary_source: C7BoundaryTokenRecord,
) -> C7ConsumptionEmission:
    accounting = tuple(getattr(run, name) for name in CONSUMPTION_FIELDS)
    if any(value is None for value in accounting):
        raise C7SchemaIncomplete("synthetic source run lacks consumption accounting")
    return C7ConsumptionEmission(
        source_run=run,
        optimizer_steps=run.terminal.optimizer_step,
        stream_bytes=run.stream_bytes,  # type: ignore[arg-type]
        stream_tokens=run.stream_tokens,  # type: ignore[arg-type]
        stream_docs=run.stream_docs,  # type: ignore[arg-type]
        trained_bytes=run.trained_bytes,  # type: ignore[arg-type]
        trained_tokens=run.trained_tokens,  # type: ignore[arg-type]
        trained_docs_full=run.trained_docs_full,  # type: ignore[arg-type]
        dropped_bytes=run.dropped_bytes,  # type: ignore[arg-type]
        dropped_tokens=run.dropped_tokens,  # type: ignore[arg-type]
        dropped_docs=run.dropped_docs,  # type: ignore[arg-type]
        boundary_doc_id=run.boundary_doc_id,
        boundary_doc_consumed_tokens=run.boundary_doc_consumed_tokens,
        boundary_source=boundary_source,
    )


def emit_c7_stage1_toy() -> C7Stage1ToyEmission:
    """Drive production receipt builders with a deterministic CPU source run."""

    (
        matrix,
        cumulative_consumed_bytes,
        training_plan,
        boundary_source,
    ) = _toy_matrix()
    selection = select_vocabulary_v2(
        matrix,
        admissibility=build_rung_b_admissibility_v2(),
    )
    source_run = matrix.runs[0]
    consumption = _toy_consumption(source_run, boundary_source)
    base_flop_evidence = _toy_base_flop_evidence(matrix, selection)
    confirmation_budget = build_confirmation_budget_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=base_flop_evidence,
    )
    confirmation_budget_row = next(
        row
        for row in confirmation_budget.rows
        if row.vocab_size == confirmation_budget.fresh_vocab_size
    )
    confirmation_cumulative_consumed_bytes = cumulative_consumed_bytes[
        : confirmation_budget_row.planned_optimizer_steps
    ]
    confirmation_boundary_source = _toy_boundary_source("fresh-confirmation")
    confirmation_training_plan = _toy_confirmation_training_plan(
        training_plan,
        confirmation_cumulative_consumed_bytes,
        confirmation_budget_row,
        confirmation_boundary_source,
    )
    checkpoints = C7CheckpointEmission(
        base_plan=training_plan,
        base_run=source_run,
        base=C7CheckpointSeries(
            cumulative_consumed_bytes=cumulative_consumed_bytes,
            checkpoint_steps=precompute_byte_checkpoint_steps_v2(
                cumulative_consumed_bytes
            ),
        ),
        confirmation_plan=confirmation_training_plan,
        confirmation_budget_row=confirmation_budget_row,
        confirmation_boundary_source=confirmation_boundary_source,
        confirmation=C7CheckpointSeries(
            cumulative_consumed_bytes=confirmation_cumulative_consumed_bytes,
            checkpoint_steps=precompute_byte_checkpoint_steps_v2(
                confirmation_cumulative_consumed_bytes
            ),
        ),
    )
    return C7Stage1ToyEmission(
        source_matrix=matrix,
        source_selection=selection,
        rho_rows=selection.arm_statistics,
        consumption=consumption,
        base_flop_evidence=base_flop_evidence,
        confirmation_budget=confirmation_budget,
        checkpoints=checkpoints,
    )


def audit_preflight_c7_schema() -> C7SchemaAudit:
    """Verify authority, emit stage 1, and preserve the staged complete gate."""

    _verify_authority_bytes()
    stage1 = emit_c7_stage1_toy()
    if stage1.receipt_sha256 != C7_STAGE1_TOY_SHA256:
        raise C7SchemaIncomplete(
            "PF-2.4 deterministic stage-1 toy emission drifted: "
            f"{stage1.receipt_sha256}"
        )

    lines = (
        C7SchemaLine(
            name="rho_values",
            status="emitted_and_verified",
            source_type=(
                f"{GTokSelectionReceiptV2.__name__}/"
                f"{ArmTerminalStatisticsV2.__name__}"
            ),
            source_fields=(
                *_require_fields(GTokSelectionReceiptV2, ("arm_statistics",)),
                *_require_fields(
                    ArmTerminalStatisticsV2,
                    ("vocab_size", "seeds", "seed_bpbs", "rho_bpb_micros"),
                ),
            ),
            reason=(
                "the production selection join sorts two distinct source runs by seed "
                "for each registered arm, computes binary64 rho, and half-even rounds "
                "to six decimals"
            ),
        ),
        C7SchemaLine(
            name="consumption_fields",
            status="emitted_and_verified",
            source_type=(
                f"{GTokRunReceiptV2.__name__}/{C7ConsumptionEmission.__name__}"
            ),
            source_fields=(
                *_require_fields(
                    GTokRunReceiptV2,
                    CONSUMPTION_FIELDS + BOUNDARY_FIELDS,
                ),
                *_require_fields(
                    C7ConsumptionEmission,
                    (
                        "source_run",
                        "boundary_source",
                        "boundary_doc_token_length",
                        "optimizer_steps",
                    ),
                ),
            ),
            reason=(
                "the populated toy accounting closes tokens, bytes, and documents; "
                "trained tokens equal n times 524,288 and its sole optional boundary "
                "record is a strict token prefix"
            ),
        ),
        C7SchemaLine(
            name="integer_f_star",
            status="emitted_and_verified",
            source_type=(
                f"{BaseRunFlopEvidenceV2.__name__}/"
                f"{ConfirmationBudgetReceiptV2.__name__}"
            ),
            source_fields=(
                *_require_fields(
                    BaseRunFlopEvidenceV2,
                    ("vocab_size", "seed", "measured_flops", "steps"),
                ),
                *_require_fields(
                    ConfirmationBudgetReceiptV2,
                    ("pair", "seeds", "target_flops", "rows"),
                ),
            ),
            reason=(
                "the production budget join binds four source-run FLOP ledgers and "
                "emits F* as the exact integer min of the pair's two exact floor means"
            ),
        ),
        C7SchemaLine(
            name="checkpoint_step_indices",
            status="emitted_and_verified",
            source_type=(
                f"{TrainingPlanV2.__name__}/{ConfirmationTrainingPlanV2.__name__}/"
                f"{GTokRunReceiptV2.__name__}/{ConfirmationArmFlopPlanV2.__name__}/"
                f"{C7CheckpointSeries.__name__}/{C7CheckpointEmission.__name__}"
            ),
            source_fields=(
                *_require_fields(TrainingPlanV2, ("bpb_checkpoint_steps",)),
                *_require_fields(
                    ConfirmationTrainingPlanV2,
                    ("bpb_checkpoint_steps",),
                ),
                *_require_fields(
                    ConfirmationArmFlopPlanV2,
                    ("vocab_size", "planned_optimizer_steps"),
                ),
                *_require_fields(
                    C7CheckpointSeries,
                    ("cumulative_consumed_bytes", "checkpoint_steps"),
                ),
                *_require_fields(
                    C7CheckpointEmission,
                    (
                        "base_plan",
                        "base_run",
                        "base",
                        "confirmation_plan",
                        "confirmation_budget_row",
                        "confirmation_boundary_source",
                        "confirmation",
                    ),
                ),
            ),
            reason=(
                "production base and fresh-confirmation pre-launch plans separately "
                "record three strictly increasing first exact byte-fraction crossings; "
                "each terminal index equals its own n, the base run echoes its values, "
                "and fresh n is joined to the emitted confirmation budget row"
            ),
        ),
        C7SchemaLine(
            name="gate_rate_by_k",
            status="stage_2_pending_sidecar",
            source_type=CompositionReceipt.__name__,
            source_fields=_require_fields(
                CompositionReceipt,
                (
                    "requested_visits",
                    "executed_visits",
                    "sidecar_firing_fraction_by_step",
                ),
            ),
            reason="PF-2.4 leaves gate-rate-vs-K pending the WEFT-1 sidecar",
        ),
        C7SchemaLine(
            name="realized_eta_lambda",
            status="struck_by_pf2_4",
            source_type=None,
            source_fields=(),
            reason=(
                "PF-2.4 applies PF-1.6 explicitly: realized eta-lambda belongs to "
                "MEM-SYN-FW and is removed from C7"
            ),
        ),
        C7SchemaLine(
            name="lambda_adapters",
            status="stage_2_pending_catch_26_c_jac_1",
            source_type=LoopLipschitzReceipt.__name__,
            source_fields=_require_fields(
                LoopLipschitzReceipt,
                ("lambda_adapters",),
            ),
            reason=(
                "PF-2.4 splits Lambda_k; the certified adapter line remains pending "
                "catch #26 / C-JAC-1"
            ),
        ),
        C7SchemaLine(
            name="lambda_hat_core",
            status="stage_2_pending_catch_26_c_jac_1",
            source_type=LoopLipschitzReceipt.__name__,
            source_fields=_require_fields(
                LoopLipschitzReceipt,
                ("lambda_hat_core",),
            ),
            reason=(
                "PF-2.4 splits Lambda_k; the core line is explicitly an estimate, "
                "not a production nonlinear-visit certificate, and remains pending "
                "catch #26 / C-JAC-1"
            ),
        ),
    )
    complete = all(line.status in RESOLVED_C7_STATUSES for line in lines)
    return C7SchemaAudit(
        program_authority_sha256=PREFLIGHT_PROGRAM_SHA256,
        ratification_authority_sha256=PREFLIGHT_RATIFICATION_SHA256,
        pf1_authority_sha256=PF1_AUTHORITY_SHA256,
        pf2_authority_sha256=PF2_AUTHORITY_SHA256,
        gtok_semantics_authority_sha256s=tuple(
            authority[2] for authority in GTOK_SEMANTICS_AUTHORITIES
        ),
        authority_byte_verified=True,
        stage="stage_1_emitted_and_verified_stage_2_pending",
        stage1_emission=stage1,
        stage1_emission_sha256=stage1.receipt_sha256,
        lines=lines,
        complete=complete,
        disposition="complete_c7_gate_preserved_incomplete_stage_2_pending",
    )


def main() -> None:
    print(json.dumps(audit_preflight_c7_schema().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BOUNDARY_FIELDS",
    "C7BoundaryTokenRecord",
    "C7CheckpointEmission",
    "C7CheckpointSeries",
    "C7ConsumptionEmission",
    "C7SchemaAudit",
    "C7SchemaIncomplete",
    "C7SchemaLine",
    "C7_STAGE1_TOY_SHA256",
    "C7Stage1ToyEmission",
    "CONSUMPTION_FIELDS",
    "FULL_GLOBAL_BATCH_TOKENS",
    "GTOK_SEMANTICS_AUTHORITIES",
    "PF1_AUTHORITY_BYTES",
    "PF1_AUTHORITY_SHA256",
    "PF2_AUTHORITY_BYTES",
    "PF2_AUTHORITY_SHA256",
    "PREFLIGHT_PROGRAM_BYTES",
    "PREFLIGHT_PROGRAM_SHA256",
    "PREFLIGHT_RATIFICATION_BYTES",
    "PREFLIGHT_RATIFICATION_SHA256",
    "STAGE1_FAMILIES",
    "TOY_BOUNDARY_FIELDS",
    "audit_preflight_c7_schema",
    "emit_c7_stage1_toy",
]
