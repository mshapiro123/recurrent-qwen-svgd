"""Physical equal-FLOP confirmation and vocabulary-freeze mint for G-TOK v2.

This module is intentionally downstream of a factory-validated base matrix.  It
does four things, in order:

1. price the 20-percent guard on target rung B for every vocabulary arm;
2. select the registered top-two pair and derive one pre-launch whole-step
   confirmation horizon from exact integer FLOP evidence;
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
    _write_stop,
    build_preflight_projection_v2,
    confirmation_seed_rows_for_vocabulary_v2,
    recover_orphaned_lifecycle_attempts_v2,
    validate_lifecycle_ledger_v2,
    validate_sqlite_event_ledger_v2,
)
from training.weft1_corpus_materialize_a3 import (
    ConfirmationConsumerOrderV4,
    build_materialized_confirmation_order_v4,
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
    CompleteFlopLedgerV2,
    ConfirmationTrainingPlanV2,
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
    evaluate_heldout_v2,
    iter_packed_global_batches_v2,
    measure_output_surface_performance_v2,
    plan_confirmation_training_prefix_v2,
    require_production_a100_v2,
)
from training.weft1_gtok_v2_contract import (
    ArmCalibrationProjectionV2,
    BpbMilestoneReceiptV2,
    CampaignComputeReceiptV2,
    ConfirmationArmFlopSourceEnvelopeV2,
    ConfirmationAttemptLaunchEnvelopeV2,
    ConfirmationBaseRunFlopSourceEnvelopeV2,
    ConfirmationEvidenceClosureV2,
    ConfirmationExecutionPlanEnvelopeV2,
    ConfirmationFreshEvidenceJoinV2,
    ConfirmationLifecycleEventEvidenceV2,
    ConfirmationOrderEnvelopeV2,
    ConfirmationRetryArtifactEnvelopeV2,
    ConfirmationRetryEvidenceJoinV2,
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
CONFIRMATION_BURST_STEPS_V2 = 100
CONFIRMATION_BINDING_V2 = {
    "admissibility": (
        "V*1024/(305800000+(V-32768)*1024)<=1/5"
    ),
    "base_flop_source": "completed_physical_base_lifecycle_flop_ledgers",
    "budget": "minimum_floor_arm_mean_base_measured_flops",
    "budget_boundary": "prelaunch_n=floor(F_star*n_bytematched/F_arm)",
    "calibration": (
        "first_100_in_run_counted_steps_range_over_median_and_prelaunch_checks"
    ),
    "confirmation_evaluations": "three_H_passes_at_first_quarter_half_and_terminal_byte_crossings",
    "gpu_provenance": "physical_uuid_per_attempt_not_runtime_identity",
    "model_state": "fresh_per_attempt_no_checkpoint_or_optimizer_resume",
    "result_construction": "min_F_arm_reused_two_base_slots_max_F_arm_two_fresh_Q3_slots",
    "seed_pairing": "slot_0_with_slot_0_and_slot_1_with_slot_1",
    "offline_network": (
        "stable_policy_identity_plus_physical_parent_launch_receipt_per_attempt"
    ),
    "runtime": "same_training_runtime_and_code_closure_as_base_matrix",
    "selection": "D-C-2_winner_W_then_raw_rho_runner_up_U",
}
CONFIRMATION_BINDING_SHA256_V2 = hashlib.sha256(
    canonical_json_bytes(CONFIRMATION_BINDING_V2)
).hexdigest()
_HEX = frozenset("0123456789abcdef")


class GTokConfirmationV2Error(RuntimeError):
    """Physical confirmation evidence is absent, inconsistent, or unsafe."""


class ConfirmationFlopBandViolationV2(GTokV2Stop):
    """A completed fresh run missed the inclusive one-percent FLOP band."""

    def __init__(
        self,
        *,
        realized_flops: int,
        target_flops: int,
        retry_steps: int,
        flop_ledger: CompleteFlopLedgerV2,
        burst_flop_receipt: ConfirmationBurstFlopReceiptV2,
    ) -> None:
        if not isinstance(flop_ledger, CompleteFlopLedgerV2):
            raise TypeError("invalid-band retry requires its complete physical FLOP ledger")
        if realized_flops != flop_ledger.measured_flops:
            raise ValueError("invalid-band FLOPs differ from the complete physical ledger")
        if not isinstance(burst_flop_receipt, ConfirmationBurstFlopReceiptV2):
            raise TypeError("invalid-band retry requires its passed burst receipt")
        self.realized_flops = realized_flops
        self.target_flops = target_flops
        self.retry_steps = retry_steps
        self.flop_ledger = flop_ledger
        self.burst_flop_receipt = burst_flop_receipt
        super().__init__(
            "confirmation realized FLOPs missed the one-percent target band; "
            f"fresh retry requires {retry_steps} steps"
        )


def floor_arm_mean_flops_v2(seed0_flops: int, seed1_flops: int) -> int:
    """S1-L4's exact integer arm total, with no float conversion."""

    for name, value in (("seed0_flops", seed0_flops), ("seed1_flops", seed1_flops)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    return (seed0_flops + seed1_flops) // 2


def prelaunch_confirmation_steps_v2(
    *,
    target_flops: int,
    arm_mean_flops: int,
    byte_matched_optimizer_steps: int,
) -> int:
    """Compute ``floor(F*/f_step)`` as exact integer division.

    S2-Q5 defines ``f_step = arm_mean_flops / byte_matched_optimizer_steps``.
    Cross multiplication keeps the governed boundary exact even above 2**53.
    """

    for name, value in (
        ("target_flops", target_flops),
        ("arm_mean_flops", arm_mean_flops),
        ("byte_matched_optimizer_steps", byte_matched_optimizer_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    steps = (target_flops * byte_matched_optimizer_steps) // arm_mean_flops
    if steps < CONFIRMATION_BURST_STEPS_V2:
        raise GTokV2Stop("pre-launch confirmation horizon is shorter than 100 governed steps")
    return steps


def confirmation_flops_within_target_v2(
    *, realized_flops: int, target_flops: int
) -> bool:
    """Inclusive S1-L4 one-percent end-of-run validity, exactly."""

    for name, value in (("realized_flops", realized_flops), ("target_flops", target_flops)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    return 100 * abs(realized_flops - target_flops) <= target_flops


def confirmation_retry_steps_v2(
    *, target_flops: int, realized_flops: int, optimizer_steps: int
) -> int:
    """Exact S1-L4 retry horizon ``floor(F* n / F_realized)``."""

    for name, value in (
        ("target_flops", target_flops),
        ("realized_flops", realized_flops),
        ("optimizer_steps", optimizer_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    retry_steps = (target_flops * optimizer_steps) // realized_flops
    if retry_steps < CONFIRMATION_BURST_STEPS_V2:
        raise GTokV2Stop("corrected confirmation horizon is shorter than 100 governed steps")
    return retry_steps


@dataclass(frozen=True)
class ConfirmationBurstFlopReceiptV2:
    """Exact S2-Q4/Q5 evidence available immediately after optimizer step 100."""

    ordered_step_flops: tuple[int, ...]
    prelaunch_arm_mean_flops: int
    byte_matched_optimizer_steps: int

    def __post_init__(self) -> None:
        if len(self.ordered_step_flops) != CONFIRMATION_BURST_STEPS_V2 or any(
            type(value) is not int or value < 1 for value in self.ordered_step_flops
        ):
            raise ValueError("confirmation burst requires 100 positive exact FLOP values")
        for name in ("prelaunch_arm_mean_flops", "byte_matched_optimizer_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")

        ordered = tuple(sorted(self.ordered_step_flops))
        median_twice = ordered[49] + ordered[50]
        # S=(max-min)/median; stop iff S>0.01.  Doubling the even-sample
        # median avoids introducing a float at the strict boundary.
        if 200 * (ordered[-1] - ordered[0]) > median_twice:
            raise GTokV2Stop("100-step FLOP range/median stability exceeded one percent")
        # S2-Q4 defines the burst f_step as the arithmetic mean of these
        # 100 measurements.  Compare that mean with the inherited pre-launch
        # f_step exactly; the independent range/median gate above governs
        # within-burst variation.
        if abs(
            sum(self.ordered_step_flops) * self.byte_matched_optimizer_steps
            - CONFIRMATION_BURST_STEPS_V2 * self.prelaunch_arm_mean_flops
        ) > self.prelaunch_arm_mean_flops:
            raise GTokV2Stop("100-step FLOPs differ from pre-launch f_step by over one percent")

    @property
    def measured_flops(self) -> int:
        return sum(self.ordered_step_flops)

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_burst_flops", self)


@dataclass(frozen=True)
class ConfirmationBurstGateEvidenceV2:
    """Auditable 100-step preimage, including a gate that may fail closed."""

    ordered_step_flops: tuple[int, ...]
    prelaunch_arm_mean_flops: int
    byte_matched_optimizer_steps: int

    def __post_init__(self) -> None:
        if len(self.ordered_step_flops) != CONFIRMATION_BURST_STEPS_V2 or any(
            type(value) is not int or value < 1 for value in self.ordered_step_flops
        ):
            raise ValueError("confirmation burst gate requires 100 positive FLOP values")
        for name in ("prelaunch_arm_mean_flops", "byte_matched_optimizer_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")

    @property
    def stability_passed(self) -> bool:
        ordered = tuple(sorted(self.ordered_step_flops))
        return 200 * (ordered[-1] - ordered[0]) <= ordered[49] + ordered[50]

    @property
    def inherited_rate_passed(self) -> bool:
        return abs(
            sum(self.ordered_step_flops) * self.byte_matched_optimizer_steps
            - CONFIRMATION_BURST_STEPS_V2 * self.prelaunch_arm_mean_flops
        ) <= self.prelaunch_arm_mean_flops

    @property
    def status(self) -> str:
        if not self.stability_passed:
            return "STOP_RANGE_OVER_MEDIAN_ABOVE_ONE_PERCENT"
        if not self.inherited_rate_passed:
            return "STOP_INHERITED_FSTEP_DRIFT_ABOVE_ONE_PERCENT"
        return "GREEN_100_STEP_BURST"

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_burst_gate_evidence",
            self,
        )


class ConfirmationBurstGateViolationV2(GTokV2Stop):
    """A governed Q4 burst fired and carries the exact ordered measurements."""

    def __init__(self, evidence: ConfirmationBurstGateEvidenceV2) -> None:
        if not isinstance(evidence, ConfirmationBurstGateEvidenceV2):
            raise TypeError("Q4 burst violation requires typed evidence")
        if evidence.status == "GREEN_100_STEP_BURST":
            raise ValueError("green Q4 burst evidence cannot raise a stop")
        self.evidence = evidence
        super().__init__(evidence.status)


def confirmation_physical_burst_evidence_sha256_v2(
    *,
    compute_attempt_id: str,
    execution_plan_sha256: str,
    burst: ConfirmationBurstFlopReceiptV2,
) -> str:
    """Bind a reusable burst payload to the physical attempt that observed it."""

    if not isinstance(compute_attempt_id, str) or not compute_attempt_id:
        raise ValueError("physical burst evidence requires its compute attempt ID")
    if len(execution_plan_sha256) != 64 or any(
        character not in _HEX for character in execution_plan_sha256
    ):
        raise ValueError("physical burst evidence requires its execution-plan SHA-256")
    if not isinstance(burst, ConfirmationBurstFlopReceiptV2):
        raise TypeError("physical burst evidence requires a typed burst receipt")
    return gtok_v2_bound_sha256(
        "weft1_gtok_v2_confirmation_physical_burst_evidence",
        {
            "burst_receipt_sha256": burst.receipt_sha256,
            "compute_attempt_id": compute_attempt_id,
            "execution_plan_sha256": execution_plan_sha256,
        },
    )


def confirmation_physical_flop_ledger_evidence_sha256_v2(
    *,
    compute_attempt_id: str,
    execution_plan_sha256: str,
    flop_ledger: CompleteFlopLedgerV2,
) -> str:
    """Bind a reusable FLOP ledger payload to one physical attempt and plan."""

    if not isinstance(compute_attempt_id, str) or not compute_attempt_id:
        raise ValueError("physical FLOP evidence requires its compute attempt ID")
    if len(execution_plan_sha256) != 64 or any(
        character not in _HEX for character in execution_plan_sha256
    ):
        raise ValueError("physical FLOP evidence requires its execution-plan SHA-256")
    if not isinstance(flop_ledger, CompleteFlopLedgerV2):
        raise TypeError("physical FLOP evidence requires a complete typed ledger")
    return gtok_v2_bound_sha256(
        "weft1_gtok_v2_confirmation_physical_flop_ledger_evidence",
        {
            "compute_attempt_id": compute_attempt_id,
            "execution_plan_sha256": execution_plan_sha256,
            "flop_ledger_receipt_sha256": flop_ledger.receipt_sha256,
        },
    )


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
class ConfirmationArmFlopPlanV2:
    """One arm's exact pre-launch S2-Q5 calculation."""

    vocab_size: int
    seeds: tuple[int, int]
    base_flops: tuple[int, int]
    base_flop_evidence_sha256s: tuple[str, str]
    byte_matched_optimizer_steps: int
    arm_mean_flops: int
    target_flops: int
    planned_optimizer_steps: int

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("confirmation arm plan uses an unregistered vocabulary")
        if len(self.seeds) != 2 or len(set(self.seeds)) != 2 or any(
            type(seed) is not int for seed in self.seeds
        ):
            raise ValueError("confirmation arm plan requires two distinct exact seeds")
        if len(self.base_flops) != 2:
            raise ValueError("confirmation arm plan requires two base FLOP totals")
        if len(self.base_flop_evidence_sha256s) != 2 or any(
            len(value) != 64 or any(character not in _HEX for character in value)
            for value in self.base_flop_evidence_sha256s
        ):
            raise ValueError("confirmation arm plan requires two base evidence identities")
        expected_mean = floor_arm_mean_flops_v2(*self.base_flops)
        if self.arm_mean_flops != expected_mean:
            raise ValueError("confirmation arm mean is not the floor of its seed totals")
        expected_steps = prelaunch_confirmation_steps_v2(
            target_flops=self.target_flops,
            arm_mean_flops=self.arm_mean_flops,
            byte_matched_optimizer_steps=self.byte_matched_optimizer_steps,
        )
        if self.planned_optimizer_steps != expected_steps:
            raise ValueError("confirmation arm plan uses an inexact pre-launch horizon")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_arm_flop_plan", self)


@dataclass(frozen=True)
class ConfirmationBudgetReceiptV2:
    matrix_receipt_sha256: str
    selection_receipt_sha256: str
    pair: tuple[int, int]
    seeds: tuple[int, int]
    target_flops: int
    rows: tuple[ConfirmationArmFlopPlanV2, ...]
    binding_sha256: str = CONFIRMATION_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        for value in (self.matrix_receipt_sha256, self.selection_receipt_sha256):
            if len(value) != 64:
                raise ValueError("confirmation budget join must be SHA-256")
        if len(self.pair) != 2 or self.pair[0] == self.pair[1]:
            raise ValueError("confirmation budget requires two distinct arms")
        if (
            not isinstance(self.seeds, tuple)
            or len(self.seeds) != GTOK_SEED_COUNT
            or any(type(seed) is not int for seed in self.seeds)
            or len(set(self.seeds)) != GTOK_SEED_COUNT
        ):
            raise ValueError("confirmation budget requires two distinct exact seeds")
        if tuple(row.vocab_size for row in self.rows) != tuple(sorted(self.pair)):
            raise ValueError("confirmation budget requires one ordered row per arm")
        if any(row.seeds != self.seeds for row in self.rows):
            raise ValueError("confirmation budget arm seed order drifted")
        expected_target = min(row.arm_mean_flops for row in self.rows)
        if len({row.arm_mean_flops for row in self.rows}) != 2:
            raise GTokV2Stop(
                "confirmation pair has no unique min-FLOP arm; return to strategy"
            )
        if self.target_flops != expected_target or any(
            row.target_flops != self.target_flops for row in self.rows
        ):
            raise ValueError("confirmation budget target is not min floor arm mean")
        if self.binding_sha256 != CONFIRMATION_BINDING_SHA256_V2:
            raise ValueError("confirmation budget binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_budget", self)

    @property
    def reused_vocab_size(self) -> int:
        return min(self.rows, key=lambda row: row.arm_mean_flops).vocab_size

    @property
    def fresh_vocab_size(self) -> int:
        return next(
            row.vocab_size for row in self.rows if row.vocab_size != self.reused_vocab_size
        )


@dataclass(frozen=True)
class ConfirmationExecutionPlanV2:
    vocab_size: int
    seed_slot: int
    registry_key: str
    seed: int
    initialization_seed: int
    data_order_seed: int
    data_order_sha256: str
    confirmation_order_receipt_sha256: str
    physical_d6_evidence_sha256: str
    document_multiset_sha256: str
    framed_payload_sha256: str
    order_document_count: int
    order_retained_text_bytes: int
    target_flops: int
    arm_mean_flops: int
    byte_matched_optimizer_steps: int
    arm_flop_plan_sha256: str
    training_plan: ConfirmationTrainingPlanV2
    retry_of_realized_flops: int | None = None
    retry_of_optimizer_steps: int | None = None
    heldout_evaluation_steps: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("confirmation plan vocabulary is unregistered")
        if self.seed_slot not in (0, 1):
            raise ValueError("confirmation plan seed slot must be 0 or 1")
        if self.registry_key != f"gtok.confirm.{self.vocab_size}.{self.seed_slot}":
            raise ValueError("confirmation plan registry key drifted")
        for name in ("seed", "initialization_seed", "data_order_seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name in (
            "data_order_sha256",
            "confirmation_order_receipt_sha256",
            "physical_d6_evidence_sha256",
            "document_multiset_sha256",
            "framed_payload_sha256",
            "arm_flop_plan_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if type(self.order_document_count) is not int or self.order_document_count < 1:
            raise ValueError("confirmation order document count must be positive")
        if (
            type(self.order_retained_text_bytes) is not int
            or self.order_retained_text_bytes < 1
        ):
            raise ValueError("confirmation order retained bytes must be positive")
        if (
            self.training_plan.confirmation_order_receipt_sha256
            != self.confirmation_order_receipt_sha256
        ):
            raise ValueError("confirmation training plan differs from its Q3 order receipt")
        if (
            self.training_plan.stream_docs != self.order_document_count
            or self.training_plan.stream_bytes != self.order_retained_text_bytes
        ):
            raise ValueError("confirmation training plan differs from Q3 order accounting")
        for name in ("target_flops", "arm_mean_flops", "byte_matched_optimizer_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if (self.retry_of_realized_flops is None) != (
            self.retry_of_optimizer_steps is None
        ):
            raise ValueError("confirmation retry FLOPs and prior horizon must appear together")
        if self.retry_of_realized_flops is None:
            expected_steps = prelaunch_confirmation_steps_v2(
                target_flops=self.target_flops,
                arm_mean_flops=self.arm_mean_flops,
                byte_matched_optimizer_steps=self.byte_matched_optimizer_steps,
            )
        else:
            assert self.retry_of_optimizer_steps is not None
            expected_steps = confirmation_retry_steps_v2(
                target_flops=self.target_flops,
                realized_flops=self.retry_of_realized_flops,
                optimizer_steps=self.retry_of_optimizer_steps,
            )
        if self.training_plan.optimizer_steps != expected_steps:
            raise ValueError("confirmation training plan differs from its exact FLOP horizon")
        if self.training_plan.optimizer_steps < CONFIRMATION_BURST_STEPS_V2:
            raise ValueError("confirmation plan is too short for literal calibration")
        if self.heldout_evaluation_steps is not None:
            if (
                not isinstance(self.heldout_evaluation_steps, tuple)
                or len(self.heldout_evaluation_steps) != 3
                or tuple(sorted(set(self.heldout_evaluation_steps)))
                != self.heldout_evaluation_steps
                or self.heldout_evaluation_steps[-1]
                != self.training_plan.optimizer_steps
            ):
                raise ValueError(
                    "confirmation byte checkpoints are not distinct and terminal"
                )

    @property
    def common_flop_budget(self) -> int:
        """Compatibility spelling for the downstream v2 contract."""

        return self.target_flops

    @property
    def base_flop_evidence_sha256(self) -> str:
        """Compatibility spelling; the join is now the arm-level S2 plan."""

        return self.arm_flop_plan_sha256

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_confirmation_execution_plan", self)


def _attempt_projection_a100_microseconds_v2(
    projection: ArmCalibrationProjectionV2,
    execution_plan: ConfirmationExecutionPlanV2,
) -> int:
    """Price one exact confirmation horizon from the inherited base rate."""

    if projection.scope != "confirmation" or projection.vocab_size != execution_plan.vocab_size:
        raise ValueError("confirmation execution plan differs from its inherited rate source")
    return _attempt_projection_for_token_slots_v2(
        projection,
        execution_plan.training_plan.compute_token_slots,
    )


def _attempt_projection_for_token_slots_v2(
    projection: ArmCalibrationProjectionV2,
    planned_compute_token_slots: int,
) -> int:
    if type(planned_compute_token_slots) is not int or planned_compute_token_slots < 1:
        raise ValueError("confirmation attempt projection requires positive token slots")
    training = (
        projection.measured_a100_microseconds
        * planned_compute_token_slots
        + projection.measured_tokens
        - 1
    ) // projection.measured_tokens
    return training + (
        projection.measured_heldout_evaluation_a100_microseconds
        * projection.heldout_evaluations_per_full_run
    ) + (
        projection.measured_output_surface_a100_microseconds
        * projection.output_surface_benchmarks_per_full_run
    )


@dataclass(frozen=True)
class ConfirmationPhysicalMeasurementV2:
    run: ComputeConfirmationRunV2
    flop_ledger: CompleteFlopLedgerV2
    execution_plan_sha256: str
    base_flop_evidence_sha256: str
    training_plan_sha256: str
    heldout_evaluation_steps: tuple[int, int, int]
    burst_flop_receipt: ConfirmationBurstFlopReceiptV2
    physical_flop_ledger_sha256: str
    physical_optimizer_steps: int
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.flop_ledger, CompleteFlopLedgerV2):
            raise TypeError("confirmation measurement requires its complete FLOP ledger")
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
        if not isinstance(self.burst_flop_receipt, ConfirmationBurstFlopReceiptV2):
            raise TypeError("physical confirmation requires its in-run FLOP burst receipt")
        expected_physical_flop_sha256 = (
            confirmation_physical_flop_ledger_evidence_sha256_v2(
                compute_attempt_id=self.run.compute_attempt_id,
                execution_plan_sha256=self.execution_plan_sha256,
                flop_ledger=self.flop_ledger,
            )
        )
        if (
            self.physical_flop_ledger_sha256 != expected_physical_flop_sha256
            or self.run.measured_flops != self.flop_ledger.measured_flops
            or self.physical_optimizer_steps != self.flop_ledger.optimizer_steps
            or self.run.trained_tokens != self.flop_ledger.compute_token_slots
        ):
            raise ValueError("confirmation run differs from its complete physical FLOP ledger")
        if not confirmation_flops_within_target_v2(
            realized_flops=self.run.measured_flops,
            target_flops=self.run.common_flop_budget,
        ):
            raise ValueError("physical confirmation missed its one-percent FLOP band")
        if self.training_runtime_receipt_sha256 != self.run.training_runtime_receipt_sha256:
            raise ValueError("physical confirmation runtime join drifted")
        if self.code_closure_receipt_sha256 != self.run.code_closure_receipt_sha256:
            raise ValueError("physical confirmation code join drifted")
        if self.checkpoint_retained is not False:
            raise ValueError("physical confirmation may not retain checkpoints")


@dataclass(frozen=True)
class ComputeConfirmationCampaignResultV2:
    selection: GTokSelectionReceiptV2
    budget: ConfirmationBudgetReceiptV2
    preflight: PreflightProjectionReceiptV2
    compute: CampaignComputeReceiptV2
    runs: tuple[ComputeConfirmationRunV2, ...]
    evidence_closure: ConfirmationEvidenceClosureV2
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
    budget_receipt_sha256: str
    preflight_receipt_sha256: str
    compute_receipt_sha256: str
    run_receipt_sha256s: tuple[str, ...]
    authority_status: str = "NON_AUTHORITATIVE_INJECTED_CONFIRMATION_EXECUTORS"
    binding_sha256: str = CONFIRMATION_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        values = (
            self.selection_receipt_sha256,
            self.budget_receipt_sha256,
            self.preflight_receipt_sha256,
            self.compute_receipt_sha256,
            *self.run_receipt_sha256s,
        )
        if not self.run_receipt_sha256s or any(
            len(value) != 64 or any(character not in _HEX for character in value)
            for value in values
        ):
            raise ValueError("dry-run evidence identities must be lowercase SHA-256")


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


def build_confirmation_budget_v2(
    *,
    matrix: ValidatedGTokMatrixV2,
    selection: GTokSelectionReceiptV2,
    base_flop_evidence: tuple[BaseRunFlopEvidenceV2, ...],
) -> ConfirmationBudgetReceiptV2:
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
    by_key = {(row.vocab_size, row.seed): row for row in selected}
    arm_inputs: list[
        tuple[int, tuple[BaseRunFlopEvidenceV2, BaseRunFlopEvidenceV2], int, int]
    ] = []
    for vocab_size in sorted(selection.compute_confirmation_pair):
        seed_rows = tuple(by_key[(vocab_size, seed)] for seed in matrix.seeds)
        if len(seed_rows) != 2:
            raise GTokConfirmationV2Error("confirmation arm requires two base seed rows")
        step_counts = {len(row.steps) for row in seed_rows}
        if len(step_counts) != 1:
            raise GTokV2Stop("byte-matched optimizer steps split by seed; return to strategy")
        mean_flops = floor_arm_mean_flops_v2(
            seed_rows[0].measured_flops,
            seed_rows[1].measured_flops,
        )
        arm_inputs.append(
            (vocab_size, (seed_rows[0], seed_rows[1]), next(iter(step_counts)), mean_flops)
        )
    target = min(row[3] for row in arm_inputs)
    rows = tuple(
        ConfirmationArmFlopPlanV2(
            vocab_size=vocab_size,
            seeds=(matrix.seeds[0], matrix.seeds[1]),
            base_flops=(seed_rows[0].measured_flops, seed_rows[1].measured_flops),
            base_flop_evidence_sha256s=(
                seed_rows[0].receipt_sha256,
                seed_rows[1].receipt_sha256,
            ),
            byte_matched_optimizer_steps=byte_matched_steps,
            arm_mean_flops=mean_flops,
            target_flops=target,
            planned_optimizer_steps=prelaunch_confirmation_steps_v2(
                target_flops=target,
                arm_mean_flops=mean_flops,
                byte_matched_optimizer_steps=byte_matched_steps,
            ),
        )
        for vocab_size, seed_rows, byte_matched_steps, mean_flops in arm_inputs
    )
    return ConfirmationBudgetReceiptV2(
        matrix_receipt_sha256=matrix.receipt_sha256,
        selection_receipt_sha256=selection.receipt_sha256,
        pair=pair,
        seeds=matrix.seeds,
        target_flops=target,
        rows=rows,
    )


def build_confirmation_prefix_plan_v2(
    *,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    tokenizer: Tokenizer,
    optimizer_steps: int,
) -> TrainingPlanV2:
    """Precompute the exact ``n``-step stream prefix before optimizer step 1."""

    if optimizer_steps < CONFIRMATION_BURST_STEPS_V2:
        raise GTokConfirmationV2Error("confirmation prefix step count is invalid")
    digest = hashlib.sha256()
    burst_digest = hashlib.sha256()
    slots = predictions = raw_bytes = documents = 0
    burst_slots = burst_predictions = burst_raw_bytes = burst_documents = 0
    observed = 0
    for step, batch in enumerate(
        iter_packed_global_batches_v2(document_factory(), tokenizer=tokenizer), start=1
    ):
        _update_training_plan_digest_v2(digest, batch)
        slots += batch.input_ids.numel()
        predictions += batch.valid_prediction_count
        raw_bytes += batch.completed_raw_bytes
        documents += batch.completed_document_count
        if step <= CONFIRMATION_BURST_STEPS_V2:
            _update_training_plan_digest_v2(burst_digest, batch)
            burst_slots += batch.input_ids.numel()
            burst_predictions += batch.valid_prediction_count
            burst_raw_bytes += batch.completed_raw_bytes
            burst_documents += batch.completed_document_count
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
        calibration_prefix_steps=CONFIRMATION_BURST_STEPS_V2,
        calibration_prefix_compute_token_slots=burst_slots,
        calibration_prefix_valid_prediction_count=burst_predictions,
        calibration_prefix_realized_raw_bytes=burst_raw_bytes,
        calibration_prefix_document_count=burst_documents,
        calibration_prefix_packed_stream_sha256=burst_digest.hexdigest(),
    )


def precompute_byte_checkpoint_steps_v2(
    cumulative_consumed_bytes: tuple[int, ...],
) -> tuple[int, int, int]:
    """Return S1-L6's first 25%, first 50%, and terminal step indices.

    The caller must supply exact cumulative *consumed* stream bytes from Q2's
    token-boundary accounting, not completed-document bytes.  Keeping that
    interface explicit prevents the old document-completion proxy from being
    silently mistaken for the governed quantity.
    """

    if len(cumulative_consumed_bytes) < CONFIRMATION_BURST_STEPS_V2 or any(
        type(value) is not int or value < 0 for value in cumulative_consumed_bytes
    ):
        raise ValueError("byte checkpoint planning requires exact cumulative bytes")
    if any(
        later < earlier
        for earlier, later in zip(
            cumulative_consumed_bytes,
            cumulative_consumed_bytes[1:],
        )
    ):
        raise ValueError("cumulative consumed bytes must be monotone")
    total = cumulative_consumed_bytes[-1]
    if total < 1:
        raise ValueError("byte checkpoint planning requires positive B_total")
    quarter = next(
        step
        for step, value in enumerate(cumulative_consumed_bytes, start=1)
        if 4 * value >= total
    )
    half = next(
        step
        for step, value in enumerate(cumulative_consumed_bytes, start=1)
        if 2 * value >= total
    )
    result = (quarter, half, len(cumulative_consumed_bytes))
    if tuple(sorted(set(result))) != result:
        raise GTokV2Stop("byte-fraction checkpoints are not three distinct steps")
    return result


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
        evaluation_steps = execution_plan.heldout_evaluation_steps
        if evaluation_steps is None:
            raise GTokV2Stop(
                "Q2 cumulative-consumed-byte checkpoints are not bound pre-launch"
            )
        observations: list[BpbMilestoneReceiptV2] = []
        burst_flop_receipt: ConfirmationBurstFlopReceiptV2 | None = None
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
            previous_raw_bytes = raw_bytes
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
            if step == CONFIRMATION_BURST_STEPS_V2:
                burst_gate_evidence = ConfirmationBurstGateEvidenceV2(
                    ordered_step_flops=accountant.ordered_step_flops,
                    prelaunch_arm_mean_flops=execution_plan.arm_mean_flops,
                    byte_matched_optimizer_steps=(
                        execution_plan.byte_matched_optimizer_steps
                    ),
                )
                if burst_gate_evidence.status != "GREEN_100_STEP_BURST":
                    raise ConfirmationBurstGateViolationV2(burst_gate_evidence)
                burst_flop_receipt = ConfirmationBurstFlopReceiptV2(
                    ordered_step_flops=burst_gate_evidence.ordered_step_flops,
                    prelaunch_arm_mean_flops=(
                        burst_gate_evidence.prelaunch_arm_mean_flops
                    ),
                    byte_matched_optimizer_steps=(
                        burst_gate_evidence.byte_matched_optimizer_steps
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
                label = {
                    evaluation_steps[0]: "after_1b",
                    evaluation_steps[1]: "after_2b",
                    evaluation_steps[2]: "terminal_realized_T",
                }[step]
                observations.append(
                    BpbMilestoneReceiptV2(
                        label=label,
                        optimizer_step=step,
                        previous_training_raw_bytes=previous_raw_bytes,
                        training_raw_bytes=raw_bytes,
                        heldout_stream_sha256=corpus_heldout_stream_sha256,
                        strata=strata,
                    )
                )
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
            or raw_bytes != plan.trained_bytes
            or documents != plan.trained_docs_full
            or digest.hexdigest() != plan.packed_stream_sha256
            or len(observations) != CONFIRMATION_HELDOUT_EVALUATIONS_V2
            or burst_flop_receipt is None
        ):
            raise GTokTrainingV2Error("confirmation execution differs from its physical prefix plan")
        ledger = accountant.finalize(plan)
        if not confirmation_flops_within_target_v2(
            realized_flops=ledger.measured_flops,
            target_flops=execution_plan.target_flops,
        ):
            raise ConfirmationFlopBandViolationV2(
                realized_flops=ledger.measured_flops,
                target_flops=execution_plan.target_flops,
                retry_steps=confirmation_retry_steps_v2(
                    target_flops=execution_plan.target_flops,
                    realized_flops=ledger.measured_flops,
                    optimizer_steps=plan.optimizer_steps,
                ),
                flop_ledger=ledger,
                burst_flop_receipt=burst_flop_receipt,
            )
        elapsed = checked_elapsed()
        run = ComputeConfirmationRunV2(
            vocab_size=execution_plan.vocab_size,
            seed_slot=execution_plan.seed_slot,
            registry_key=execution_plan.registry_key,
            seed=execution_plan.seed,
            initialization_seed=execution_plan.initialization_seed,
            data_order_seed=execution_plan.data_order_seed,
            data_order_sha256=execution_plan.data_order_sha256,
            confirmation_order_receipt_sha256=(
                execution_plan.confirmation_order_receipt_sha256
            ),
            physical_d6_evidence_sha256=(
                execution_plan.physical_d6_evidence_sha256
            ),
            document_multiset_sha256=execution_plan.document_multiset_sha256,
            framed_payload_sha256=execution_plan.framed_payload_sha256,
            execution_plan_sha256=execution_plan.receipt_sha256,
            training_plan_sha256=plan.receipt_sha256,
            base_run_receipt_sha256=base_run_receipt_sha256,
            compute_attempt_id=compute_attempt_id,
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=ledger.measured_flops,
            heldout_stream_sha256=corpus_heldout_stream_sha256,
            observations=tuple(observations),
            measured_a100_microseconds=elapsed,
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
            stream_bytes=plan.stream_bytes,
            stream_docs=plan.stream_docs,
            stream_tokens=plan.stream_tokens,
            trained_tokens=plan.trained_tokens,
            dropped_tokens=plan.dropped_tokens,
            trained_bytes=plan.trained_bytes,
            dropped_bytes=plan.dropped_bytes,
            trained_docs_full=plan.trained_docs_full,
            boundary_doc_id=plan.boundary_doc_id,
            boundary_doc_consumed_tokens=plan.boundary_doc_consumed_tokens,
            dropped_docs=plan.dropped_docs,
            gpu_uuid_provenance=gpu_uuid_provenance,
        )
        return ConfirmationPhysicalMeasurementV2(
            run=run,
            flop_ledger=ledger,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=plan.receipt_sha256,
            heldout_evaluation_steps=evaluation_steps,
            burst_flop_receipt=burst_flop_receipt,
            physical_flop_ledger_sha256=(
                confirmation_physical_flop_ledger_evidence_sha256_v2(
                    compute_attempt_id=compute_attempt_id,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    flop_ledger=ledger,
                )
            ),
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


def _confirmation_attempt_launch_path_v2(root: Path, attempt_id: str) -> Path:
    return root / "attempt-launches" / f"{attempt_id}.json"


def _confirmation_attempt_launch_payload_v2(
    *,
    attempt_id: str,
    logical_attempt_id: str,
    execution_plan: ConfirmationExecutionPlanV2,
    projection: ArmCalibrationProjectionV2,
    projected_run_a100_microseconds: int,
) -> Mapping[str, Any]:
    return {
        "attempt_id": attempt_id,
        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
        "calibration_projection_sha256": projection.receipt_sha256,
        "execution_plan_sha256": execution_plan.receipt_sha256,
        "logical_attempt_id": logical_attempt_id,
        "planned_compute_token_slots": (
            execution_plan.training_plan.compute_token_slots
        ),
        "projected_run_a100_microseconds": projected_run_a100_microseconds,
        "schema": "weft1_gtok_v2_confirmation_attempt_launch",
        "seed": execution_plan.seed,
        "vocab_size": execution_plan.vocab_size,
        "watchdog_limit_a100_microseconds": (
            GTOK_PER_RUN_WATCHDOG_MULTIPLIER
            * projected_run_a100_microseconds
        ),
    }


def _load_confirmation_attempt_launch_v2(
    root: Path,
    attempt_id: str,
) -> ConfirmationAttemptLaunchEnvelopeV2:
    path = _confirmation_attempt_launch_path_v2(root, attempt_id)
    try:
        raw, payload = load_canonical_json_snapshot(path)
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise GTokConfirmationV2Error(
            f"confirmation attempt lacks its durable launch binding: {attempt_id}"
        ) from error
    if not isinstance(payload, Mapping):
        raise GTokConfirmationV2Error("confirmation attempt-launch payload is invalid")
    return ConfirmationAttemptLaunchEnvelopeV2(
        payload=payload,
        physical_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _complete_flop_ledger_from_mapping_v2(raw_ledger: Mapping[str, Any]) -> CompleteFlopLedgerV2:
    shapes = []
    for raw_shape in raw_ledger["shapes"]:
        shape = dict(raw_shape)
        shape["profiler_rows"] = tuple(
            ProfilerOperatorFlopRowV2(**row) for row in shape["profiler_rows"]
        )
        shape["unsupported_rows"] = tuple(
            AnalyticUnsupportedFlopRowV2(**row) for row in shape["unsupported_rows"]
        )
        shape["zero_flop_profiler_operators"] = tuple(
            shape["zero_flop_profiler_operators"]
        )
        shapes.append(PhysicalShapeFlopReceiptV2(**shape))
    payload = dict(raw_ledger)
    payload["shapes"] = tuple(shapes)
    return CompleteFlopLedgerV2(**payload)


@dataclass(frozen=True)
class _LoadedRetryArtifactV2:
    ordinal: int
    attempt: ComputeAttemptReceiptV2
    retry_plan: ConfirmationExecutionPlanV2
    retry_join: ConfirmationRetryEvidenceJoinV2
    artifact_envelope: ConfirmationRetryArtifactEnvelopeV2


def _load_retry_chain_v2(
    root: Path,
    initial_plan: ConfirmationExecutionPlanV2,
    projection: ArmCalibrationProjectionV2 | None = None,
) -> tuple[tuple[_LoadedRetryArtifactV2, ...], ConfirmationExecutionPlanV2]:
    logical_attempt_id = _confirmation_attempt_id(
        "run", initial_plan.vocab_size, initial_plan.seed
    )
    lifecycle = (
        validate_lifecycle_ledger_v2(root)
        if (root / "campaign-lifecycle.sqlite3").exists()
        else ()
    )
    terminal_events = _terminal_events_for_logical_v2(
        lifecycle,
        logical_attempt_id=logical_attempt_id,
    )
    failed_events = {
        event.attempt_id: event
        for event in terminal_events
        if event.terminal_status == "failed"
    }
    candidates: list[_LoadedRetryArtifactV2] = []
    for path in sorted(root.glob("invalid-flop-band-*.json")):
        try:
            raw, stored = load_canonical_json_snapshot(path)
            if not isinstance(stored, Mapping):
                raise TypeError("retry artifact is not a mapping")
            if (
                stored.get("schema")
                != "weft1_gtok_v2_invalid_confirmation_flop_band"
                or stored.get("binding_sha256") != CONFIRMATION_BINDING_SHA256_V2
            ):
                raise ValueError("retry artifact authority drifted")
            raw_attempt = stored.get("attempt")
            raw_plan = stored.get("retry_execution_plan")
            raw_ledger = stored.get("invalid_physical_flop_ledger")
            raw_burst = stored.get("passed_burst_flop_receipt")
            if (
                not isinstance(raw_attempt, Mapping)
                or not isinstance(raw_plan, Mapping)
                or not isinstance(raw_ledger, Mapping)
                or not isinstance(raw_burst, Mapping)
            ):
                raise TypeError(
                    "retry artifact omits its attempt, plan, burst, or FLOP ledger"
                )
            if (
                raw_attempt.get("vocab_size") != initial_plan.vocab_size
                or raw_attempt.get("seed") != initial_plan.seed
            ):
                continue
            attempt = ComputeAttemptReceiptV2(**raw_attempt)
            raw_training = raw_plan.get("training_plan")
            if not isinstance(raw_training, Mapping):
                raise TypeError("retry execution plan omits its training plan")
            training_plan = ConfirmationTrainingPlanV2(
                **{
                    key: value
                    for key, value in raw_training.items()
                    if key != "bpb_checkpoint_steps"
                },
                bpb_checkpoint_steps=tuple(raw_training["bpb_checkpoint_steps"]),
            )
            retry_plan = ConfirmationExecutionPlanV2(
                **{
                    key: value
                    for key, value in raw_plan.items()
                    if key not in ("training_plan", "heldout_evaluation_steps")
                },
                training_plan=training_plan,
                heldout_evaluation_steps=tuple(raw_plan["heldout_evaluation_steps"]),
            )
            ledger = _complete_flop_ledger_from_mapping_v2(raw_ledger)
            burst = ConfirmationBurstFlopReceiptV2(
                ordered_step_flops=tuple(raw_burst["ordered_step_flops"]),
                prelaunch_arm_mean_flops=int(
                    raw_burst["prelaunch_arm_mean_flops"]
                ),
                byte_matched_optimizer_steps=int(
                    raw_burst["byte_matched_optimizer_steps"]
                ),
            )
            attempt_id = attempt.attempt_id
            if attempt_id == logical_attempt_id:
                physical_suffix = 0
            elif attempt_id.startswith(f"{logical_attempt_id}.retry-"):
                physical_suffix = int(attempt_id.rsplit(".retry-", 1)[1])
            else:
                raise ValueError("retry artifact attempt left its logical slot")
            if physical_suffix < 0:
                raise ValueError("retry artifact physical-attempt suffix is invalid")
            ordinal = int(stored["correction_ordinal"])
            if ordinal < 0:
                raise ValueError("retry artifact correction ordinal is invalid")
            failed_event = failed_events.get(attempt_id)
            if failed_event is None:
                raise ValueError("retry artifact lacks its failed lifecycle terminal")
            retry_join = ConfirmationRetryEvidenceJoinV2(
                vocab_size=initial_plan.vocab_size,
                seed_slot=initial_plan.seed_slot,
                correction_ordinal=ordinal,
                failed_attempt_id=attempt_id,
                failed_attempt_receipt_sha256=attempt.receipt_sha256,
                failed_execution_plan_sha256=str(
                    stored["failed_execution_plan_sha256"]
                ),
                failed_terminal_lifecycle_event_sha256=(
                    failed_event.receipt_sha256
                ),
                invalid_physical_flop_ledger_sha256=(
                    confirmation_physical_flop_ledger_evidence_sha256_v2(
                        compute_attempt_id=attempt_id,
                        execution_plan_sha256=str(
                            stored["failed_execution_plan_sha256"]
                        ),
                        flop_ledger=ledger,
                    )
                ),
                realized_flops=int(stored["realized_flops"]),
                target_flops=int(stored["target_flops"]),
                prior_optimizer_steps=int(stored["failed_optimizer_steps"]),
                retry_optimizer_steps=int(stored["retry_steps"]),
                retry_execution_plan_sha256=retry_plan.receipt_sha256,
                retry_artifact_physical_sha256=hashlib.sha256(raw).hexdigest(),
            )
            if (
                attempt.status != "failed"
                or stored.get("attempt_receipt_sha256") != attempt.receipt_sha256
                or stored.get("failed_projected_run_a100_microseconds")
                != attempt.projected_run_a100_microseconds
                or stored.get("failed_terminal_lifecycle_event_sha256")
                != failed_event.receipt_sha256
                or stored.get("invalid_physical_flop_ledger_sha256")
                != retry_join.invalid_physical_flop_ledger_sha256
                or stored.get("invalid_flop_ledger_receipt_sha256")
                != ledger.receipt_sha256
                or stored.get("passed_burst_receipt_sha256")
                != burst.receipt_sha256
                or stored.get("passed_physical_burst_evidence_sha256")
                != confirmation_physical_burst_evidence_sha256_v2(
                    compute_attempt_id=attempt_id,
                    execution_plan_sha256=str(
                        stored["failed_execution_plan_sha256"]
                    ),
                    burst=burst,
                )
                or retry_join.realized_flops != ledger.measured_flops
                or ledger.optimizer_steps != retry_join.prior_optimizer_steps
                or ledger.compute_token_slots
                != attempt.planned_compute_token_slots
                or retry_plan.retry_of_realized_flops
                != retry_join.realized_flops
                or retry_plan.retry_of_optimizer_steps
                != retry_join.prior_optimizer_steps
                or retry_plan.training_plan.optimizer_steps
                != retry_join.retry_optimizer_steps
                or stored.get("retry_execution_plan_sha256")
                != retry_plan.receipt_sha256
            ):
                raise ValueError("retry artifact evidence joins drifted")
            if projection is not None and (
                projection.vocab_size != initial_plan.vocab_size
                or stored.get("retry_projected_run_a100_microseconds")
                != _attempt_projection_a100_microseconds_v2(
                    projection,
                    retry_plan,
                )
            ):
                raise ValueError("retry artifact projected cost differs on replay")
            candidates.append(
                _LoadedRetryArtifactV2(
                    ordinal=ordinal,
                    attempt=attempt,
                    retry_plan=retry_plan,
                    retry_join=retry_join,
                    artifact_envelope=ConfirmationRetryArtifactEnvelopeV2(
                        payload=stored,
                        physical_sha256=hashlib.sha256(raw).hexdigest(),
                    ),
                )
            )
        except (KeyError, OSError, StrictJsonError, TypeError, ValueError) as error:
            raise GTokConfirmationV2Error(
                f"stored confirmation retry artifact is invalid: {path.name}"
            ) from error
    ordered = tuple(sorted(candidates, key=lambda row: row.ordinal))
    if {row.attempt.attempt_id for row in ordered} != set(failed_events):
        raise GTokConfirmationV2Error(
            "failed confirmation terminals and invalid-band artifacts differ"
        )
    if tuple(row.ordinal for row in ordered) != tuple(range(len(ordered))):
        raise GTokConfirmationV2Error("confirmation retry chain is not contiguous")
    stable_fields = (
        "vocab_size",
        "seed_slot",
        "registry_key",
        "seed",
        "initialization_seed",
        "data_order_seed",
        "data_order_sha256",
        "confirmation_order_receipt_sha256",
        "physical_d6_evidence_sha256",
        "document_multiset_sha256",
        "framed_payload_sha256",
        "order_document_count",
        "order_retained_text_bytes",
        "target_flops",
        "arm_mean_flops",
        "byte_matched_optimizer_steps",
        "arm_flop_plan_sha256",
    )
    current = initial_plan
    for row in ordered:
        persisted_burst = row.artifact_envelope.payload.get(
            "passed_burst_flop_receipt"
        )
        if (
            not isinstance(persisted_burst, Mapping)
            or row.attempt.execution_plan_sha256 != current.receipt_sha256
            or row.retry_join.failed_execution_plan_sha256 != current.receipt_sha256
            or row.retry_join.prior_optimizer_steps
            != current.training_plan.optimizer_steps
            or persisted_burst.get("prelaunch_arm_mean_flops")
            != current.arm_mean_flops
            or persisted_burst.get("byte_matched_optimizer_steps")
            != current.byte_matched_optimizer_steps
            or any(
                getattr(row.retry_plan, name) != getattr(initial_plan, name)
                for name in stable_fields
            )
        ):
            raise GTokConfirmationV2Error("confirmation retry chain changed stable evidence")
        current = row.retry_plan
    completed = tuple(
        event for event in terminal_events if event.terminal_status == "completed"
    )
    if completed:
        measurement = _confirmation_measurement_from_lifecycle_v2(completed[0])
        if (
            measurement.execution_plan_sha256 != current.receipt_sha256
        ):
            raise GTokConfirmationV2Error(
                "completed confirmation differs from its recovered retry chain"
            )
    return ordered, current


def _resume_retry_execution_plan_v2(
    root: Path,
    initial_plan: ConfirmationExecutionPlanV2,
    projection: ArmCalibrationProjectionV2,
) -> ConfirmationExecutionPlanV2:
    """Recover the latest durable corrected horizon for one logical slot."""

    _chain, plan = _load_retry_chain_v2(root, initial_plan, projection)
    return plan


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
    raw_burst = payload.get("burst_flop_receipt")
    raw_ledger = payload.get("flop_ledger")
    if (
        not isinstance(raw_run, Mapping)
        or not isinstance(raw_burst, Mapping)
        or not isinstance(raw_ledger, Mapping)
    ):
        raise GTokConfirmationV2Error("completed confirmation omits its run receipt")
    raw_observations = raw_run.get("observations")
    if not isinstance(raw_observations, (list, tuple)):
        raise GTokConfirmationV2Error("completed confirmation omits BPB observations")
    try:
        observations = []
        for raw_observation in raw_observations:
            if not isinstance(raw_observation, Mapping):
                raise TypeError("confirmation observation is not a mapping")
            raw_strata = raw_observation.get("strata")
            if not isinstance(raw_strata, (list, tuple)):
                raise TypeError("confirmation observation omits strata")
            observations.append(
                BpbMilestoneReceiptV2(
                    **{
                        key: value
                        for key, value in raw_observation.items()
                        if key != "strata"
                    },
                    strata=tuple(StratumNllReceipt(**row) for row in raw_strata),
                )
            )
        run = ComputeConfirmationRunV2(
            **{
                key: value
                for key, value in raw_run.items()
                if key not in ("observations", "measured_a100_microseconds")
            },
            observations=tuple(observations),
            measured_a100_microseconds=event.charged_a100_microseconds,
        )
        ledger = _complete_flop_ledger_from_mapping_v2(raw_ledger)
        return ConfirmationPhysicalMeasurementV2(
            **{
                key: value
                for key, value in payload.items()
                if key not in (
                    "run",
                    "flop_ledger",
                    "heldout_evaluation_steps",
                    "burst_flop_receipt",
                )
            },
            run=run,
            flop_ledger=ledger,
            heldout_evaluation_steps=tuple(payload["heldout_evaluation_steps"]),
            burst_flop_receipt=ConfirmationBurstFlopReceiptV2(
                ordered_step_flops=tuple(raw_burst["ordered_step_flops"]),
                prelaunch_arm_mean_flops=raw_burst["prelaunch_arm_mean_flops"],
                byte_matched_optimizer_steps=raw_burst[
                    "byte_matched_optimizer_steps"
                ],
            ),
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
    execution_plan: ConfirmationExecutionPlanV2,
    known_execution_plans: Mapping[str, ConfirmationExecutionPlanV2],
) -> None:
    existing = {row.attempt_id: row for row in attempts}
    for event in terminal_events:
        status = event.terminal_status
        if status not in ("completed", "failed", "preempted", "aborted_watchdog"):
            raise GTokConfirmationV2Error("terminal lifecycle status is unregistered")
        launch_envelope = _load_confirmation_attempt_launch_v2(root, event.attempt_id)
        launch = launch_envelope.payload
        launch_plan = known_execution_plans.get(str(launch["execution_plan_sha256"]))
        if launch_plan is None:
            raise GTokConfirmationV2Error(
                "attempt launch names an unknown execution plan"
            )
        launch_slots = launch_plan.training_plan.compute_token_slots
        launch_projected = _attempt_projection_for_token_slots_v2(
            projection,
            launch_slots,
        )
        if (
            launch["attempt_id"] != event.attempt_id
            or launch["logical_attempt_id"] != event.logical_attempt_id
            or launch["calibration_projection_sha256"] != projection.receipt_sha256
            or launch["planned_compute_token_slots"] != launch_slots
            or launch["projected_run_a100_microseconds"] != launch_projected
            or launch["watchdog_limit_a100_microseconds"]
            != GTOK_PER_RUN_WATCHDOG_MULTIPLIER * launch_projected
            or launch["vocab_size"] != launch_plan.vocab_size
            or launch["seed"] != launch_plan.seed
        ):
            raise GTokConfirmationV2Error(
                "attempt lifecycle differs from its durable launch binding"
            )
        prior = existing.get(event.attempt_id)
        if prior is not None:
            assert prior.planned_compute_token_slots is not None
            prior_plan = known_execution_plans.get(str(prior.execution_plan_sha256))
            if prior_plan is None:
                raise GTokConfirmationV2Error(
                    "persisted attempt names an unknown execution plan"
                )
            expected_projected_run = _attempt_projection_for_token_slots_v2(
                projection,
                prior_plan.training_plan.compute_token_slots,
            )
            if (
                prior.scope != "confirmation"
                or prior.kind != "full_run"
                or prior.vocab_size != launch_plan.vocab_size
                or prior.seed != launch_plan.seed
                or prior.status != status
                or prior.consumed_a100_microseconds
                != event.charged_a100_microseconds
                or prior.calibration_projection_sha256 != projection.receipt_sha256
                or prior.execution_plan_sha256 != launch_plan.receipt_sha256
                or prior.planned_compute_token_slots
                != launch_slots
                or prior.projected_run_a100_microseconds != expected_projected_run
                or prior.watchdog_limit_a100_microseconds
                != GTOK_PER_RUN_WATCHDOG_MULTIPLIER * expected_projected_run
            ):
                raise GTokConfirmationV2Error(
                    "persisted attempt differs from its lifecycle terminal"
                )
            if status == "completed":
                measurement = _confirmation_measurement_from_lifecycle_v2(event)
                if (
                    prior.execution_plan_sha256
                    != measurement.execution_plan_sha256
                    or prior.planned_compute_token_slots
                    != measurement.run.trained_tokens
                ):
                    raise GTokConfirmationV2Error(
                        "completed attempt differs from its priced execution plan"
                    )
            continue
        if status == "completed":
            measurement = _confirmation_measurement_from_lifecycle_v2(event)
            execution_plan_sha256 = measurement.execution_plan_sha256
            planned_slots = measurement.run.trained_tokens
            if (
                execution_plan_sha256 != launch_plan.receipt_sha256
                or planned_slots != launch_slots
            ):
                raise GTokConfirmationV2Error(
                    "completed attempt differs from its durable launch binding"
                )
        else:
            execution_plan_sha256 = launch_plan.receipt_sha256
            planned_slots = launch_slots
        projected_run = _attempt_projection_for_token_slots_v2(
            projection,
            planned_slots,
        )
        watchdog = 2 * projected_run
        exceeded = event.charged_a100_microseconds > watchdog
        if exceeded:
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
            kind="full_run",
            vocab_size=launch_plan.vocab_size,
            seed=launch_plan.seed,
            consumed_a100_microseconds=event.charged_a100_microseconds,
            status=status,
            calibration_projection_sha256=projection.receipt_sha256,
            projected_run_a100_microseconds=projected_run,
            watchdog_limit_a100_microseconds=watchdog,
            execution_plan_sha256=execution_plan_sha256,
            planned_compute_token_slots=planned_slots,
            hard_abort_issued=hard_abort,
        )
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
    budget = build_confirmation_budget_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=base_flop_evidence,
    )
    _write_receipt_v2(
        root / "confirmation-flop-budget.json",
        "weft1_gtok_v2_confirmation_budget",
        budget,
    )

    tokenizers = {row.receipt.vocab_size: row.load() for row in tokenizer_arms}
    budget_by_vocab = {row.vocab_size: row for row in budget.rows}
    fresh_vocab_size = budget.fresh_vocab_size
    arm_budget = budget_by_vocab[fresh_vocab_size]
    confirmation_seed_rows = confirmation_seed_rows_for_vocabulary_v2(fresh_vocab_size)
    confirmation_orders: dict[int, ConfirmationConsumerOrderV4] = {}
    confirmation_order_physical_sha256s: dict[int, str] = {}
    plans: list[ConfirmationExecutionPlanV2] = []
    for seed_row in confirmation_seed_rows:
        order_receipt = build_materialized_confirmation_order_v4(
            source.root,
            confirmation_run_seed=seed_row.run_seed,
            data_order_seed=seed_row.data_order_seed,
            expected_physical_d6_evidence_sha256=(
                source.physical_d6_evidence_sha256
            ),
        )
        confirmation_orders[seed_row.run_seed] = order_receipt
        confirmation_order_physical_sha256s[seed_row.run_seed] = _write_or_validate_v2(
            root / f"confirmation-order-v{fresh_vocab_size}-slot{seed_row.seed_slot}.json",
            {
                "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                "payload": asdict(order_receipt),
                "receipt_sha256": order_receipt.receipt_sha256,
                "schema": "weft1_gtok_confirmation_consumer_order_v4",
            },
        )
        plan = plan_confirmation_training_prefix_v2(
            lambda order_receipt=order_receipt: source.confirmation_training_documents(
                order_receipt
            ),
            tokenizer=tokenizers[fresh_vocab_size],
            optimizer_steps=arm_budget.planned_optimizer_steps,
            confirmation_order_receipt=order_receipt,
        )
        plans.append(
            ConfirmationExecutionPlanV2(
                vocab_size=fresh_vocab_size,
                seed_slot=seed_row.seed_slot,
                registry_key=seed_row.registry_key,
                seed=seed_row.run_seed,
                initialization_seed=seed_row.initialization_seed,
                data_order_seed=seed_row.data_order_seed,
                data_order_sha256=order_receipt.ordered_raw_content_ids_sha256,
                confirmation_order_receipt_sha256=order_receipt.receipt_sha256,
                physical_d6_evidence_sha256=(
                    order_receipt.physical_d6_evidence_sha256
                ),
                document_multiset_sha256=order_receipt.document_multiset_sha256,
                framed_payload_sha256=order_receipt.framed_payload_sha256,
                order_document_count=order_receipt.document_count,
                order_retained_text_bytes=order_receipt.retained_text_bytes,
                target_flops=budget.target_flops,
                arm_mean_flops=arm_budget.arm_mean_flops,
                byte_matched_optimizer_steps=arm_budget.byte_matched_optimizer_steps,
                arm_flop_plan_sha256=arm_budget.receipt_sha256,
                training_plan=plan,
                heldout_evaluation_steps=plan.bpb_checkpoint_steps,
            )
        )
    plans_tuple = tuple(plans)
    initial_plans_tuple = plans_tuple
    for plan in plans_tuple:
        order_receipt = confirmation_orders[plan.seed]
        _write_or_validate_v2(
            root
            / f"confirmation-execution-plan-v{plan.vocab_size}-slot{plan.seed_slot}.json",
            {
                "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                "confirmation_order": asdict(order_receipt),
                "confirmation_order_receipt_sha256": order_receipt.receipt_sha256,
                "execution_plan": asdict(plan),
                "execution_plan_sha256": plan.receipt_sha256,
                "schema": "weft1_gtok_v2_confirmation_execution_plan_envelope",
                "training_plan_sha256": plan.training_plan.receipt_sha256,
            },
        )
    if len(
        {
            (row.training_plan.optimizer_steps, row.training_plan.compute_token_slots)
            for row in plans_tuple
        }
    ) != 1:
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
    cumulative = matrix.compute.consumed_a100_microseconds + sum(
        row.consumed_a100_microseconds for row in attempts
    )
    if any(row.kind == "calibration" for row in attempts):
        raise GTokConfirmationV2Error(
            "S2 confirmation may not reuse a pre-S2 standalone calibration ledger"
        )
    base_projection_by_vocab = {
        row.vocab_size: row for row in matrix.compute.preflight.calibrations
    }
    projections: list[ArmCalibrationProjectionV2] = []
    for vocab_size in (fresh_vocab_size,):
        plan = plans_tuple[0].training_plan
        try:
            base_projection = base_projection_by_vocab[vocab_size]
        except KeyError as error:
            raise GTokConfirmationV2Error(
                "confirmation arm lacks completed base calibration evidence"
            ) from error
        inherited_measured_tokens = (
            base_projection.measured_tokens
            if authoritative
            else plan.compute_token_slots
        )
        inherited_measured_time = (
            base_projection.measured_a100_microseconds
            if authoritative
            else base_projection.projected_run_a100_microseconds
        )
        training_projection = (
            inherited_measured_time * plan.compute_token_slots
            + inherited_measured_tokens
            - 1
        ) // inherited_measured_tokens
        projected_run = training_projection + (
            base_projection.measured_heldout_evaluation_a100_microseconds
            * base_projection.heldout_evaluations_per_full_run
        ) + (
            base_projection.measured_output_surface_a100_microseconds
            * base_projection.output_surface_benchmarks_per_full_run
        )
        projections.append(
            ArmCalibrationProjectionV2(
                scope="confirmation",
                vocab_size=vocab_size,
                calibration_attempt_id=(
                    f"inherited-base-calibration-v{vocab_size}"
                ),
                calibration_steps=base_projection.calibration_steps,
                measured_tokens=inherited_measured_tokens,
                measured_a100_microseconds=inherited_measured_time,
                planned_tokens_per_run=plan.compute_token_slots,
                projected_run_a100_microseconds=projected_run,
                charged_calibration_a100_microseconds=0,
                measured_heldout_evaluation_a100_microseconds=(
                    base_projection.measured_heldout_evaluation_a100_microseconds
                ),
                heldout_evaluations_per_full_run=(
                    base_projection.heldout_evaluations_per_full_run
                ),
                measured_output_surface_a100_microseconds=(
                    base_projection.measured_output_surface_a100_microseconds
                ),
                output_surface_benchmarks_per_full_run=(
                    base_projection.output_surface_benchmarks_per_full_run
                ),
                projection_source="completed_base_calibration",
                projection_source_receipt_sha256=base_projection.receipt_sha256,
            )
        )

    try:
        preflight = build_preflight_projection_v2(
            tuple(projections),
            prior_campaign_a100_microseconds=matrix.compute.consumed_a100_microseconds,
            prior_event_ledger_sha256=matrix.compute.event_ledger_sha256,
            scope="confirmation",
            recovered_attempt_a100_microseconds=0,
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
    lifecycle = (
        validate_lifecycle_ledger_v2(root)
        if (root / "campaign-lifecycle.sqlite3").exists()
        else ()
    )
    recovered_retry_states = tuple(
        _load_retry_chain_v2(
            root,
            plan,
            projection_by_vocab[plan.vocab_size],
        )
        for plan in plans_tuple
    )
    persisted_attempts_by_id = {row.attempt_id: row for row in attempts}
    for chain, _recovered_plan in recovered_retry_states:
        for artifact in chain:
            persisted = persisted_attempts_by_id.get(artifact.attempt.attempt_id)
            if (
                persisted is None
                or canonical_json_bytes(persisted)
                != canonical_json_bytes(artifact.attempt)
            ):
                raise GTokConfirmationV2Error(
                    "retry artifact differs from its durable compute-attempt receipt"
                )
    resumed_plans_tuple = tuple(state[1] for state in recovered_retry_states)
    correction_ordinal_by_slot = {
        plan.seed_slot: len(state[0])
        for plan, state in zip(plans_tuple, recovered_retry_states, strict=True)
    }
    known_execution_plans_by_slot: dict[
        int, dict[str, ConfirmationExecutionPlanV2]
    ] = {}
    for initial_plan, (retry_chain, _recovered_plan) in zip(
        plans_tuple,
        recovered_retry_states,
        strict=True,
    ):
        known = {initial_plan.receipt_sha256: initial_plan}
        known.update(
            {row.retry_plan.receipt_sha256: row.retry_plan for row in retry_chain}
        )
        known_execution_plans_by_slot[initial_plan.seed_slot] = known
    completed_logical_attempts: set[str] = set()
    for execution_plan in resumed_plans_tuple:
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
            execution_plan=execution_plan,
            known_execution_plans=(
                known_execution_plans_by_slot[execution_plan.seed_slot]
            ),
        )
        if any(row.terminal_status == "completed" for row in terminal_events):
            completed_logical_attempts.add(logical_attempt_id)
    cumulative = matrix.compute.consumed_a100_microseconds + sum(
        row.consumed_a100_microseconds for row in attempts
    )
    pending = tuple(
        _confirmation_attempt_id("run", row.vocab_size, row.seed)
        for row in resumed_plans_tuple
        if _confirmation_attempt_id("run", row.vocab_size, row.seed)
        not in completed_logical_attempts
    )
    remaining_projection = sum(
        _attempt_projection_a100_microseconds_v2(
            projection_by_vocab[row.vocab_size],
            row,
        )
        for row in resumed_plans_tuple
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
    execution_queue = list(resumed_plans_tuple)
    successful_plans: dict[int, ConfirmationExecutionPlanV2] = {}
    while execution_queue:
        execution_plan = execution_queue.pop(0)
        revalidate_code_closure()
        projection = projection_by_vocab[execution_plan.vocab_size]
        attempt_projection = _attempt_projection_a100_microseconds_v2(
            projection,
            execution_plan,
        )
        logical_attempt_id = _confirmation_attempt_id(
            "run", execution_plan.vocab_size, execution_plan.seed
        )
        pending = tuple(row for row in pending if row != logical_attempt_id)
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
            measurement = _confirmation_measurement_from_lifecycle_v2(
                completed_event
            )
            attempt_id = completed_event.attempt_id
        else:
            attempt_id = _next_physical_attempt_id_v2(
                logical_attempt_id,
                lifecycle,
            )
            queued_projection = attempt_projection
            queued_logical_ids = [logical_attempt_id]
            for queued_plan in execution_queue:
                queued_logical_id = _confirmation_attempt_id(
                    "run", queued_plan.vocab_size, queued_plan.seed
                )
                queued_terminals = _terminal_events_for_logical_v2(
                    lifecycle,
                    logical_attempt_id=queued_logical_id,
                )
                if any(row.terminal_status == "completed" for row in queued_terminals):
                    continue
                queued_projection += _attempt_projection_a100_microseconds_v2(
                    projection_by_vocab[queued_plan.vocab_size],
                    queued_plan,
                )
                queued_logical_ids.append(queued_logical_id)
            if cumulative + queued_projection > GTOK_TRIPWIRE_A100_MICROSECONDS:
                _write_stop(
                    root,
                    reason="PRESTART_CONFIRMATION_QUEUE_EXCEEDS_12_A100_HOURS",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=tuple(queued_logical_ids),
                    running=(),
                )
                raise GTokV2Stop(
                    "confirmation queue leaves no governed budget before START"
                )
            launch_payload = _confirmation_attempt_launch_payload_v2(
                attempt_id=attempt_id,
                logical_attempt_id=logical_attempt_id,
                execution_plan=execution_plan,
                projection=projection,
                projected_run_a100_microseconds=attempt_projection,
            )
            _write_or_validate_v2(
                _confirmation_attempt_launch_path_v2(root, attempt_id),
                launch_payload,
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
                            (
                                execution_plan.vocab_size,
                                matrix.seeds[execution_plan.seed_slot],
                            )
                        ].receipt_sha256,
                        compute_attempt_id=attempt_id,
                        watchdog_limit_a100_microseconds=(
                            GTOK_PER_RUN_WATCHDOG_MULTIPLIER
                            * attempt_projection
                        ),
                        prior_campaign_a100_microseconds=cumulative,
                        gpu_uuid_provenance=gpu_uuid_provenance,
                        document_factory=lambda execution_plan=execution_plan: source.confirmation_training_documents(
                            confirmation_orders[execution_plan.seed]
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
            except ConfirmationBurstGateViolationV2 as error:
                consumed = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                watchdog = 2 * attempt_projection
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="full_run",
                    vocab_size=execution_plan.vocab_size,
                    seed=execution_plan.seed,
                    consumed_a100_microseconds=consumed,
                    status="failed",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=attempt_projection,
                    watchdog_limit_a100_microseconds=watchdog,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    planned_compute_token_slots=(
                        execution_plan.training_plan.compute_token_slots
                    ),
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += consumed
                failed_terminal = next(
                    row
                    for row in validate_lifecycle_ledger_v2(root)
                    if row.attempt_id == attempt_id
                    and row.phase == "TERMINAL"
                    and row.terminal_status == "failed"
                )
                physical_burst_gate_sha256 = gtok_v2_bound_sha256(
                    "weft1_gtok_v2_confirmation_physical_burst_gate_evidence",
                    {
                        "burst_gate_receipt_sha256": error.evidence.receipt_sha256,
                        "compute_attempt_id": attempt_id,
                        "execution_plan_sha256": execution_plan.receipt_sha256,
                    },
                )
                burst_stop_physical_sha256 = _write_or_validate_v2(
                    root / f"invalid-confirmation-burst-{attempt_id}.json",
                    {
                        "attempt": asdict(attempt),
                        "attempt_receipt_sha256": attempt.receipt_sha256,
                        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                        "burst_gate_evidence": asdict(error.evidence),
                        "burst_gate_receipt_sha256": error.evidence.receipt_sha256,
                        "failed_execution_plan_sha256": (
                            execution_plan.receipt_sha256
                        ),
                        "failed_terminal_lifecycle_event_sha256": (
                            failed_terminal.receipt_sha256
                        ),
                        "physical_burst_gate_evidence_sha256": (
                            physical_burst_gate_sha256
                        ),
                        "schema": "weft1_gtok_v2_invalid_confirmation_burst_gate",
                        "status": error.evidence.status,
                    },
                )
                _write_stop(
                    root,
                    reason=error.evidence.status,
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(),
                    decision_evidence_receipt_sha256=(
                        error.evidence.receipt_sha256
                    ),
                    decision_evidence_physical_sha256=(
                        burst_stop_physical_sha256
                    ),
                )
                raise GTokV2Stop(
                    "confirmation Q4 burst gate fired; return to strategy"
                ) from error
            except ConfirmationFlopBandViolationV2 as error:
                consumed = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                watchdog = 2 * attempt_projection
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="confirmation",
                    kind="full_run",
                    vocab_size=execution_plan.vocab_size,
                    seed=execution_plan.seed,
                    consumed_a100_microseconds=consumed,
                    status="failed",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=attempt_projection,
                    watchdog_limit_a100_microseconds=watchdog,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    planned_compute_token_slots=(
                        execution_plan.training_plan.compute_token_slots
                    ),
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += consumed
                retry_training_plan = plan_confirmation_training_prefix_v2(
                    lambda execution_plan=execution_plan: source.confirmation_training_documents(
                        confirmation_orders[execution_plan.seed]
                    ),
                    tokenizer=tokenizers[execution_plan.vocab_size],
                    optimizer_steps=error.retry_steps,
                    confirmation_order_receipt=confirmation_orders[execution_plan.seed],
                )
                retry_plan = replace(
                    execution_plan,
                    training_plan=retry_training_plan,
                    retry_of_realized_flops=error.realized_flops,
                    retry_of_optimizer_steps=(
                        execution_plan.training_plan.optimizer_steps
                    ),
                    heldout_evaluation_steps=(
                        retry_training_plan.bpb_checkpoint_steps
                    ),
                )
                known_execution_plans_by_slot[execution_plan.seed_slot][
                    retry_plan.receipt_sha256
                ] = retry_plan
                retry_projected_run = _attempt_projection_a100_microseconds_v2(
                    projection,
                    retry_plan,
                )
                failed_terminal = next(
                    row
                    for row in validate_lifecycle_ledger_v2(root)
                    if row.attempt_id == attempt_id
                    and row.phase == "TERMINAL"
                    and row.terminal_status == "failed"
                )
                invalid_physical_flop_ledger_sha256 = (
                    confirmation_physical_flop_ledger_evidence_sha256_v2(
                        compute_attempt_id=attempt_id,
                        execution_plan_sha256=execution_plan.receipt_sha256,
                        flop_ledger=error.flop_ledger,
                    )
                )
                passed_physical_burst_evidence_sha256 = (
                    confirmation_physical_burst_evidence_sha256_v2(
                        compute_attempt_id=attempt_id,
                        execution_plan_sha256=execution_plan.receipt_sha256,
                        burst=error.burst_flop_receipt,
                    )
                )
                correction_ordinal = correction_ordinal_by_slot[
                    execution_plan.seed_slot
                ]
                _write_or_validate_v2(
                    root / f"invalid-flop-band-{attempt_id}.json",
                    {
                        "attempt": asdict(attempt),
                        "attempt_receipt_sha256": attempt.receipt_sha256,
                        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                        "correction_ordinal": correction_ordinal,
                        "failed_execution_plan_sha256": (
                            execution_plan.receipt_sha256
                        ),
                        "failed_optimizer_steps": (
                            execution_plan.training_plan.optimizer_steps
                        ),
                        "failed_projected_run_a100_microseconds": (
                            attempt_projection
                        ),
                        "failed_terminal_lifecycle_event_sha256": (
                            failed_terminal.receipt_sha256
                        ),
                        "invalid_physical_flop_ledger": asdict(error.flop_ledger),
                        "invalid_physical_flop_ledger_sha256": (
                            invalid_physical_flop_ledger_sha256
                        ),
                        "invalid_flop_ledger_receipt_sha256": (
                            error.flop_ledger.receipt_sha256
                        ),
                        "passed_burst_flop_receipt": asdict(
                            error.burst_flop_receipt
                        ),
                        "passed_burst_receipt_sha256": (
                            error.burst_flop_receipt.receipt_sha256
                        ),
                        "passed_physical_burst_evidence_sha256": (
                            passed_physical_burst_evidence_sha256
                        ),
                        "realized_flops": error.realized_flops,
                        "retry_execution_plan": asdict(retry_plan),
                        "retry_execution_plan_sha256": retry_plan.receipt_sha256,
                        "retry_projected_run_a100_microseconds": (
                            retry_projected_run
                        ),
                        "retry_steps": error.retry_steps,
                        "schema": "weft1_gtok_v2_invalid_confirmation_flop_band",
                        "target_flops": error.target_flops,
                    },
                )
                other_pending_projection = sum(
                    _attempt_projection_a100_microseconds_v2(
                        projection_by_vocab[row.vocab_size],
                        row,
                    )
                    for row in execution_queue
                    if not any(
                        event.terminal_status == "completed"
                        for event in _terminal_events_for_logical_v2(
                            validate_lifecycle_ledger_v2(root),
                            logical_attempt_id=_confirmation_attempt_id(
                                "run", row.vocab_size, row.seed
                            ),
                        )
                    )
                )
                if (
                    cumulative + retry_projected_run + other_pending_projection
                    > GTOK_TRIPWIRE_A100_MICROSECONDS
                ):
                    _write_stop(
                        root,
                        reason="INVALID_FLOP_BAND_RETRY_EXCEEDS_12_A100_HOURS",
                        cumulative=cumulative,
                        attempts=tuple(attempts),
                        pending=(logical_attempt_id,),
                        running=(),
                    )
                    raise GTokV2Stop(
                        "invalid confirmation run left no governed retry budget"
                    ) from error
                pending = tuple(dict.fromkeys((logical_attempt_id, *pending)))
                correction_ordinal_by_slot[execution_plan.seed_slot] += 1
                execution_queue.insert(0, retry_plan)
                continue
            except (GTokRunWatchdogV2, GTokCampaignTripwireV2) as error:
                consumed = int(getattr(error, "_gtok_lifecycle_charge_v2", 1))
                watchdog = 2 * attempt_projection
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
                    projected_run_a100_microseconds=attempt_projection,
                    watchdog_limit_a100_microseconds=watchdog,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    planned_compute_token_slots=(
                        execution_plan.training_plan.compute_token_slots
                    ),
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
                watchdog = 2 * attempt_projection
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
                    projected_run_a100_microseconds=attempt_projection,
                    watchdog_limit_a100_microseconds=watchdog,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    planned_compute_token_slots=(
                        execution_plan.training_plan.compute_token_slots
                    ),
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
        watchdog = 2 * attempt_projection
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
                projected_run_a100_microseconds=attempt_projection,
                watchdog_limit_a100_microseconds=watchdog,
                execution_plan_sha256=execution_plan.receipt_sha256,
                planned_compute_token_slots=(
                    execution_plan.training_plan.compute_token_slots
                ),
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
            or run.execution_plan_sha256 != execution_plan.receipt_sha256
            or measurement.training_plan_sha256
            != execution_plan.training_plan.receipt_sha256
            or run.training_plan_sha256
            != execution_plan.training_plan.receipt_sha256
            or measurement.heldout_evaluation_steps
            != execution_plan.heldout_evaluation_steps
            or measurement.base_flop_evidence_sha256
            != execution_plan.base_flop_evidence_sha256
            or run.confirmation_order_receipt_sha256
            != execution_plan.confirmation_order_receipt_sha256
            or run.physical_d6_evidence_sha256
            != execution_plan.physical_d6_evidence_sha256
            or run.document_multiset_sha256
            != execution_plan.document_multiset_sha256
            or run.data_order_sha256 != execution_plan.data_order_sha256
            or run.framed_payload_sha256 != execution_plan.framed_payload_sha256
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
                projected_run_a100_microseconds=attempt_projection,
                watchdog_limit_a100_microseconds=watchdog,
                execution_plan_sha256=execution_plan.receipt_sha256,
                planned_compute_token_slots=(
                    execution_plan.training_plan.compute_token_slots
                ),
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
            execution_plan=execution_plan,
            known_execution_plans=(
                known_execution_plans_by_slot[execution_plan.seed_slot]
            ),
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
        successful_plans[execution_plan.seed_slot] = execution_plan

    plans_tuple = tuple(successful_plans[slot] for slot in (0, 1))

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
            budget_receipt_sha256=budget.receipt_sha256,
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
    fresh_joins: list[ConfirmationFreshEvidenceJoinV2] = []
    for run in sorted(runs_tuple, key=lambda row: row.seed_slot):
        plan = plans_tuple[run.seed_slot]
        order_receipt = confirmation_orders[run.seed]
        terminal_event = next(
            event
            for event in final_lifecycle
            if event.attempt_id == run.compute_attempt_id
            and event.phase == "TERMINAL"
            and event.terminal_status == "completed"
        )
        measurement = _confirmation_measurement_from_lifecycle_v2(terminal_event)
        fresh_joins.append(
            ConfirmationFreshEvidenceJoinV2(
                vocab_size=run.vocab_size,
                seed_slot=run.seed_slot,
                fresh_run_receipt_sha256=run.receipt_sha256,
                confirmation_order_receipt_sha256=order_receipt.receipt_sha256,
                physical_d6_evidence_sha256=(
                    order_receipt.physical_d6_evidence_sha256
                ),
                document_multiset_sha256=order_receipt.document_multiset_sha256,
                ordered_raw_content_ids_sha256=(
                    order_receipt.ordered_raw_content_ids_sha256
                ),
                framed_payload_sha256=order_receipt.framed_payload_sha256,
                order_document_count=order_receipt.document_count,
                order_retained_text_bytes=order_receipt.retained_text_bytes,
                execution_plan_sha256=plan.receipt_sha256,
                training_plan_sha256=plan.training_plan.receipt_sha256,
                compute_attempt_id=run.compute_attempt_id,
                terminal_lifecycle_event_sha256=terminal_event.receipt_sha256,
                burst_receipt_sha256=(
                    confirmation_physical_burst_evidence_sha256_v2(
                        compute_attempt_id=run.compute_attempt_id,
                        execution_plan_sha256=plan.receipt_sha256,
                        burst=measurement.burst_flop_receipt,
                    )
                ),
                physical_flop_ledger_sha256=(
                    measurement.physical_flop_ledger_sha256
                ),
            )
        )
    retry_joins: list[ConfirmationRetryEvidenceJoinV2] = []
    retry_artifacts: list[ConfirmationRetryArtifactEnvelopeV2] = []
    execution_plan_envelopes: list[ConfirmationExecutionPlanEnvelopeV2] = []
    for initial_plan in initial_plans_tuple:
        execution_plan_envelopes.append(
            ConfirmationExecutionPlanEnvelopeV2(
                payload=asdict(initial_plan),
                receipt_sha256=initial_plan.receipt_sha256,
            )
        )
        retry_chain, recovered_plan = _load_retry_chain_v2(
            root,
            initial_plan,
            projection_by_vocab[initial_plan.vocab_size],
        )
        if recovered_plan.receipt_sha256 != plans_tuple[initial_plan.seed_slot].receipt_sha256:
            raise GTokConfirmationV2Error(
                "successful confirmation plan differs from its durable retry chain"
            )
        retry_joins.extend(row.retry_join for row in retry_chain)
        retry_artifacts.extend(row.artifact_envelope for row in retry_chain)
        execution_plan_envelopes.extend(
            ConfirmationExecutionPlanEnvelopeV2(
                payload=asdict(row.retry_plan),
                receipt_sha256=row.retry_plan.receipt_sha256,
            )
            for row in retry_chain
        )
    base_measurement_by_key = {
        (row.run.vocab_size, row.run.seed): row for row in base.measurements
    }
    base_flop_evidence_by_key = {
        (row.vocab_size, row.seed): row for row in base_flop_evidence
    }
    base_flop_sources: list[ConfirmationArmFlopSourceEnvelopeV2] = []
    for arm_plan in budget.rows:
        source_rows: list[ConfirmationBaseRunFlopSourceEnvelopeV2] = []
        for seed in matrix.seeds:
            key = (arm_plan.vocab_size, seed)
            try:
                measurement = base_measurement_by_key[key]
                evidence = base_flop_evidence_by_key[key]
            except KeyError as error:
                raise GTokConfirmationV2Error(
                    "confirmation decision lacks a raw base FLOP source"
                ) from error
            if (
                measurement.run.receipt_sha256
                != evidence.base_run_receipt_sha256
                or measurement.flop_ledger.receipt_sha256
                != evidence.flop_ledger_sha256
            ):
                raise GTokConfirmationV2Error(
                    "base measurement and reconstructed FLOP evidence disagree"
                )
            source_rows.append(
                ConfirmationBaseRunFlopSourceEnvelopeV2(
                    flop_ledger_payload=asdict(measurement.flop_ledger),
                    flop_ledger_receipt_sha256=(
                        measurement.flop_ledger.receipt_sha256
                    ),
                    base_flop_evidence_payload=asdict(evidence),
                    base_flop_evidence_receipt_sha256=evidence.receipt_sha256,
                )
            )
        if len(source_rows) != 2:
            raise GTokConfirmationV2Error(
                "confirmation arm lacks two base FLOP sources"
            )
        base_flop_sources.append(
            ConfirmationArmFlopSourceEnvelopeV2(
                arm_plan_payload=asdict(arm_plan),
                arm_plan_receipt_sha256=arm_plan.receipt_sha256,
                base_runs=(source_rows[0], source_rows[1]),
            )
        )
    lifecycle_preimages = tuple(
        ConfirmationLifecycleEventEvidenceV2(**asdict(event))
        for event in final_lifecycle
    )
    if tuple(row.receipt_sha256 for row in lifecycle_preimages) != tuple(
        row.receipt_sha256 for row in final_lifecycle
    ):
        raise GTokConfirmationV2Error(
            "confirmation lifecycle preimages changed their physical identities"
        )
    evidence_closure = ConfirmationEvidenceClosureV2(
        compute_event_ledger_sha256=compute.event_ledger_sha256,
        lifecycle_ledger_sha256=gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_lifecycle_ledger",
            lifecycle_preimages,
        ),
        lifecycle_events=lifecycle_preimages,
        execution_plans=tuple(execution_plan_envelopes),
        confirmation_orders=tuple(
            ConfirmationOrderEnvelopeV2(
                payload=asdict(confirmation_orders[plan.seed]),
                receipt_sha256=confirmation_orders[plan.seed].receipt_sha256,
                physical_sha256=confirmation_order_physical_sha256s[plan.seed],
            )
            for plan in plans_tuple
        ),
        attempt_launches=tuple(
            _load_confirmation_attempt_launch_v2(root, attempt.attempt_id)
            for attempt in attempt_tuple
        ),
        base_flop_sources=tuple(base_flop_sources),
        fresh_joins=tuple(fresh_joins),
        retry_joins=tuple(retry_joins),
        retry_artifacts=tuple(retry_artifacts),
    )
    confirmation = validate_compute_confirmation_v2(
        runs_tuple,
        matrix=matrix,
        selection=selection,
        compute=compute,
        evidence_closure=evidence_closure,
    )
    pair_base_runs = tuple(
        row for row in matrix.runs if row.vocab_size in selection.compute_confirmation_pair
    )
    decision_physical_sha256 = _write_or_validate_v2(
        root / "confirmation-decision.json",
        {
            "base_pair_runs": tuple(asdict(row) for row in pair_base_runs),
            "base_pair_run_receipt_sha256s": tuple(
                row.receipt_sha256 for row in pair_base_runs
            ),
            "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
            "confirmation": asdict(confirmation),
            "confirmation_receipt_sha256": confirmation.receipt_sha256,
            "evidence_closure": asdict(evidence_closure),
            "evidence_closure_sha256": evidence_closure.receipt_sha256,
            "successful_execution_plans": tuple(asdict(row) for row in plans_tuple),
            "successful_execution_plan_sha256s": tuple(
                row.receipt_sha256 for row in plans_tuple
            ),
            "confirmation_orders": tuple(
                asdict(confirmation_orders[row.seed]) for row in plans_tuple
            ),
            "confirmation_order_receipt_sha256s": tuple(
                confirmation_orders[row.seed].receipt_sha256 for row in plans_tuple
            ),
            "fresh_runs": tuple(asdict(row) for row in runs_tuple),
            "fresh_run_receipt_sha256s": tuple(
                row.receipt_sha256 for row in runs_tuple
            ),
            "schema": "weft1_gtok_v2_compute_confirmation_decision",
        },
    )
    if confirmation.status != "GREEN_NO_REVERSAL":
        _write_stop(
            root,
            reason=confirmation.status,
            cumulative=cumulative,
            attempts=attempt_tuple,
            pending=(),
            running=(),
            decision_evidence_receipt_sha256=confirmation.receipt_sha256,
            decision_evidence_physical_sha256=decision_physical_sha256,
        )
        raise GTokV2Stop(
            "compute-matched confirmation requires strategy escalation"
        )
    basis = _basis_from_selection_v2(matrix, selection)
    freeze = mint_vocabulary_freeze_v2(
        matrix=matrix,
        selection=selection,
        confirmation=confirmation,
        basis=basis,
    )
    result = ComputeConfirmationCampaignResultV2(
        selection=selection,
        budget=budget,
        preflight=preflight,
        compute=compute,
        runs=runs_tuple,
        evidence_closure=evidence_closure,
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
            "confirmation_decision_physical_sha256": decision_physical_sha256,
            "evidence_closure": asdict(evidence_closure),
            "evidence_closure_sha256": evidence_closure.receipt_sha256,
            "successful_execution_plans": tuple(asdict(row) for row in plans_tuple),
            "successful_execution_plan_sha256s": tuple(
                row.receipt_sha256 for row in plans_tuple
            ),
            "confirmation_orders": tuple(
                asdict(confirmation_orders[row.seed]) for row in plans_tuple
            ),
            "confirmation_order_receipt_sha256s": tuple(
                confirmation_orders[row.seed].receipt_sha256 for row in plans_tuple
            ),
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
    "CONFIRMATION_BURST_STEPS_V2",
    "ComputeConfirmationCampaignResultV2",
    "ConfirmationArmFlopPlanV2",
    "ConfirmationBudgetReceiptV2",
    "ConfirmationBurstGateEvidenceV2",
    "ConfirmationBurstGateViolationV2",
    "ConfirmationBurstFlopReceiptV2",
    "ConfirmationExecutionPlanV2",
    "ConfirmationFlopBandViolationV2",
    "ConfirmationPhysicalMeasurementV2",
    "DryRunComputeConfirmationV2",
    "GTokConfirmationV2Error",
    "RUNG_B_ANCHOR_PARAMETER_COUNT_V2",
    "RUNG_B_ANCHOR_VOCAB_SIZE_V2",
    "RUNG_B_MODEL_WIDTH_V2",
    "build_confirmation_prefix_plan_v2",
    "build_confirmation_budget_v2",
    "build_rung_b_admissibility_v2",
    "load_base_run_flop_evidence_v2",
    "confirmation_flops_within_target_v2",
    "confirmation_physical_burst_evidence_sha256_v2",
    "confirmation_physical_flop_ledger_evidence_sha256_v2",
    "confirmation_retry_steps_v2",
    "floor_arm_mean_flops_v2",
    "precompute_byte_checkpoint_steps_v2",
    "prelaunch_confirmation_steps_v2",
    "run_compute_confirmation_and_freeze_v2",
    "run_compute_confirmation_dry_run_v2",
]
