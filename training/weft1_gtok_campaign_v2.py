"""Forward-only base-campaign orchestrator for the WEFT-1 G-TOK screen.

The orchestrator is deliberately sequential.  That makes the cumulative
A100-time meter, pending/running set, and hard-abort surface literal rather than
an eventually-consistent scheduler claim.  Every calibration and full-run
attempt is written as one exclusive event before a later attempt may start.
Model state is never serialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import AbstractContextManager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

import torch
from tokenizers import Tokenizer

from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_materialize_a3 import GTOK_TRAINING_SEEDS
from training.weft1_gtok_contract import (
    FlatAdamWRecipe,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
)
from training.weft1_gtok_training_v2 import (
    AnalyticUnsupportedFlopRowV2,
    ArmMeasurementPanelV2,
    CalibrationMeasurementV2,
    CompleteFlopLedgerV2,
    FullRunMeasurementV2,
    GTokCampaignTripwireV2,
    GTokRunWatchdogV2,
    INITIALIZATION_RECIPE_SHA256_V2,
    OutputSurfacePerformanceV2,
    PhysicalShapeFlopReceiptV2,
    ProfilerOperatorFlopRowV2,
    StratumCompressionMetricsV2,
    TokenizerCorpusMetricsV2,
    TrainingDocumentV2,
    TrainingPlanV2,
    V4CorpusSourceV2,
    VocabularyFractionRowV2,
    build_gtok_proxy_model_v2,
    calibrate_arm_v2,
    execute_full_run_v2,
    measure_tokenizer_corpus_metrics_v2,
    plan_training_stream_v2,
    shared_nonvocabulary_state_sha256_v2,
    validate_calibration_prefix_v2,
)
from training.weft1_gtok_code_closure_v2 import (
    GTokCodeClosureReceiptV2,
    validate_gtok_code_closure_v2,
)
from training.weft1_gtok_offline_v2 import (
    GTokOfflineV2Error,
    load_offline_parent_receipt_v2,
)
from training.weft1_gtok_determinism_v2 import (
    CUDA_DETERMINISM_POLICY_SHA256_V2,
    CudaDeterminismAttestationV2,
    DeterminismReplayPairProjectionV2,
    DeterminismReplayPlanBindingV2,
    DeterminismReplayReplicaV2,
    GTokDeterminismV2Stop,
    PrecalibrationDeterminismReplayReceiptV2,
    ReplayBatchShapeV2,
    DeterminismReplayFingerprintV2,
    _mint_precalibration_determinism_replay_receipt_v2,
    apply_and_attest_cuda_determinism_policy_v2,
    canonical_determinism_replay_plan_bindings_v2,
    execute_precalibration_determinism_replay_replica_v2,
    load_precalibration_determinism_replay_receipt_v2,
    project_second_determinism_replay_replica_v2,
    write_precalibration_determinism_replay_receipt_v2,
)
from training.weft1_gtok_v2_contract import (
    ArmCalibrationProjectionV2,
    BpbMilestoneReceiptV2,
    CampaignComputeReceiptV2,
    ComputeAttemptReceiptV2,
    FrozenScreenCorpusV2,
    GTOK_PER_RUN_WATCHDOG_MULTIPLIER,
    GTOK_TRIPWIRE_A100_MICROSECONDS,
    GTokRunReceiptV2,
    GTokV2Stop,
    PrecalibrationReplayAttemptReceiptV2,
    PreflightProjectionReceiptV2,
    RuntimeTripwireSnapshotV2,
    TokenizerArmReceiptV2,
    ValidatedGTokMatrixV2,
    gtok_v2_bound_sha256,
    validate_complete_gtok_matrix_v2,
)
from training.weft1_seed import derive_module_seed
from training.weft1_strict_io import (
    StrictJsonError,
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


GTOK_GOVERNED_SEED_ROWS_V2 = (
    (
        4_069_725_298_476_216_533,
        9_305_630_768_498_788_030,
        10_666_192_988_433_719_740,
    ),
    (
        13_256_058_689_613_801_745,
        12_171_684_496_048_357_438,
        4_197_282_192_878_334_768,
    ),
)
GTOK_MICROBATCH_SEQUENCES_V2 = 8
GTOK_ACCUMULATION_SLICES_V2 = 256 // GTOK_MICROBATCH_SEQUENCES_V2
CONFIRMATION_SEMANTICS_AUTHORITY_STATUS_V2 = (
    "UNRESOLVED_EXACT_FLOP_AND_TOP_TWO_STRATEGY_RULING_REQUIRED"
)
_DERIVED_SEED_ROWS_V2 = tuple(
    (
        training_seed,
        derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.init.shared.{training_seed}",
        ),
        derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{training_seed}",
        ),
    )
    for training_seed in GTOK_TRAINING_SEEDS
)
if _DERIVED_SEED_ROWS_V2 != GTOK_GOVERNED_SEED_ROWS_V2:
    raise RuntimeError("A2 governed G-TOK seed derivation drifted")
GTOK_GOVERNED_TRAINING_SEEDS_V2 = tuple(
    row[0] for row in GTOK_GOVERNED_SEED_ROWS_V2
)
GTOK_GOVERNED_INITIALIZATION_SEEDS_V2 = tuple(
    row[1] for row in GTOK_GOVERNED_SEED_ROWS_V2
)
GTOK_GOVERNED_DATA_ORDER_SEEDS_V2 = tuple(
    row[2] for row in GTOK_GOVERNED_SEED_ROWS_V2
)

CAMPAIGN_BINDING_V2 = {
    "attempt_execution": "sequential_ascending_vocab_then_seed",
    "calibration": "ascending_vocab_before_any_full_run",
    "cpu_precompute": "separate_offline_exact_training_venv_before_A100_allocation",
    "event_storage": "SQLite_BEGIN_IMMEDIATE_append_only_plus_exclusive_JSON_mirror",
    "model_artifacts": "never_serialized",
    "runtime_meter": (
        "precalibration_replay_replicas_plus_calibrations_plus_every_full_attempt_"
        "including_retries_orphans_and_failures"
    ),
    "resume": "attempt_boundary_only_fresh_model_no_model_or_optimizer_state",
    "heartbeat_a100_microseconds": 30_000_000,
    "gpu_uuid": "recorded_per_attempt_provenance_not_runtime_identity",
    "projection": "80_measured_steps_plus_3_H_evals_plus_1_output_surface_panel",
    "microbatch_sequences": GTOK_MICROBATCH_SEQUENCES_V2,
    "gradient_accumulation_slices": GTOK_ACCUMULATION_SLICES_V2,
    "seed_rows": GTOK_GOVERNED_SEED_ROWS_V2,
    "subprocess_launches": "none_campaign_executes_in_one_attested_process",
    "precalibration_determinism_replay": (
        "canonical_one_fresh_pair_per_distinct_vocab_terminal_rows_key"
    ),
    "precalibration_determinism_policy_sha256": (
        CUDA_DETERMINISM_POLICY_SHA256_V2
    ),
}
CAMPAIGN_BINDING_SHA256_V2 = hashlib.sha256(
    canonical_json_bytes(CAMPAIGN_BINDING_V2)
).hexdigest()
_HEX = frozenset("0123456789abcdef")
HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2 = 30_000_000


class GTokCampaignV2Error(RuntimeError):
    """Campaign assembly or durable evidence failed."""


def require_resolved_confirmation_semantics_v2() -> None:
    """Block authoritative P-C spend until the two strategy rulings arrive."""

    raise GTokV2Stop(CONFIRMATION_SEMANTICS_AUTHORITY_STATUS_V2)


@dataclass(frozen=True)
class InitializationArmStateV2:
    vocab_size: int
    shared_nonvocabulary_state_sha256: str

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("initialization evidence uses an unregistered vocabulary")
        if (
            len(self.shared_nonvocabulary_state_sha256) != 64
            or any(
                character not in _HEX
                for character in self.shared_nonvocabulary_state_sha256
            )
        ):
            raise ValueError("initialization evidence requires a state SHA-256")


@dataclass(frozen=True)
class InitializationSeedStateV2:
    training_seed: int
    initialization_seed: int
    arms: tuple[InitializationArmStateV2, ...]

    def __post_init__(self) -> None:
        if tuple(row.vocab_size for row in self.arms) != GTOK_VOCABULARY_ARMS:
            raise ValueError("initialization evidence requires all four arms in order")
        if len({row.shared_nonvocabulary_state_sha256 for row in self.arms}) != 1:
            raise ValueError("non-vocabulary initialization differs across vocabulary arms")


@dataclass(frozen=True)
class InitializationEqualityEvidenceV2:
    rows: tuple[InitializationSeedStateV2, ...]
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    offline_network_receipt_sha256: str
    initialization_recipe_sha256: str = INITIALIZATION_RECIPE_SHA256_V2
    meter_classification: str = "PRE_CALIBRATION_CPU_ONLY_NOT_A2_TRIPWIRE"
    status: str = "PASS_BOTH_GOVERNED_SEEDS_ALL_FOUR_ARMS_EQUAL"

    def __post_init__(self) -> None:
        expected = tuple(
            (training_seed, initialization_seed)
            for training_seed, initialization_seed, _data_seed in GTOK_GOVERNED_SEED_ROWS_V2
        )
        if tuple(
            (row.training_seed, row.initialization_seed) for row in self.rows
        ) != expected:
            raise ValueError("initialization evidence differs from governed seed rows")
        for name in (
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
            "offline_network_receipt_sha256",
            "initialization_recipe_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be SHA-256")
        if self.initialization_recipe_sha256 != INITIALIZATION_RECIPE_SHA256_V2:
            raise ValueError("initialization recipe binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_initialization_equality", self)


def _validate_full_initialization_v2(
    measurement: FullRunMeasurementV2,
    *,
    evidence: InitializationEqualityEvidenceV2,
    vocab_size: int,
    training_seed: int,
) -> None:
    expected_row = next(
        (row for row in evidence.rows if row.training_seed == training_seed),
        None,
    )
    if expected_row is None:
        raise GTokCampaignV2Error(
            "full-run seed is absent from initialization equality evidence"
        )
    expected_by_vocab = {
        arm.vocab_size: arm.shared_nonvocabulary_state_sha256
        for arm in expected_row.arms
    }
    if (
        measurement.run.initialization_seed != expected_row.initialization_seed
        or measurement.run.shared_initial_state_sha256
        != expected_by_vocab.get(vocab_size)
    ):
        raise GTokCampaignV2Error(
            "full-run initialization differs from pre-spend equality evidence"
        )


@dataclass(frozen=True)
class CampaignLifecycleEventV2:
    """Durable START/heartbeat/terminal evidence for one physical attempt."""

    logical_attempt_id: str
    attempt_id: str
    scope: str
    kind: str
    phase: str
    charged_a100_microseconds: int
    terminal_status: str | None
    completion_payload: Mapping[str, Any] | None = None
    gpu_uuid_provenance: str | None = None
    offline_network_launch_receipt_sha256: str | None = None
    heartbeat_interval_a100_microseconds: int = (
        HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2
    )
    checkpoint_retained: bool = False

    def __post_init__(self) -> None:
        for name in ("logical_attempt_id", "attempt_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        if self.scope not in ("base_screen", "confirmation"):
            raise ValueError("lifecycle scope is unregistered")
        if self.kind not in ("determinism_replay", "calibration", "full_run"):
            raise ValueError("lifecycle kind is unregistered")
        if self.phase not in ("START", "HEARTBEAT", "TERMINAL"):
            raise ValueError("lifecycle phase is unregistered")
        if (
            type(self.charged_a100_microseconds) is not int
            or self.charged_a100_microseconds < 1
        ):
            raise ValueError("lifecycle charge must be a positive exact integer")
        if self.phase == "TERMINAL":
            if self.terminal_status not in (
                "completed",
                "failed",
                "preempted",
                "aborted_watchdog",
            ):
                raise ValueError("terminal lifecycle status is unregistered")
            if self.terminal_status == "completed":
                if not isinstance(self.completion_payload, Mapping):
                    raise ValueError("completed lifecycle event requires its evidence payload")
            elif self.completion_payload is not None:
                raise ValueError("non-completed terminal may not carry completion evidence")
        elif self.terminal_status is not None:
            raise ValueError("nonterminal lifecycle event may not claim a status")
        elif self.completion_payload is not None:
            raise ValueError("nonterminal lifecycle event may not carry completion evidence")
        if (
            self.heartbeat_interval_a100_microseconds
            != HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2
        ):
            raise ValueError("heartbeat cadence drifted")
        if self.checkpoint_retained is not False:
            raise ValueError("campaign resume may not retain model or optimizer state")
        if self.gpu_uuid_provenance is not None and (
            not isinstance(self.gpu_uuid_provenance, str)
            or not self.gpu_uuid_provenance.startswith("GPU-")
            or len(self.gpu_uuid_provenance) <= 4
        ):
            raise ValueError("lifecycle GPU provenance must be an NVIDIA GPU UUID")
        if self.offline_network_launch_receipt_sha256 is not None and (
            len(self.offline_network_launch_receipt_sha256) != 64
            or any(
                character not in _HEX
                for character in self.offline_network_launch_receipt_sha256
            )
        ):
            raise ValueError("lifecycle offline launch receipt must be SHA-256")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_lifecycle_event", self)


@dataclass(frozen=True)
class TokenizerExecutionArmV2:
    receipt: TokenizerArmReceiptV2
    tokenizer_json_path: Path
    offline_network_receipt_sha256: str
    offline_network_policy_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.offline_network_receipt_sha256) != 64
            or any(
                character not in _HEX
                for character in self.offline_network_receipt_sha256
            )
        ):
            raise ValueError("tokenizer execution arm requires an offline receipt SHA-256")
        if (
            len(self.offline_network_policy_sha256) != 64
            or any(
                character not in _HEX for character in self.offline_network_policy_sha256
            )
        ):
            raise ValueError("tokenizer execution arm requires an offline policy SHA-256")

    def load(self) -> Tokenizer:
        path = assert_no_symlink_ancestors(self.tokenizer_json_path).resolve(strict=True)
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != self.receipt.tokenizer_json_sha256:
            raise GTokCampaignV2Error("tokenizer artifact differs from its arm receipt")
        tokenizer = Tokenizer.from_str(payload.decode("utf-8", errors="strict"))
        if tokenizer.get_vocab_size(with_added_tokens=True) != self.receipt.vocab_size:
            raise GTokCampaignV2Error("tokenizer artifact vocabulary size drifted")
        return tokenizer


def _revalidate_tokenizer_producer_evidence_v2(
    *,
    artifact_root: Path,
    evidence: Mapping[str, Any],
    receipt: TokenizerArmReceiptV2,
    corpus: FrozenScreenCorpusV2,
    offline_network_receipt_sha256: str,
    offline_network_policy_sha256: str,
    producer_python_executable_sha256: str,
) -> None:
    """Re-open the double-fit producer boundary and rederive local invariants.

    The tokenizer panel is an index, not authority to restate producer facts.
    Every consumer therefore re-opens the parent receipt, both worker receipts,
    and both tokenizer artifacts.  Corpus-scale BPE counter replay remains bound
    by the two matching worker receipts and the frozen T identity; all
    artifact-local properties are independently recomputed here.
    """

    from training.weft1_gtok_tokenizer_a2 import (
        tokenizer_artifact_sha256,
        tokenizer_inventory_sha256,
        tokenizer_merges_sha256,
        validate_tokenizer_json,
    )
    from training.weft1_corpus_pa import DEFAULT_REQUIREMENTS_LOCK_SHA256
    from training.weft1_gtok_tokenizer_v2 import (
        TOKENIZER_FILENAME_V2,
        WORKER_RECEIPT_FILENAME_V2,
        GTokTokenizerV2Error,
        _parse_worker_receipt,
        _pretokenizer_regex_sha256,
        _reserved_inventory_sha256,
        tokenizer_byte_round_trip_receipt_v2,
    )

    try:
        root = assert_no_symlink_ancestors(artifact_root).resolve(strict=True)
        parent_path = root / "tokenizer-arm-receipt.json"
        parent_raw, parent = load_canonical_json_snapshot(parent_path)
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error(
            "tokenizer producer parent receipt is absent or invalid"
        ) from error
    if (
        parent_raw != canonical_json_bytes(parent) + b"\n"
        or canonical_json_bytes(parent) != canonical_json_bytes(evidence)
    ):
        raise GTokCampaignV2Error(
            "tokenizer panel evidence differs from its physical producer receipt"
        )

    worker_rows = []
    artifacts: list[bytes] = []
    for label in ("fit-a", "fit-b"):
        worker_root = root / label
        try:
            resolved_worker_root = assert_no_symlink_ancestors(worker_root).resolve(
                strict=True
            )
            worker = _parse_worker_receipt(
                resolved_worker_root / WORKER_RECEIPT_FILENAME_V2
            )
            recorded_root = assert_no_symlink_ancestors(
                Path(worker.output_root)
            ).resolve(strict=True)
            artifact_path = assert_no_symlink_ancestors(
                resolved_worker_root / TOKENIZER_FILENAME_V2
            ).resolve(strict=True)
            payload = artifact_path.read_bytes()
            validate_tokenizer_json(payload, expected_vocab_size=receipt.vocab_size)
            round_trip = tokenizer_byte_round_trip_receipt_v2(payload)
        except Exception as error:
            raise GTokCampaignV2Error(
                f"tokenizer producer {label} evidence is absent or invalid"
            ) from error
        if recorded_root != resolved_worker_root:
            raise GTokCampaignV2Error(
                f"tokenizer producer {label} output-root evidence drifted"
            )
        derived = {
            "byte_round_trip_receipt_sha256": round_trip["receipt_sha256"],
            "merges_sha256": tokenizer_merges_sha256(payload),
            "pretokenizer_regex_sha256": _pretokenizer_regex_sha256(),
            "reserved_inventory_sha256": _reserved_inventory_sha256(),
            "token_inventory_sha256": tokenizer_inventory_sha256(payload),
            "tokenizer_json_sha256": tokenizer_artifact_sha256(payload),
        }
        if any(getattr(worker, name) != value for name, value in derived.items()):
            raise GTokCampaignV2Error(
                f"tokenizer producer {label} artifact-local evidence drifted"
            )
        worker_rows.append(worker)
        artifacts.append(payload)

    first, second = worker_rows
    double_fit = evidence.get("double_fit")
    if not isinstance(double_fit, Mapping):
        raise GTokCampaignV2Error("tokenizer producer double-fit evidence is absent")
    if (
        double_fit.get("first_worker_receipt_sha256") != first.receipt_sha256
        or double_fit.get("second_worker_receipt_sha256") != second.receipt_sha256
        or double_fit.get("first_process_id") != first.process_id
        or double_fit.get("second_process_id") != second.process_id
        or first.process_id == second.process_id
        or artifacts[0] != artifacts[1]
    ):
        raise GTokCampaignV2Error(
            "tokenizer producer receipts do not prove two matching fresh fits"
        )

    worker_shared_fields = (
        "vocab_size",
        "tokenizer_json_sha256",
        "merges_sha256",
        "token_inventory_sha256",
        "reserved_inventory_sha256",
        "pretokenizer_regex_sha256",
        "fit_stream_sha256",
        "full_corpus_manifest_sha256",
        "screen_submanifest_sha256",
        "physical_d6_evidence_sha256",
        "tokenizer_fit_input_receipt_sha256",
        "bpe_safety_receipt_sha256",
        "byte_round_trip_receipt_sha256",
        "executable_sha256",
        "dependency_lock_sha256",
        "environment_identity_sha256",
        "runtime_attestation_receipt_sha256",
        "offline_network_receipt_sha256",
        "offline_network_policy_sha256",
        "tokenizers_version",
    )
    if any(
        getattr(first, name) != getattr(second, name)
        for name in worker_shared_fields
    ):
        raise GTokCampaignV2Error("tokenizer producer worker evidence drifted")
    expected_worker_fields = {
        "byte_round_trip_receipt_sha256": receipt.byte_round_trip_receipt_sha256,
        "fit_stream_sha256": corpus.training_stream_sha256,
        "full_corpus_manifest_sha256": corpus.full_corpus_manifest_sha256,
        "merges_sha256": receipt.merges_sha256,
        "offline_network_policy_sha256": offline_network_policy_sha256,
        "offline_network_receipt_sha256": offline_network_receipt_sha256,
        "physical_d6_evidence_sha256": corpus.d6_physical_evidence_sha256,
        "pretokenizer_regex_sha256": receipt.pretokenizer_regex_sha256,
        "reserved_inventory_sha256": receipt.reserved_inventory_sha256,
        "screen_submanifest_sha256": corpus.screen_submanifest_sha256,
        "token_inventory_sha256": receipt.token_inventory_sha256,
        "tokenizer_json_sha256": receipt.tokenizer_json_sha256,
        "tokenizers_version": receipt.tokenizer_version,
        "vocab_size": receipt.vocab_size,
    }
    if any(
        getattr(first, name) != value
        for name, value in expected_worker_fields.items()
    ):
        raise GTokCampaignV2Error(
            "tokenizer producer evidence differs from arm, corpus, or launcher"
        )
    expected_runtime_attestation = gtok_v2_bound_sha256(
        "weft1_gtok_v2_tokenizer_runtime_attestation",
        {
            "dependency_lock_sha256": first.dependency_lock_sha256,
            "environment_identity_sha256": first.environment_identity_sha256,
            "executable_sha256": first.executable_sha256,
        },
    )
    if (
        first.dependency_lock_sha256 != DEFAULT_REQUIREMENTS_LOCK_SHA256
        or first.executable_sha256 != producer_python_executable_sha256
        or first.runtime_attestation_receipt_sha256
        != expected_runtime_attestation
    ):
        raise GTokCampaignV2Error(
            "tokenizer producer runtime differs from the authenticated P-A launcher"
        )


def load_tokenizer_execution_panel_v2(
    *,
    panel_receipt_path: Path,
    artifact_root: Path,
    offline_parent_receipt_path: Path,
    corpus: FrozenScreenCorpusV2,
) -> tuple[TokenizerExecutionArmV2, ...]:
    """Load the canonical four-arm panel and re-open every selected artifact."""

    if (
        not isinstance(panel_receipt_path, Path)
        or not isinstance(artifact_root, Path)
        or not isinstance(offline_parent_receipt_path, Path)
    ):
        raise TypeError("tokenizer panel paths must be pathlib.Path")
    if not isinstance(corpus, FrozenScreenCorpusV2):
        raise TypeError("tokenizer panel requires a frozen P-B corpus")
    panel_path = assert_no_symlink_ancestors(panel_receipt_path).resolve(strict=True)
    artifacts = assert_no_symlink_ancestors(artifact_root).resolve(strict=True)
    try:
        tokenizer_offline, tokenizer_offline_physical_sha256 = (
            load_offline_parent_receipt_v2(offline_parent_receipt_path)
        )
        expected_tokenizer_cli = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_weft1_gtok_v2.py"
        ).resolve(strict=True)
        observed_tokenizer_cli = assert_no_symlink_ancestors(
            Path(tokenizer_offline.campaign_script)
        ).resolve(strict=True)
    except (GTokOfflineV2Error, OSError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error(
            "tokenizer offline launch receipt is absent or invalid"
        ) from error
    if (
        observed_tokenizer_cli != expected_tokenizer_cli
        or hashlib.sha256(observed_tokenizer_cli.read_bytes()).hexdigest()
        != tokenizer_offline.campaign_script_sha256
    ):
        raise GTokCampaignV2Error(
            "tokenizer offline launch did not bind the exact production tokenizer CLI"
        )
    try:
        raw, panel = load_canonical_json_snapshot(panel_path)
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error("tokenizer panel is absent or invalid") from error
    if (
        not isinstance(panel, Mapping)
        or raw != canonical_json_bytes(panel) + b"\n"
        or set(panel)
        != {
            "arms",
            "offline_network_receipt_sha256",
            "offline_network_policy_sha256",
            "schema",
            "vocabularies",
        }
        or panel.get("schema") != "weft1_gtok_v2_tokenizer_panel"
        or tuple(panel.get("vocabularies", ())) != GTOK_VOCABULARY_ARMS
    ):
        raise GTokCampaignV2Error("tokenizer panel envelope drifted")
    panel_offline_sha256 = panel.get("offline_network_receipt_sha256")
    if (
        not isinstance(panel_offline_sha256, str)
        or len(panel_offline_sha256) != 64
        or any(character not in _HEX for character in panel_offline_sha256)
    ):
        raise GTokCampaignV2Error("tokenizer panel lacks its offline launch identity")
    if panel_offline_sha256 != tokenizer_offline_physical_sha256:
        raise GTokCampaignV2Error(
            "tokenizer panel offline identity differs from its physical launch receipt"
        )
    panel_offline_policy_sha256 = panel.get("offline_network_policy_sha256")
    if panel_offline_policy_sha256 != tokenizer_offline.policy_sha256:
        raise GTokCampaignV2Error(
            "tokenizer panel offline policy differs from its authenticated launcher"
        )
    rows = panel.get("arms")
    if not isinstance(rows, list) or len(rows) != len(GTOK_VOCABULARY_ARMS):
        raise GTokCampaignV2Error("tokenizer panel arm count drifted")

    from training.weft1_gtok_tokenizer_v2 import DOUBLE_FIT_SCHEMA_V2

    arms: list[TokenizerExecutionArmV2] = []
    for vocab_size, row in zip(GTOK_VOCABULARY_ARMS, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "arm",
            "arm_receipt_sha256",
            "corpus_receipt_sha256",
            "evidence",
            "offline_network_receipt_sha256",
            "offline_network_policy_sha256",
            "output_root",
        }:
            raise GTokCampaignV2Error("tokenizer panel row fields drifted")
        payload = row.get("arm")
        evidence = row.get("evidence")
        if not isinstance(payload, Mapping) or not isinstance(evidence, Mapping):
            raise GTokCampaignV2Error("tokenizer panel arm evidence is absent")
        try:
            receipt = TokenizerArmReceiptV2(**payload)
        except (TypeError, ValueError) as error:
            raise GTokCampaignV2Error("tokenizer arm receipt is invalid") from error
        if receipt.vocab_size != vocab_size or row.get("arm_receipt_sha256") != receipt.receipt_sha256:
            raise GTokCampaignV2Error("tokenizer arm order or receipt identity drifted")
        if row.get("corpus_receipt_sha256") != corpus.receipt_sha256:
            raise GTokCampaignV2Error("tokenizer panel names a different frozen corpus")
        if row.get("offline_network_receipt_sha256") != panel_offline_sha256:
            raise GTokCampaignV2Error("tokenizer panel row offline identity drifted")
        if row.get("offline_network_policy_sha256") != panel_offline_policy_sha256:
            raise GTokCampaignV2Error("tokenizer panel row offline policy drifted")
        if set(evidence) != {
            "arm",
            "arm_receipt_sha256",
            "double_fit",
            "offline_network_receipt_sha256",
            "offline_network_policy_sha256",
            "selected_artifact_relative_path",
        } or evidence.get("arm") != payload or evidence.get("arm_receipt_sha256") != receipt.receipt_sha256:
            raise GTokCampaignV2Error("tokenizer parent evidence differs from its arm")
        if evidence.get("offline_network_receipt_sha256") != panel_offline_sha256:
            raise GTokCampaignV2Error("tokenizer parent offline identity drifted")
        if evidence.get("offline_network_policy_sha256") != panel_offline_policy_sha256:
            raise GTokCampaignV2Error("tokenizer parent offline policy drifted")
        double_fit = evidence.get("double_fit")
        if not isinstance(double_fit, Mapping):
            raise GTokCampaignV2Error("tokenizer double-fit evidence is absent")
        double_core = dict(double_fit)
        double_identity = double_core.pop("receipt_sha256", None)
        if (
            set(double_core)
            != {
                "first_process_id",
                "first_worker_receipt_sha256",
                "offline_network_receipt_sha256",
                "offline_network_policy_sha256",
                "second_process_id",
                "second_worker_receipt_sha256",
                "status",
                "tokenizer_json_sha256",
                "vocab_size",
            }
            or type(double_core.get("first_process_id")) is not int
            or type(double_core.get("second_process_id")) is not int
            or double_core["first_process_id"] < 1
            or double_core["second_process_id"] < 1
            or double_core["first_process_id"] == double_core["second_process_id"]
            or double_core.get("status") != "PARENT_REHASHED_SUBPROCESSES_MATCH"
            or double_core.get("tokenizer_json_sha256") != receipt.tokenizer_json_sha256
            or double_core.get("vocab_size") != receipt.vocab_size
            or double_core.get("offline_network_receipt_sha256")
            != panel_offline_sha256
            or double_core.get("offline_network_policy_sha256")
            != panel_offline_policy_sha256
            or any(
                not isinstance(double_core.get(name), str)
                or len(double_core[name]) != 64
                or any(character not in _HEX for character in double_core[name])
                for name in (
                    "first_worker_receipt_sha256",
                    "second_worker_receipt_sha256",
                )
            )
            or double_identity != receipt.double_fit_receipt_sha256
            or double_identity != gtok_v2_bound_sha256(DOUBLE_FIT_SCHEMA_V2, double_core)
        ):
            raise GTokCampaignV2Error("tokenizer double-fit identity drifted")
        selected = evidence.get("selected_artifact_relative_path")
        if selected != "fit-a/tokenizer.json":
            raise GTokCampaignV2Error("tokenizer selected artifact path drifted")
        if not isinstance(row.get("output_root"), str) or not row["output_root"]:
            raise GTokCampaignV2Error("tokenizer process output-root evidence is absent")
        arm_artifact_root = artifacts / f"vocab-{vocab_size}"
        _revalidate_tokenizer_producer_evidence_v2(
            artifact_root=arm_artifact_root,
            evidence=evidence,
            receipt=receipt,
            corpus=corpus,
            offline_network_receipt_sha256=tokenizer_offline_physical_sha256,
            offline_network_policy_sha256=tokenizer_offline.policy_sha256,
            producer_python_executable_sha256=(
                tokenizer_offline.python_executable_sha256
            ),
        )
        execution_arm = TokenizerExecutionArmV2(
            receipt=receipt,
            tokenizer_json_path=arm_artifact_root / "fit-a" / "tokenizer.json",
            offline_network_receipt_sha256=tokenizer_offline_physical_sha256,
            offline_network_policy_sha256=tokenizer_offline.policy_sha256,
        )
        execution_arm.load()
        arms.append(execution_arm)
    return tuple(arms)


@dataclass(frozen=True)
class BaseCampaignResultV2:
    preflight: PreflightProjectionReceiptV2
    compute: CampaignComputeReceiptV2
    runs: tuple[GTokRunReceiptV2, ...]
    measurements: tuple[FullRunMeasurementV2, ...]
    matrix: ValidatedGTokMatrixV2
    plans: tuple[tuple[int, int, TrainingPlanV2], ...]
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    offline_network_receipt_sha256: str
    microbatch_sequences: int = GTOK_MICROBATCH_SEQUENCES_V2
    cpu_runtime_identity_sha256: str | None = None
    gpu_uuid_provenance_by_attempt: tuple[tuple[str, str], ...] = ()
    offline_network_receipt_sha256_by_attempt: tuple[tuple[str, str], ...] = ()
    precalibration_determinism_authority_sha256: str | None = None
    precalibration_determinism_replay_plan_set_sha256: str | None = None
    precalibration_determinism_replay_receipt_sha256s: tuple[str, ...] = ()
    campaign_binding_sha256: str = CAMPAIGN_BINDING_SHA256_V2


@dataclass(frozen=True)
class DryRunCampaignResultV2:
    """Non-authoritative orchestration result for synthetic executor tests.

    It deliberately has no ``matrix`` field, so injected measurements cannot
    cross the factory mint boundary even if they satisfy every value-level
    contract.
    """

    preflight: PreflightProjectionReceiptV2
    compute: CampaignComputeReceiptV2
    runs: tuple[GTokRunReceiptV2, ...]
    measurements: tuple[FullRunMeasurementV2, ...]
    plans: tuple[tuple[int, int, TrainingPlanV2], ...]
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    offline_network_receipt_sha256: str | None
    microbatch_sequences: int = GTOK_MICROBATCH_SEQUENCES_V2
    gpu_uuid_provenance: str | None = None
    cpu_runtime_identity_sha256: str | None = None
    authority_status: str = "NON_AUTHORITATIVE_INJECTED_EXECUTORS"
    campaign_binding_sha256: str = CAMPAIGN_BINDING_SHA256_V2


@dataclass(frozen=True)
class CampaignStopArtifactV2:
    reason: str
    cumulative_a100_microseconds: int
    attempts: tuple[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...
    ]
    pending_attempt_ids: tuple[str, ...]
    running_attempt_ids: tuple[str, ...]
    hard_abort_attempt_ids: tuple[str, ...]
    hard_abort_and_report: bool
    return_to_strategy: bool
    calibration_projection_evidence_receipt_sha256: str | None = None
    calibration_projection_evidence_physical_sha256: str | None = None
    campaign_binding_sha256: str = CAMPAIGN_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        references = (
            self.calibration_projection_evidence_receipt_sha256,
            self.calibration_projection_evidence_physical_sha256,
        )
        if (references[0] is None) != (references[1] is None):
            raise GTokCampaignV2Error("STOP projection evidence join is incomplete")
        if references[0] is not None and any(
            len(value) != 64 or any(character not in _HEX for character in value)
            for value in references
        ):
            raise GTokCampaignV2Error("STOP projection evidence join is not SHA-256")


@dataclass(frozen=True)
class CalibrationProjectionEvidenceV2:
    """Durable inputs from which the aggregate 12-hour gate is recomputable."""

    calibrations: tuple[ArmCalibrationProjectionV2, ...]
    calibration_measurements: tuple[CalibrationMeasurementV2, ...]
    calibration_attempts: tuple[ComputeAttemptReceiptV2, ...]
    event_ledger_sha256: str
    projected_campaign_a100_microseconds: int
    tripwire_a100_microseconds: int = GTOK_TRIPWIRE_A100_MICROSECONDS
    full_run_launch_count: int = 0
    precalibration_replay_a100_microseconds: int = 0
    precalibration_determinism_authority_sha256: str | None = None
    campaign_binding_sha256: str = CAMPAIGN_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        if tuple(row.vocab_size for row in self.calibrations) != GTOK_VOCABULARY_ARMS:
            raise GTokCampaignV2Error("projection evidence requires all four arms")
        if len(self.calibration_measurements) != len(self.calibrations) or any(
            not isinstance(row, CalibrationMeasurementV2)
            for row in self.calibration_measurements
        ):
            raise GTokCampaignV2Error(
                "projection evidence requires each physical calibration measurement"
            )
        for projection, measurement in zip(
            self.calibrations,
            self.calibration_measurements,
            strict=True,
        ):
            if (
                measurement.steps != projection.calibration_steps
                or measurement.measured_tokens != projection.measured_tokens
                or measurement.measured_a100_microseconds
                != projection.measured_a100_microseconds
                or measurement.planned_tokens_per_run
                != projection.planned_tokens_per_run
                or measurement.measured_heldout_evaluation_a100_microseconds
                != projection.measured_heldout_evaluation_a100_microseconds
                or measurement.measured_output_surface_a100_microseconds
                != projection.measured_output_surface_a100_microseconds
            ):
                raise GTokCampaignV2Error(
                    "projection differs from its physical calibration measurement"
                )
        if not self.calibration_attempts or any(
            row.kind != "calibration" for row in self.calibration_attempts
        ):
            raise GTokCampaignV2Error("projection evidence requires calibration attempts")
        attempts_by_id = {row.attempt_id: row for row in self.calibration_attempts}
        for projection in self.calibrations:
            selected = attempts_by_id.get(projection.calibration_attempt_id)
            if selected is None or selected.status != "completed":
                raise GTokCampaignV2Error(
                    "projection evidence lacks its selected completed calibration"
                )
        if any(
            row.status not in ("completed", "preempted", "failed")
            for row in self.calibration_attempts
        ):
            raise GTokCampaignV2Error("projection evidence includes a live calibration")
        selected_ids = {row.calibration_attempt_id for row in self.calibrations}
        recovered = sum(
            row.consumed_a100_microseconds
            for row in self.calibration_attempts
            if row.attempt_id not in selected_ids
        )
        if (
            type(self.precalibration_replay_a100_microseconds) is not int
            or self.precalibration_replay_a100_microseconds < 0
        ):
            raise GTokCampaignV2Error("pre-calibration replay charge is invalid")
        if self.precalibration_replay_a100_microseconds:
            if (
                not isinstance(
                    self.precalibration_determinism_authority_sha256, str
                )
                or len(self.precalibration_determinism_authority_sha256) != 64
                or any(
                    character not in _HEX
                    for character in self.precalibration_determinism_authority_sha256
                )
            ):
                raise GTokCampaignV2Error(
                    "projection evidence lacks replay authority identity"
                )
        elif self.precalibration_determinism_authority_sha256 is not None:
            raise GTokCampaignV2Error("projection evidence replay binding is incomplete")
        expected = self.precalibration_replay_a100_microseconds + recovered + sum(
            row.projected_scope_a100_microseconds for row in self.calibrations
        )
        if self.projected_campaign_a100_microseconds != expected:
            raise GTokCampaignV2Error("projection evidence aggregate is not recomputable")
        if self.event_ledger_sha256 != _event_ledger_sha256(self.calibration_attempts):
            raise GTokCampaignV2Error("projection evidence ledger join drifted")
        if self.tripwire_a100_microseconds != GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise GTokCampaignV2Error("projection evidence tripwire drifted")
        if self.full_run_launch_count != 0:
            raise GTokV2Stop("projection evidence was written after a full-run launch")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_calibration_projection_evidence", self
        )


class CalibrationExecutorV2(Protocol):
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


class FullRunExecutorV2(Protocol):
    def __call__(
        self,
        *,
        vocab_size: int,
        seed: int,
        tokenizer: Tokenizer,
        tokenizer_receipt: TokenizerArmReceiptV2,
        plan: TrainingPlanV2,
        initialization_seed: int,
        data_order_seed: int,
        data_order_sha256: str,
        compute_attempt_id: str,
        watchdog_limit_a100_microseconds: int,
        prior_campaign_a100_microseconds: int,
        document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    ) -> FullRunMeasurementV2: ...


@dataclass(frozen=True)
class PreCalibrationPlanRowV2:
    vocab_size: int
    training_seed: int
    plan: TrainingPlanV2

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("pre-calibration plan uses an unregistered vocabulary")
        if self.training_seed not in GTOK_GOVERNED_TRAINING_SEEDS_V2:
            raise ValueError("pre-calibration plan uses an unregistered training seed")
        if not isinstance(self.plan, TrainingPlanV2):
            raise TypeError("pre-calibration plan row requires TrainingPlanV2")


@dataclass(frozen=True)
class PreCalibrationArmMetricsV2:
    vocab_size: int
    tokenizer_receipt_sha256: str
    tokenizer_corpus: TokenizerCorpusMetricsV2

    def __post_init__(self) -> None:
        if self.vocab_size not in GTOK_VOCABULARY_ARMS:
            raise ValueError("pre-calibration metrics use an unregistered vocabulary")
        if (
            len(self.tokenizer_receipt_sha256) != 64
            or any(character not in _HEX for character in self.tokenizer_receipt_sha256)
        ):
            raise ValueError("pre-calibration metrics require a tokenizer SHA-256")
        if (
            self.tokenizer_corpus.nonreserved_row_count + 64
            != self.vocab_size
        ):
            raise ValueError("pre-calibration metrics differ from their vocabulary arm")


@dataclass(frozen=True)
class PreCalibrationCpuEvidenceV2:
    plan_rows: tuple[PreCalibrationPlanRowV2, ...]
    arm_metrics: tuple[PreCalibrationArmMetricsV2, ...]
    initialization_rows: tuple[InitializationSeedStateV2, ...]
    frozen_screen_corpus_sha256: str
    code_closure_receipt_sha256: str
    cpu_runtime_identity_sha256: str
    offline_network_policy_sha256: str
    offline_network_receipt_sha256: str
    generator_script_sha256: str
    microbatch_sequences: int = GTOK_MICROBATCH_SEQUENCES_V2
    meter_classification: str = "PRE_CALIBRATION_CPU_ONLY_NOT_A2_TRIPWIRE"
    accelerator_operations_executed: bool = False

    def __post_init__(self) -> None:
        expected_plan_keys = tuple(
            (vocab_size, seed)
            for vocab_size in GTOK_VOCABULARY_ARMS
            for seed in GTOK_GOVERNED_TRAINING_SEEDS_V2
        )
        if tuple((row.vocab_size, row.training_seed) for row in self.plan_rows) != expected_plan_keys:
            raise ValueError("pre-calibration evidence requires all eight governed plans")
        if tuple(row.vocab_size for row in self.arm_metrics) != GTOK_VOCABULARY_ARMS:
            raise ValueError("pre-calibration evidence requires all four arm metrics")
        expected_initialization = tuple(
            (training_seed, initialization_seed)
            for training_seed, initialization_seed, _data_seed in GTOK_GOVERNED_SEED_ROWS_V2
        )
        if tuple(
            (row.training_seed, row.initialization_seed)
            for row in self.initialization_rows
        ) != expected_initialization:
            raise ValueError("pre-calibration evidence requires both governed initialization rows")
        for name in (
            "frozen_screen_corpus_sha256",
            "code_closure_receipt_sha256",
            "cpu_runtime_identity_sha256",
            "offline_network_policy_sha256",
            "offline_network_receipt_sha256",
            "generator_script_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be SHA-256")
        if self.meter_classification != "PRE_CALIBRATION_CPU_ONLY_NOT_A2_TRIPWIRE":
            raise ValueError("pre-calibration meter classification drifted")
        if self.accelerator_operations_executed is not False:
            raise ValueError("pre-calibration evidence may not claim accelerator work")
        if self.microbatch_sequences != GTOK_MICROBATCH_SEQUENCES_V2:
            raise ValueError("pre-calibration evidence microbatch literal drifted")

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256("weft1_gtok_v2_precalibration_cpu_evidence", self)


@dataclass(frozen=True)
class PrecalibrationDeterminismAuthorityV2:
    """Complete replay-plan and receipt identity ledger before calibration."""

    replay_plan_bindings: tuple[DeterminismReplayPlanBindingV2, ...]
    replay_receipt_sha256s: tuple[str, ...]
    replay_attempt_receipt_sha256s: tuple[str, ...]
    charged_a100_microseconds: int
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    policy_receipt_sha256: str = CUDA_DETERMINISM_POLICY_SHA256_V2
    status: str = "GREEN_BEFORE_CALIBRATION"

    def __post_init__(self) -> None:
        keys = tuple(
            (row.vocab_size, row.terminal_rows) for row in self.replay_plan_bindings
        )
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("pre-calibration replay plans must be canonical and unique")
        if len(self.replay_receipt_sha256s) != len(self.replay_plan_bindings):
            raise ValueError("pre-calibration authority requires one receipt per replay plan")
        if not self.replay_attempt_receipt_sha256s:
            raise ValueError("pre-calibration authority requires its all-attempt meter")
        if len(self.replay_attempt_receipt_sha256s) < (
            2 * len(self.replay_plan_bindings)
        ):
            raise ValueError(
                "pre-calibration authority omits one or more fresh replay replicas"
            )
        for name, values in (
            ("replay_receipt_sha256s", self.replay_receipt_sha256s),
            (
                "replay_attempt_receipt_sha256s",
                self.replay_attempt_receipt_sha256s,
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique append-only identities")
            for value in values:
                if len(value) != 64 or any(character not in _HEX for character in value):
                    raise ValueError(f"{name} must contain SHA-256 values")
        for name in (
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
            "policy_receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be SHA-256")
        if self.policy_receipt_sha256 != CUDA_DETERMINISM_POLICY_SHA256_V2:
            raise ValueError("pre-calibration authority determinism policy drifted")
        if (
            type(self.charged_a100_microseconds) is not int
            or self.charged_a100_microseconds < 1
        ):
            raise ValueError("pre-calibration authority requires a positive replay charge")
        if self.charged_a100_microseconds > GTOK_TRIPWIRE_A100_MICROSECONDS:
            raise GTokV2Stop("pre-calibration replay charge crossed 12 A100-hours")
        if self.status != "GREEN_BEFORE_CALIBRATION":
            raise ValueError("pre-calibration determinism authority is not green")

    @property
    def replay_plan_set_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_precalibration_replay_plan_set",
            self.replay_plan_bindings,
        )

    @property
    def receipt_sha256(self) -> str:
        return gtok_v2_bound_sha256(
            "weft1_gtok_v2_precalibration_determinism_authority",
            self,
        )


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite campaign evidence: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _event_ledger_sha256(
    attempts: tuple[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...
    ],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            tuple(
                {
                    "attempt_id": attempt.attempt_id,
                    "receipt_sha256": attempt.receipt_sha256,
                }
                for attempt in attempts
            )
        )
    ).hexdigest()


def _lifecycle_ledger_path(root: Path) -> Path:
    return root / "campaign-lifecycle.sqlite3"


def _append_lifecycle_event_v2(root: Path, event: CampaignLifecycleEventV2) -> int:
    """Append one lifecycle row under BEGIN IMMEDIATE, then durable mirror it."""

    database = _lifecycle_ledger_path(root)
    connection = sqlite3.connect(database, isolation_level=None, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_index INTEGER PRIMARY KEY,
                attempt_id TEXT NOT NULL,
                payload BLOB NOT NULL,
                receipt_sha256 TEXT NOT NULL UNIQUE,
                campaign_binding_sha256 TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS lifecycle_events_no_update
            BEFORE UPDATE ON lifecycle_events
            BEGIN SELECT RAISE(ABORT, 'lifecycle ledger is append-only'); END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS lifecycle_events_no_delete
            BEFORE DELETE ON lifecycle_events
            BEGIN SELECT RAISE(ABORT, 'lifecycle ledger is append-only'); END
            """
        )
        index = int(
            connection.execute("SELECT COUNT(*) FROM lifecycle_events").fetchone()[0]
        )
        previous = connection.execute(
            """
            SELECT payload FROM lifecycle_events
            WHERE attempt_id = ? ORDER BY event_index DESC LIMIT 1
            """,
            (event.attempt_id,),
        ).fetchone()
        if previous is None:
            if event.phase != "START":
                raise GTokCampaignV2Error("lifecycle attempt must begin with START")
        else:
            prior = json.loads(bytes(previous[0]).decode("utf-8", errors="strict"))
            if prior["phase"] == "TERMINAL":
                raise GTokCampaignV2Error("terminal lifecycle attempt cannot be extended")
            if event.phase == "START":
                raise GTokCampaignV2Error("lifecycle attempt may have only one START")
            if event.charged_a100_microseconds < int(
                prior["charged_a100_microseconds"]
            ):
                raise GTokCampaignV2Error("lifecycle charge regressed")
        connection.execute(
            """
            INSERT INTO lifecycle_events (
                event_index, attempt_id, payload, receipt_sha256,
                campaign_binding_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                index,
                event.attempt_id,
                canonical_json_bytes(asdict(event)),
                event.receipt_sha256,
                CAMPAIGN_BINDING_SHA256_V2,
            ),
        )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    _exclusive_write(
        root / "lifecycle-events" / f"{index:06d}-{event.receipt_sha256}.json",
        {
            "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
            "event_index": index,
            "payload": asdict(event),
            "receipt_sha256": event.receipt_sha256,
            "schema": "weft1_gtok_v2_lifecycle_event",
        },
    )
    return index


def validate_lifecycle_ledger_v2(root: Path) -> tuple[CampaignLifecycleEventV2, ...]:
    database = assert_no_symlink_ancestors(_lifecycle_ledger_path(root)).resolve(
        strict=True
    )
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = tuple(
            connection.execute(
                """
                SELECT event_index, payload, receipt_sha256,
                       campaign_binding_sha256
                FROM lifecycle_events ORDER BY event_index
                """
            )
        )
    except sqlite3.DatabaseError as error:
        raise GTokCampaignV2Error("lifecycle ledger is unreadable") from error
    finally:
        connection.close()
    events: list[CampaignLifecycleEventV2] = []
    prior_by_attempt: dict[str, CampaignLifecycleEventV2] = {}
    for expected_index, row in enumerate(rows):
        index, payload_raw, receipt_sha, binding_sha = row
        try:
            payload = json.loads(bytes(payload_raw).decode("utf-8", errors="strict"))
            event = CampaignLifecycleEventV2(**payload)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise GTokCampaignV2Error("lifecycle payload is invalid") from error
        if (
            index != expected_index
            or receipt_sha != event.receipt_sha256
            or binding_sha != CAMPAIGN_BINDING_SHA256_V2
        ):
            raise GTokCampaignV2Error("lifecycle ledger identity drifted")
        previous = prior_by_attempt.get(event.attempt_id)
        if previous is None:
            if event.phase != "START":
                raise GTokCampaignV2Error("lifecycle attempt lacks START")
        elif (
            previous.phase == "TERMINAL"
            or event.phase == "START"
            or event.charged_a100_microseconds
            < previous.charged_a100_microseconds
        ):
            raise GTokCampaignV2Error("lifecycle transition is invalid")
        mirror = (
            root
            / "lifecycle-events"
            / f"{expected_index:06d}-{event.receipt_sha256}.json"
        )
        try:
            mirror_raw, mirror_payload = load_canonical_json_snapshot(mirror)
        except (OSError, StrictJsonError, TypeError, ValueError) as error:
            raise GTokCampaignV2Error("lifecycle mirror is absent or invalid") from error
        expected_mirror = {
            "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
            "event_index": expected_index,
            "payload": asdict(event),
            "receipt_sha256": event.receipt_sha256,
            "schema": "weft1_gtok_v2_lifecycle_event",
        }
        if mirror_raw != canonical_json_bytes(expected_mirror) + b"\n":
            raise GTokCampaignV2Error("lifecycle mirror differs from SQLite")
        events.append(event)
        prior_by_attempt[event.attempt_id] = event
    return tuple(events)


def _latest_lifecycle_by_attempt_v2(
    events: tuple[CampaignLifecycleEventV2, ...],
) -> dict[str, CampaignLifecycleEventV2]:
    latest: dict[str, CampaignLifecycleEventV2] = {}
    for event in events:
        latest[event.attempt_id] = event
    return latest


def recover_orphaned_lifecycle_attempts_v2(
    root: Path,
) -> tuple[CampaignLifecycleEventV2, ...]:
    """Close hard-killed attempts at their last durable charged reading."""

    events = validate_lifecycle_ledger_v2(root)
    latest = _latest_lifecycle_by_attempt_v2(events)
    recovered: list[CampaignLifecycleEventV2] = []
    for attempt_id in sorted(latest):
        event = latest[attempt_id]
        if event.phase == "TERMINAL":
            continue
        terminal = replace(
            event,
            phase="TERMINAL",
            # The kill may occur anywhere before the next durable heartbeat.
            # Charging one full cadence above the last durable lower bound
            # prevents repeated pre-heartbeat kills from bypassing the 12h
            # campaign meter.
            charged_a100_microseconds=max(1, event.charged_a100_microseconds)
            + HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2,
            terminal_status="preempted",
        )
        _append_lifecycle_event_v2(root, terminal)
        recovered.append(terminal)
    validate_lifecycle_ledger_v2(root)
    return tuple(recovered)


def _next_physical_attempt_id_v2(
    logical_attempt_id: str,
    events: tuple[CampaignLifecycleEventV2, ...],
) -> str:
    existing = tuple(
        sorted(
            {
                event.attempt_id
                for event in events
                if event.logical_attempt_id == logical_attempt_id
            }
        )
    )
    if not existing:
        return logical_attempt_id
    retry = 1
    while f"{logical_attempt_id}.retry-{retry}" in existing:
        retry += 1
    return f"{logical_attempt_id}.retry-{retry}"


class _AttemptHeartbeatV2(AbstractContextManager["_AttemptHeartbeatV2"]):
    """Background durable heartbeat; it never serializes training state."""

    def __init__(
        self,
        *,
        root: Path,
        logical_attempt_id: str,
        attempt_id: str,
        scope: str,
        kind: str,
        gpu_uuid_provenance: str | None,
        offline_network_launch_receipt_sha256: str | None,
    ) -> None:
        self.root = root
        self.logical_attempt_id = logical_attempt_id
        self.attempt_id = attempt_id
        self.scope = scope
        self.kind = kind
        self.gpu_uuid_provenance = gpu_uuid_provenance
        self.offline_network_launch_receipt_sha256 = (
            offline_network_launch_receipt_sha256
        )
        self.started_ns = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _elapsed(self) -> int:
        return max(1, math.ceil((time.perf_counter_ns() - self.started_ns) / 1_000))

    def __enter__(self) -> "_AttemptHeartbeatV2":
        self.started_ns = time.perf_counter_ns()
        _append_lifecycle_event_v2(
            self.root,
            CampaignLifecycleEventV2(
                logical_attempt_id=self.logical_attempt_id,
                attempt_id=self.attempt_id,
                scope=self.scope,
                kind=self.kind,
                phase="START",
                charged_a100_microseconds=1,
                terminal_status=None,
                gpu_uuid_provenance=self.gpu_uuid_provenance,
                offline_network_launch_receipt_sha256=(
                    self.offline_network_launch_receipt_sha256
                ),
            ),
        )

        def pulse() -> None:
            while not self._stop.wait(
                HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2 / 1_000_000
            ):
                try:
                    _append_lifecycle_event_v2(
                        self.root,
                        CampaignLifecycleEventV2(
                            logical_attempt_id=self.logical_attempt_id,
                            attempt_id=self.attempt_id,
                            scope=self.scope,
                            kind=self.kind,
                            phase="HEARTBEAT",
                            charged_a100_microseconds=self._elapsed(),
                            terminal_status=None,
                            gpu_uuid_provenance=self.gpu_uuid_provenance,
                            offline_network_launch_receipt_sha256=(
                                self.offline_network_launch_receipt_sha256
                            ),
                        ),
                    )
                except BaseException as error:  # fail on the owning thread boundary
                    self._error = error
                    self._stop.set()
                    return

        self._thread = threading.Thread(
            target=pulse,
            name=f"weft1-gtok-heartbeat-{self.attempt_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def terminal(
        self,
        *,
        status: str,
        charged_a100_microseconds: int,
        completion_payload: Mapping[str, Any] | None = None,
    ) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise GTokCampaignV2Error("durable heartbeat writer failed") from self._error
        charged = max(self._elapsed(), charged_a100_microseconds)
        _append_lifecycle_event_v2(
            self.root,
            CampaignLifecycleEventV2(
                logical_attempt_id=self.logical_attempt_id,
                attempt_id=self.attempt_id,
                scope=self.scope,
                kind=self.kind,
                phase="TERMINAL",
                charged_a100_microseconds=charged,
                terminal_status=status,
                completion_payload=completion_payload,
                gpu_uuid_provenance=self.gpu_uuid_provenance,
                offline_network_launch_receipt_sha256=(
                    self.offline_network_launch_receipt_sha256
                ),
            ),
        )
        return charged

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None and exc is None:
            raise GTokCampaignV2Error("durable heartbeat writer failed") from self._error
        return None


def _execute_with_lifecycle_v2(
    *,
    root: Path,
    logical_attempt_id: str,
    attempt_id: str,
    scope: str,
    kind: str,
    operation: Callable[[], Any],
    success_charge: Callable[[Any], int],
    success_payload: Callable[[Any], Mapping[str, Any]],
    gpu_uuid_provenance: str | None = None,
    offline_network_launch_receipt_sha256: str | None = None,
    success_watchdog_limit_a100_microseconds: int | None = None,
) -> tuple[Any, int]:
    if (
        success_watchdog_limit_a100_microseconds is not None
        and (
            type(success_watchdog_limit_a100_microseconds) is not int
            or success_watchdog_limit_a100_microseconds < 1
        )
    ):
        raise ValueError("lifecycle success watchdog must be a positive exact integer")
    with _AttemptHeartbeatV2(
        root=root,
        logical_attempt_id=logical_attempt_id,
        attempt_id=attempt_id,
        scope=scope,
        kind=kind,
        gpu_uuid_provenance=gpu_uuid_provenance,
        offline_network_launch_receipt_sha256=(
            offline_network_launch_receipt_sha256
        ),
    ) as heartbeat:
        try:
            value = operation()
        except BaseException as error:
            if isinstance(error, GTokRunWatchdogV2):
                status = "aborted_watchdog"
                reported = error.consumed_a100_microseconds
            elif isinstance(error, GTokCampaignTripwireV2):
                status = "preempted"
                reported = max(1, error.consumed_a100_microseconds)
            elif isinstance(error, GTokDeterminismV2Stop) and error.reason == (
                "DETERMINISM_REPLAY_WATCHDOG"
            ):
                status = "aborted_watchdog"
                reported = heartbeat._elapsed()
            elif isinstance(error, GTokDeterminismV2Stop) and error.reason in (
                "DETERMINISM_REPLAY_PROJECTED_TRIPWIRE",
                "DETERMINISM_REPLAY_RUNTIME_TRIPWIRE",
                "DETERMINISM_REPLAY_RUNTIME_METER",
            ):
                status = "preempted"
                reported = heartbeat._elapsed()
            else:
                status = "failed"
                reported = heartbeat._elapsed()
            charged = heartbeat.terminal(
                status=status,
                charged_a100_microseconds=reported,
            )
            try:
                setattr(error, "_gtok_lifecycle_charge_v2", charged)
            except BaseException as attribute_error:
                raise GTokCampaignV2Error(
                    "failed attempt could not carry its durable lifecycle charge"
                ) from attribute_error
            raise
        reported_success_charge = success_charge(value)
        lifecycle_equivalent_charge = max(
            heartbeat._elapsed(),
            reported_success_charge,
        )
        if (
            success_watchdog_limit_a100_microseconds is not None
            and lifecycle_equivalent_charge
            > success_watchdog_limit_a100_microseconds
        ):
            charged = heartbeat.terminal(
                status="aborted_watchdog",
                charged_a100_microseconds=lifecycle_equivalent_charge,
            )
            error = GTokDeterminismV2Stop(
                "DETERMINISM_REPLAY_WATCHDOG",
                "outer replay lifecycle charge crossed the 2x watchdog",
            )
            setattr(error, "_gtok_lifecycle_charge_v2", charged)
            raise error
        charged = heartbeat.terminal(
            status="completed",
            charged_a100_microseconds=reported_success_charge,
            completion_payload=success_payload(value),
        )
    return value, charged


def _persist_attempt(
    root: Path,
    index: int,
    attempt: ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2,
) -> None:
    physical_json_sha256 = _exclusive_write(
        root / "events" / f"{index:04d}-{attempt.attempt_id}.json",
        {
            "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
            "payload": asdict(attempt),
            "receipt_sha256": attempt.receipt_sha256,
            "schema": "weft1_gtok_v2_campaign_attempt_event",
        },
    )
    database = root / "campaign-events.sqlite3"
    payload = canonical_json_bytes(asdict(attempt))
    connection = sqlite3.connect(database, isolation_level=None, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_events (
                event_index INTEGER PRIMARY KEY,
                attempt_id TEXT NOT NULL UNIQUE,
                payload BLOB NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                physical_json_sha256 TEXT NOT NULL,
                campaign_binding_sha256 TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS campaign_events_no_update
            BEFORE UPDATE ON campaign_events
            BEGIN SELECT RAISE(ABORT, 'campaign event ledger is append-only'); END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS campaign_events_no_delete
            BEFORE DELETE ON campaign_events
            BEGIN SELECT RAISE(ABORT, 'campaign event ledger is append-only'); END
            """
        )
        count = int(connection.execute("SELECT COUNT(*) FROM campaign_events").fetchone()[0])
        if count != index:
            raise GTokCampaignV2Error("SQLite event index is not append-only contiguous")
        connection.execute(
            """
            INSERT INTO campaign_events (
                event_index, attempt_id, payload, receipt_sha256,
                physical_json_sha256, campaign_binding_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                index,
                attempt.attempt_id,
                payload,
                attempt.receipt_sha256,
                physical_json_sha256,
                CAMPAIGN_BINDING_SHA256_V2,
            ),
        )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def validate_sqlite_event_ledger_v2(
    root: Path,
    attempts: tuple[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...
    ],
) -> str:
    """Replay the physical BEGIN-IMMEDIATE ledger and return its logical SHA."""

    database = assert_no_symlink_ancestors(root / "campaign-events.sqlite3").resolve(
        strict=True
    )
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = tuple(
            connection.execute(
                """
                SELECT event_index, attempt_id, payload, receipt_sha256,
                       physical_json_sha256, campaign_binding_sha256
                FROM campaign_events ORDER BY event_index
                """
            )
        )
    except sqlite3.DatabaseError as error:
        raise GTokCampaignV2Error("SQLite campaign ledger is unreadable") from error
    finally:
        connection.close()
    if len(rows) != len(attempts):
        raise GTokCampaignV2Error("SQLite campaign ledger event count drifted")
    for index, (row, attempt) in enumerate(zip(rows, attempts, strict=True)):
        event_index, attempt_id, payload, receipt_sha, json_sha, binding_sha = row
        if (
            event_index != index
            or attempt_id != attempt.attempt_id
            or bytes(payload) != canonical_json_bytes(asdict(attempt))
            or receipt_sha != attempt.receipt_sha256
            or binding_sha != CAMPAIGN_BINDING_SHA256_V2
            or not isinstance(json_sha, str)
            or len(json_sha) != 64
        ):
            raise GTokCampaignV2Error("SQLite campaign event payload or identity drifted")
        mirror = root / "events" / f"{index:04d}-{attempt.attempt_id}.json"
        if hashlib.sha256(mirror.read_bytes()).hexdigest() != json_sha:
            raise GTokCampaignV2Error("SQLite campaign event JSON mirror drifted")
    return _event_ledger_sha256(attempts)


def _load_persisted_attempts_v2(
    root: Path,
) -> tuple[ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...]:
    database = root / "campaign-events.sqlite3"
    if not database.exists():
        return ()
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = tuple(
            connection.execute(
                "SELECT payload FROM campaign_events ORDER BY event_index"
            )
        )
    except sqlite3.DatabaseError as error:
        raise GTokCampaignV2Error("persisted campaign attempt ledger is unreadable") from error
    finally:
        connection.close()
    attempts: list[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2
    ] = []
    for (raw,) in rows:
        try:
            payload = json.loads(bytes(raw).decode("utf-8", errors="strict"))
            if payload.get("kind") == "determinism_replay":
                attempts.append(PrecalibrationReplayAttemptReceiptV2(**payload))
            else:
                attempts.append(ComputeAttemptReceiptV2(**payload))
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise GTokCampaignV2Error("persisted campaign attempt is invalid") from error
    result = tuple(attempts)
    validate_sqlite_event_ledger_v2(root, result)
    return result


def _measurement_from_lifecycle_v2(
    event: CampaignLifecycleEventV2,
) -> CalibrationMeasurementV2:
    if (
        event.phase != "TERMINAL"
        or event.terminal_status != "completed"
        or not isinstance(event.completion_payload, Mapping)
    ):
        raise GTokCampaignV2Error("calibration lifecycle has no completed evidence")
    try:
        return CalibrationMeasurementV2(**event.completion_payload)
    except (TypeError, ValueError) as error:
        raise GTokCampaignV2Error("completed calibration payload is invalid") from error


def _run_from_lifecycle_v2(event: CampaignLifecycleEventV2) -> GTokRunReceiptV2:
    if (
        event.phase != "TERMINAL"
        or event.terminal_status != "completed"
        or not isinstance(event.completion_payload, Mapping)
    ):
        raise GTokCampaignV2Error("run lifecycle has no completed evidence")
    raw_run = event.completion_payload.get("run")
    if not isinstance(raw_run, Mapping):
        raise GTokCampaignV2Error("completed run payload omits its run receipt")
    optimizer_payload = raw_run.get("optimizer")
    if (
        not isinstance(optimizer_payload, Mapping)
        or canonical_json_bytes(optimizer_payload)
        != canonical_json_bytes(asdict(a1_flat_adamw_recipe()))
    ):
        raise GTokCampaignV2Error("resumed run optimizer differs from flat A1 AdamW")
    raw_observations = raw_run.get("observations")
    if not isinstance(raw_observations, list):
        raise GTokCampaignV2Error("resumed run observations are absent")
    observations: list[BpbMilestoneReceiptV2] = []
    for raw_observation in raw_observations:
        if not isinstance(raw_observation, Mapping):
            raise GTokCampaignV2Error("resumed observation is invalid")
        raw_strata = raw_observation.get("strata")
        if not isinstance(raw_strata, list):
            raise GTokCampaignV2Error("resumed observation strata are absent")
        try:
            strata = tuple(StratumNllReceipt(**row) for row in raw_strata)
            observations.append(
                BpbMilestoneReceiptV2(
                    **{
                        key: value
                        for key, value in raw_observation.items()
                        if key != "strata"
                    },
                    strata=strata,
                )
            )
        except (TypeError, ValueError) as error:
            raise GTokCampaignV2Error("resumed observation failed validation") from error
    payload = {
        key: value
        for key, value in raw_run.items()
        if key not in ("optimizer", "observations", "measured_a100_microseconds")
    }
    try:
        return GTokRunReceiptV2(
            **payload,
            measured_a100_microseconds=event.charged_a100_microseconds,
            optimizer=a1_flat_adamw_recipe(),
            observations=tuple(observations),
        )
    except (TypeError, ValueError) as error:
        raise GTokCampaignV2Error("resumed run receipt failed validation") from error


def _full_measurement_from_lifecycle_v2(
    event: CampaignLifecycleEventV2,
) -> FullRunMeasurementV2:
    """Rehydrate and revalidate the complete profiler/section-6.8 evidence."""

    run = _run_from_lifecycle_v2(event)
    raw = event.completion_payload
    if not isinstance(raw, Mapping):
        raise GTokCampaignV2Error("completed run has no physical measurement payload")
    raw_ledger = raw.get("flop_ledger")
    raw_panel = raw.get("measurement_panel")
    if not isinstance(raw_ledger, Mapping) or not isinstance(raw_panel, Mapping):
        raise GTokCampaignV2Error("completed run omits profiler or section-6.8 evidence")
    try:
        shapes = []
        for raw_shape in raw_ledger["shapes"]:
            shape = dict(raw_shape)
            shape["profiler_rows"] = tuple(
                ProfilerOperatorFlopRowV2(**row) for row in shape["profiler_rows"]
            )
            shape["unsupported_rows"] = tuple(
                AnalyticUnsupportedFlopRowV2(**row)
                for row in shape["unsupported_rows"]
            )
            shape["zero_flop_profiler_operators"] = tuple(
                shape["zero_flop_profiler_operators"]
            )
            shapes.append(PhysicalShapeFlopReceiptV2(**shape))
        ledger_payload = dict(raw_ledger)
        ledger_payload["shapes"] = tuple(shapes)
        ledger = CompleteFlopLedgerV2(**ledger_payload)

        panel_payload = dict(raw_panel)
        tokenizer_payload = dict(panel_payload["tokenizer_corpus"])
        tokenizer_payload["strata"] = tuple(
            StratumCompressionMetricsV2(**row) for row in tokenizer_payload["strata"]
        )
        panel_payload["tokenizer_corpus"] = TokenizerCorpusMetricsV2(
            **tokenizer_payload
        )
        panel_payload["full_softmax"] = OutputSurfacePerformanceV2(
            **panel_payload["full_softmax"]
        )
        panel_payload["decode"] = OutputSurfacePerformanceV2(
            **panel_payload["decode"]
        )
        panel_payload["vocabulary_fractions"] = tuple(
            VocabularyFractionRowV2(**row)
            for row in panel_payload["vocabulary_fractions"]
        )
        panel = ArmMeasurementPanelV2(**panel_payload)
        measurement_payload = {
            key: value
            for key, value in raw.items()
            if key not in ("run", "flop_ledger", "measurement_panel")
        }
        measurement = FullRunMeasurementV2(
            **measurement_payload,
            run=run,
            flop_ledger=ledger,
            measurement_panel=panel,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error(
            "completed profiler/section-6.8 evidence failed validation"
        ) from error
    return measurement


def _replay_replica_from_lifecycle_v2(
    event: CampaignLifecycleEventV2,
) -> DeterminismReplayReplicaV2:
    """Rehydrate hashes and scalar evidence only; model state is never retained."""

    if (
        event.kind != "determinism_replay"
        or event.phase != "TERMINAL"
        or event.terminal_status != "completed"
        or not isinstance(event.completion_payload, Mapping)
    ):
        raise GTokCampaignV2Error("replay lifecycle has no completed fingerprint")
    try:
        payload = dict(event.completion_payload)
        fingerprint_payload = dict(payload["fingerprint"])
        fingerprint_payload["shapes"] = tuple(
            ReplayBatchShapeV2(**dict(row)) for row in fingerprint_payload["shapes"]
        )
        for name in (
            "model_state_sha256_by_shape",
            "optimizer_state_sha256_by_shape",
            "evaluation_output_sha256_by_shape",
        ):
            fingerprint_payload[name] = tuple(
                tuple(row) for row in fingerprint_payload[name]
            )
        fingerprint_payload["fused_backend_operator_names"] = tuple(
            fingerprint_payload["fused_backend_operator_names"]
        )
        payload["fingerprint"] = DeterminismReplayFingerprintV2(
            **fingerprint_payload
        )
        replica = DeterminismReplayReplicaV2(**payload)
    except (KeyError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error(
            "completed replay fingerprint failed strict rehydration"
        ) from error
    if replica.gpu_uuid_provenance != event.gpu_uuid_provenance:
        raise GTokCampaignV2Error("replay fingerprint GPU provenance drifted")
    return replica


def _persist_precalibration_determinism_authority_v2(
    root: Path,
    authority: PrecalibrationDeterminismAuthorityV2,
) -> None:
    envelope = {
        "payload": asdict(authority),
        "receipt_sha256": authority.receipt_sha256,
        "replay_plan_set_sha256": authority.replay_plan_set_sha256,
        "schema": "weft1_gtok_v2_precalibration_determinism_authority",
    }
    path = root / "precalibration-determinism-authority.json"
    if path.exists():
        raw, stored = load_canonical_json_snapshot(path)
        if raw != canonical_json_bytes(envelope) + b"\n":
            raise GTokCampaignV2Error(
                "stored pre-calibration determinism authority differs on resume"
            )
    else:
        _exclusive_write(path, envelope)


def _replay_logical_attempt_id_v2(
    binding: DeterminismReplayPlanBindingV2,
    replica_index: int,
) -> str:
    return (
        f"base-determinism-v{binding.vocab_size}"
        f"-t{binding.terminal_rows}-r{replica_index}"
    )


def _replay_attempt_from_terminal_v2(
    event: CampaignLifecycleEventV2,
    *,
    binding: DeterminismReplayPlanBindingV2,
    replica_index: int,
    pair_projection: DeterminismReplayPairProjectionV2 | None,
) -> PrecalibrationReplayAttemptReceiptV2:
    if (
        event.logical_attempt_id
        != _replay_logical_attempt_id_v2(binding, replica_index)
        or event.kind != "determinism_replay"
        or event.phase != "TERMINAL"
        or event.terminal_status is None
    ):
        raise GTokCampaignV2Error("terminal replay event differs from its governed cell")
    if replica_index == 1 and pair_projection is None:
        raise GTokCampaignV2Error("second replay attempt lacks its measured projection")
    return PrecalibrationReplayAttemptReceiptV2(
        attempt_id=event.attempt_id,
        scope="base_screen",
        kind="determinism_replay",
        vocab_size=binding.vocab_size,
        terminal_rows=binding.terminal_rows,
        representative_seed=binding.representative_training_seed,
        replica_index=replica_index,
        consumed_a100_microseconds=event.charged_a100_microseconds,
        status=event.terminal_status,
        replay_plan_binding_sha256=binding.receipt_sha256,
        replay_pair_projection_sha256=(
            None if pair_projection is None else pair_projection.receipt_sha256
        ),
        projected_replica_a100_microseconds=(
            None
            if pair_projection is None
            else pair_projection.projected_second_replica_a100_microseconds
        ),
        watchdog_limit_a100_microseconds=(
            None
            if pair_projection is None
            else pair_projection.second_replica_watchdog_a100_microseconds
        ),
        hard_abort_issued=(event.terminal_status == "aborted_watchdog"),
    )


def _run_precalibration_determinism_gate_v2(
    *,
    root: Path,
    replay_plan_bindings: tuple[DeterminismReplayPlanBindingV2, ...],
    plans: Mapping[tuple[int, int], TrainingPlanV2],
    device: torch.device,
    microbatch_sequences: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    gpu_uuid_provenance: str,
    offline_network_receipt_sha256: str,
    attempts: list[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2
    ],
    revalidate_code_closure: Callable[[], None],
    training_seeds: tuple[int, int],
) -> tuple[
    PrecalibrationDeterminismAuthorityV2,
    tuple[PrecalibrationDeterminismReplayReceiptV2, ...],
]:
    """Execute/resume the canonical physical replay pairs before calibration."""

    later_attempts_present = any(
        isinstance(row, ComputeAttemptReceiptV2) for row in attempts
    )
    expected_keys = tuple(
        sorted(set((row.vocab_size, row.terminal_rows) for row in replay_plan_bindings))
    )
    if tuple(
        (row.vocab_size, row.terminal_rows) for row in replay_plan_bindings
    ) != expected_keys:
        raise GTokCampaignV2Error("replay plan traversal is not canonical")

    def all_pending() -> tuple[str, ...]:
        replay = tuple(
            _replay_logical_attempt_id_v2(binding, replica_index)
            for binding in replay_plan_bindings
            for replica_index in (0, 1)
        )
        calibration = tuple(
            _attempt_id("calibration", arm) for arm in GTOK_VOCABULARY_ARMS
        )
        full = tuple(
            _attempt_id("run", arm, seed)
            for arm in GTOK_VOCABULARY_ARMS
            for seed in training_seeds
        )
        return replay + calibration + full

    def stop(reason: str, *, running: tuple[str, ...] = ()) -> None:
        lifecycle_events = (
            validate_lifecycle_ledger_v2(root)
            if _lifecycle_ledger_path(root).exists()
            else ()
        )
        terminal_logical = {
            event.logical_attempt_id
            for event in lifecycle_events
            if event.phase == "TERMINAL"
        }
        pending = tuple(
            row for row in all_pending() if row not in terminal_logical and row not in running
        )
        _write_stop(
            root,
            reason=reason,
            cumulative=sum(row.consumed_a100_microseconds for row in attempts),
            attempts=tuple(attempts),
            pending=pending,
            running=running,
        )

    def persist_terminal_attempt(
        event: CampaignLifecycleEventV2,
        *,
        binding: DeterminismReplayPlanBindingV2,
        replica_index: int,
        pair_projection: DeterminismReplayPairProjectionV2 | None,
    ) -> PrecalibrationReplayAttemptReceiptV2:
        value = _replay_attempt_from_terminal_v2(
            event,
            binding=binding,
            replica_index=replica_index,
            pair_projection=pair_projection,
        )
        existing = next(
            (row for row in attempts if row.attempt_id == value.attempt_id),
            None,
        )
        if existing is None:
            attempts.append(value)
            _persist_attempt(root, len(attempts) - 1, value)
        elif existing != value:
            raise GTokCampaignV2Error(
                "persisted replay attempt differs from its lifecycle terminal"
            )
        return value

    def terminal_events(logical_attempt_id: str) -> tuple[CampaignLifecycleEventV2, ...]:
        if not _lifecycle_ledger_path(root).exists():
            return ()
        return tuple(
            event
            for event in validate_lifecycle_ledger_v2(root)
            if event.logical_attempt_id == logical_attempt_id
            and event.phase == "TERMINAL"
        )

    if later_attempts_present:
        authority_path = root / "precalibration-determinism-authority.json"
        if not authority_path.is_file():
            stop("LATER_ATTEMPTS_WITHOUT_REPLAY_AUTHORITY")
            raise GTokV2Stop(
                "later campaign attempts exist without green replay authority"
            )
        persisted_attempt_ids = {row.attempt_id for row in attempts}
        preloaded_receipts: list[PrecalibrationDeterminismReplayReceiptV2] = []
        for binding in replay_plan_bindings:
            receipt_path = (
                root
                / "determinism-replay"
                / f"v{binding.vocab_size}-t{binding.terminal_rows}"
                / "receipt.json"
            )
            if not receipt_path.is_file():
                stop("LATER_ATTEMPTS_WITH_INCOMPLETE_REPLAY_RECEIPTS")
                raise GTokV2Stop(
                    "later campaign attempts exist with an incomplete replay receipt set"
                )
            try:
                receipt = load_precalibration_determinism_replay_receipt_v2(
                    receipt_path
                )
            except BaseException as error:
                stop("STORED_REPLAY_RECEIPT_AUTHENTICATION_FAILED")
                raise GTokV2Stop(
                    "stored replay receipt failed strict authentication"
                ) from error
            if (
                receipt.replay_plan_binding != binding
                or receipt.training_runtime_receipt_sha256
                != training_runtime_receipt_sha256
                or receipt.code_closure_receipt_sha256
                != code_closure_receipt_sha256
                or receipt.fingerprint.policy_receipt_sha256
                != CUDA_DETERMINISM_POLICY_SHA256_V2
            ):
                stop("STORED_REPLAY_RECEIPT_CURRENT_EVIDENCE_MISMATCH")
                raise GTokV2Stop(
                    "stored replay receipt differs from current plan/runtime/code/policy"
                )
            preloaded_receipts.append(receipt)
            for replica_index in (0, 1):
                terminals = terminal_events(
                    _replay_logical_attempt_id_v2(binding, replica_index)
                )
                completed = tuple(
                    row for row in terminals if row.terminal_status == "completed"
                )
                if len(completed) != 1 or completed[0].attempt_id not in persisted_attempt_ids:
                    stop("LATER_ATTEMPTS_OUTRAN_REPLAY_COMPLETION")
                    raise GTokV2Stop(
                        "later campaign attempts outran durable replay completion"
                    )
        replay_attempts = tuple(
            row
            for row in attempts
            if isinstance(row, PrecalibrationReplayAttemptReceiptV2)
        )
        binding_identities = {row.receipt_sha256 for row in replay_plan_bindings}
        if not replay_attempts or any(
            row.replay_plan_binding_sha256 not in binding_identities
            for row in replay_attempts
        ):
            stop("STORED_REPLAY_ATTEMPT_METER_PLAN_MISMATCH")
            raise GTokV2Stop(
                "stored replay attempt meter differs from the current plan set"
            )
        expected_authority = PrecalibrationDeterminismAuthorityV2(
            replay_plan_bindings=replay_plan_bindings,
            replay_receipt_sha256s=tuple(
                row.receipt_sha256 for row in preloaded_receipts
            ),
            replay_attempt_receipt_sha256s=tuple(
                row.receipt_sha256 for row in replay_attempts
            ),
            charged_a100_microseconds=sum(
                row.consumed_a100_microseconds for row in replay_attempts
            ),
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
        )
        expected_envelope = {
            "payload": asdict(expected_authority),
            "receipt_sha256": expected_authority.receipt_sha256,
            "replay_plan_set_sha256": expected_authority.replay_plan_set_sha256,
            "schema": "weft1_gtok_v2_precalibration_determinism_authority",
        }
        try:
            raw, stored = load_canonical_json_snapshot(authority_path)
        except BaseException as error:
            stop("STORED_REPLAY_AUTHORITY_AUTHENTICATION_FAILED")
            raise GTokV2Stop(
                "stored replay authority failed strict authentication"
            ) from error
        if raw != canonical_json_bytes(expected_envelope) + b"\n":
            stop("STORED_REPLAY_AUTHORITY_CURRENT_EVIDENCE_MISMATCH")
            raise GTokV2Stop(
                "stored green replay authority failed exact current-evidence authentication"
            )
        return expected_authority, tuple(preloaded_receipts)

    revalidate_code_closure()
    try:
        policy_attestation: CudaDeterminismAttestationV2 = (
            apply_and_attest_cuda_determinism_policy_v2(device=device)
        )
    except BaseException as error:
        stop(f"DETERMINISM_POLICY_FAILED:{type(error).__name__}")
        raise GTokV2Stop(
            "determinism policy attestation failed before replay"
        ) from error

    pair_receipts: list[PrecalibrationDeterminismReplayReceiptV2] = []
    for binding in replay_plan_bindings:
        plan = plans.get(
            (binding.vocab_size, binding.representative_training_seed)
        )
        if (
            not isinstance(plan, TrainingPlanV2)
            or plan.receipt_sha256 != binding.representative_plan_sha256
        ):
            stop("DETERMINISM_REPLAY_PLAN_JOIN_FAILED")
            raise GTokV2Stop("replay representative plan differs from its authority")

        logical_first = _replay_logical_attempt_id_v2(binding, 0)
        first_terminals = terminal_events(logical_first)
        completed_first = tuple(
            row for row in first_terminals if row.terminal_status == "completed"
        )
        if len(completed_first) > 1 or any(
            row.terminal_status not in ("completed", "preempted")
            for row in first_terminals
        ):
            stop("DETERMINISM_REPLAY_FIRST_TERMINAL_DRIFT")
            raise GTokV2Stop("first replay lifecycle is not resumable")
        for event in first_terminals:
            if event.terminal_status == "preempted":
                persist_terminal_attempt(
                    event,
                    binding=binding,
                    replica_index=0,
                    pair_projection=None,
                )
        if completed_first:
            first_event = completed_first[0]
            first = _replay_replica_from_lifecycle_v2(first_event)
            existing_index = next(
                (
                    index
                    for index, row in enumerate(attempts)
                    if row.attempt_id == first_event.attempt_id
                ),
                None,
            )
            if existing_index is None:
                prior_pair_charge = sum(
                    row.consumed_a100_microseconds for row in attempts
                )
                persist_terminal_attempt(
                    first_event,
                    binding=binding,
                    replica_index=0,
                    pair_projection=None,
                )
            else:
                prior_pair_charge = sum(
                    row.consumed_a100_microseconds
                    for row in attempts[:existing_index]
                )
        else:
            prior_pair_charge = sum(
                row.consumed_a100_microseconds for row in attempts
            )
            if prior_pair_charge >= GTOK_TRIPWIRE_A100_MICROSECONDS:
                stop("DETERMINISM_REPLAY_RESUME_METER_EXHAUSTED")
                raise GTokV2Stop("replay resume exhausted the cumulative tripwire")
            lifecycle = (
                validate_lifecycle_ledger_v2(root)
                if _lifecycle_ledger_path(root).exists()
                else ()
            )
            attempt_id = _next_physical_attempt_id_v2(logical_first, lifecycle)
            revalidate_code_closure()
            try:
                first, first_lifecycle_charge = _execute_with_lifecycle_v2(
                    root=root,
                    logical_attempt_id=logical_first,
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="determinism_replay",
                    operation=lambda: execute_precalibration_determinism_replay_replica_v2(
                        replica_index=0,
                        policy_attestation=policy_attestation,
                        replay_plan_binding=binding,
                        plan=plan,
                        device=device,
                        microbatch_sequences=microbatch_sequences,
                        gpu_uuid_provenance=gpu_uuid_provenance,
                        watchdog_limit_a100_microseconds=None,
                        prior_campaign_a100_microseconds=prior_pair_charge,
                    ),
                    success_charge=lambda value: value.charged_device_microseconds,
                    success_payload=lambda value: asdict(value),
                    gpu_uuid_provenance=gpu_uuid_provenance,
                    offline_network_launch_receipt_sha256=(
                        offline_network_receipt_sha256
                    ),
                )
                first_event = terminal_events(logical_first)[-1]
                if first_event.charged_a100_microseconds != first_lifecycle_charge:
                    raise GTokCampaignV2Error(
                        "first replay lifecycle return differs from its terminal event"
                    )
                persist_terminal_attempt(
                    first_event,
                    binding=binding,
                    replica_index=0,
                    pair_projection=None,
                )
            except BaseException as error:
                latest = terminal_events(logical_first)
                if latest:
                    persist_terminal_attempt(
                        latest[-1],
                        binding=binding,
                        replica_index=0,
                        pair_projection=None,
                    )
                stop(f"DETERMINISM_REPLAY_FIRST_FAILED:{type(error).__name__}")
                raise GTokV2Stop("first determinism replay replica failed") from error

        first_lifecycle_charge = first_event.charged_a100_microseconds
        if (
            first.replica_index != 0
            or first.fingerprint.replay_plan_binding_sha256
            != binding.receipt_sha256
        ):
            stop("DETERMINISM_REPLAY_FIRST_FINGERPRINT_JOIN_FAILED")
            raise GTokV2Stop("first replay fingerprint differs from its plan binding")
        try:
            pair_projection = project_second_determinism_replay_replica_v2(
                first,
                replay_plan_binding=binding,
                first_lifecycle_a100_microseconds=first_lifecycle_charge,
                prior_campaign_a100_microseconds=prior_pair_charge,
            )
        except BaseException as error:
            stop("DETERMINISM_REPLAY_PAIR_PROJECTION_FAILED")
            raise GTokV2Stop("replay pair projection crossed a gate") from error

        logical_second = _replay_logical_attempt_id_v2(binding, 1)
        second_terminals = terminal_events(logical_second)
        completed_second = tuple(
            row for row in second_terminals if row.terminal_status == "completed"
        )
        if len(completed_second) > 1 or any(
            row.terminal_status not in ("completed", "preempted")
            for row in second_terminals
        ):
            stop("DETERMINISM_REPLAY_SECOND_TERMINAL_DRIFT")
            raise GTokV2Stop("second replay lifecycle is not resumable")
        for event in second_terminals:
            if event.terminal_status == "preempted":
                persist_terminal_attempt(
                    event,
                    binding=binding,
                    replica_index=1,
                    pair_projection=pair_projection,
                )
        if completed_second:
            second_event = completed_second[0]
            second = _replay_replica_from_lifecycle_v2(second_event)
            persist_terminal_attempt(
                second_event,
                binding=binding,
                replica_index=1,
                pair_projection=pair_projection,
            )
        else:
            current_charge = sum(
                row.consumed_a100_microseconds for row in attempts
            )
            if (
                current_charge
                + pair_projection.projected_second_replica_a100_microseconds
                > GTOK_TRIPWIRE_A100_MICROSECONDS
            ):
                stop("DETERMINISM_REPLAY_SECOND_PROJECTED_TRIPWIRE")
                raise GTokV2Stop(
                    "second replay replica would cross the cumulative tripwire"
                )
            lifecycle = (
                validate_lifecycle_ledger_v2(root)
                if _lifecycle_ledger_path(root).exists()
                else ()
            )
            attempt_id = _next_physical_attempt_id_v2(logical_second, lifecycle)
            revalidate_code_closure()
            try:
                second, second_lifecycle_charge = _execute_with_lifecycle_v2(
                    root=root,
                    logical_attempt_id=logical_second,
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="determinism_replay",
                    operation=lambda: execute_precalibration_determinism_replay_replica_v2(
                        replica_index=1,
                        policy_attestation=policy_attestation,
                        replay_plan_binding=binding,
                        plan=plan,
                        device=device,
                        microbatch_sequences=microbatch_sequences,
                        gpu_uuid_provenance=gpu_uuid_provenance,
                        watchdog_limit_a100_microseconds=(
                            pair_projection.second_replica_watchdog_a100_microseconds
                        ),
                        prior_campaign_a100_microseconds=current_charge,
                    ),
                    success_charge=lambda value: value.charged_device_microseconds,
                    success_payload=lambda value: asdict(value),
                    gpu_uuid_provenance=gpu_uuid_provenance,
                    offline_network_launch_receipt_sha256=(
                        offline_network_receipt_sha256
                    ),
                    success_watchdog_limit_a100_microseconds=(
                        pair_projection.second_replica_watchdog_a100_microseconds
                    ),
                )
                second_event = terminal_events(logical_second)[-1]
                if second_event.charged_a100_microseconds != second_lifecycle_charge:
                    raise GTokCampaignV2Error(
                        "second replay lifecycle return differs from its terminal event"
                    )
                persist_terminal_attempt(
                    second_event,
                    binding=binding,
                    replica_index=1,
                    pair_projection=pair_projection,
                )
            except BaseException as error:
                latest = terminal_events(logical_second)
                if latest:
                    persist_terminal_attempt(
                        latest[-1],
                        binding=binding,
                        replica_index=1,
                        pair_projection=pair_projection,
                    )
                stop(f"DETERMINISM_REPLAY_SECOND_FAILED:{type(error).__name__}")
                raise GTokV2Stop("second determinism replay replica failed") from error

        second_lifecycle_charge = second_event.charged_a100_microseconds
        if sum(
            row.consumed_a100_microseconds for row in attempts
        ) > GTOK_TRIPWIRE_A100_MICROSECONDS:
            stop("DETERMINISM_REPLAY_RUNTIME_TRIPWIRE")
            raise GTokV2Stop("determinism replay crossed the cumulative tripwire")
        try:
            receipt = _mint_precalibration_determinism_replay_receipt_v2(
                first,
                second,
                policy_attestation=policy_attestation,
                replay_plan_binding=binding,
                pair_projection=pair_projection,
                first_lifecycle_a100_microseconds=first_lifecycle_charge,
                second_lifecycle_a100_microseconds=second_lifecycle_charge,
                training_runtime_receipt_sha256=training_runtime_receipt_sha256,
                code_closure_receipt_sha256=code_closure_receipt_sha256,
            )
            receipt_path = (
                root
                / "determinism-replay"
                / f"v{binding.vocab_size}-t{binding.terminal_rows}"
                / "receipt.json"
            )
            if receipt_path.exists():
                stored = load_precalibration_determinism_replay_receipt_v2(
                    receipt_path
                )
                if stored != receipt:
                    raise GTokCampaignV2Error(
                        "stored replay pair receipt differs on resume"
                    )
            else:
                write_precalibration_determinism_replay_receipt_v2(
                    receipt_path,
                    receipt,
                )
        except BaseException as error:
            stop("DETERMINISM_REPLAY_RECEIPT_MINT_FAILED")
            raise GTokV2Stop("determinism replay receipt failed closed") from error
        pair_receipts.append(receipt)

    replay_attempts = tuple(
        row
        for row in attempts
        if isinstance(row, PrecalibrationReplayAttemptReceiptV2)
    )
    authority = PrecalibrationDeterminismAuthorityV2(
        replay_plan_bindings=replay_plan_bindings,
        replay_receipt_sha256s=tuple(row.receipt_sha256 for row in pair_receipts),
        replay_attempt_receipt_sha256s=tuple(
            row.receipt_sha256 for row in replay_attempts
        ),
        charged_a100_microseconds=sum(
            row.consumed_a100_microseconds for row in replay_attempts
        ),
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
    )
    _persist_precalibration_determinism_authority_v2(root, authority)
    revalidate_code_closure()
    return authority, tuple(pair_receipts)


def _persist_calibration_projection_evidence(
    root: Path,
    *,
    projections: tuple[ArmCalibrationProjectionV2, ...],
    measurements: tuple[CalibrationMeasurementV2, ...],
    attempts: tuple[ComputeAttemptReceiptV2, ...],
    precalibration_determinism_authority: (
        PrecalibrationDeterminismAuthorityV2 | None
    ) = None,
) -> tuple[CalibrationProjectionEvidenceV2, str]:
    all_persisted = _load_persisted_attempts_v2(root)
    if tuple(item for item in all_persisted if item.kind == "calibration") != attempts:
        raise GTokCampaignV2Error(
            "calibration projection evidence differs from the append-only ledger"
        )
    evidence = CalibrationProjectionEvidenceV2(
        calibrations=projections,
        calibration_measurements=measurements,
        calibration_attempts=attempts,
        event_ledger_sha256=_event_ledger_sha256(attempts),
        projected_campaign_a100_microseconds=(
            (
                0
                if precalibration_determinism_authority is None
                else precalibration_determinism_authority.charged_a100_microseconds
            )
            +
            sum(row.projected_scope_a100_microseconds for row in projections)
            + sum(
                attempt.consumed_a100_microseconds
                for attempt in attempts
                if attempt.attempt_id
                not in {row.calibration_attempt_id for row in projections}
            )
        ),
        precalibration_replay_a100_microseconds=(
            0
            if precalibration_determinism_authority is None
            else precalibration_determinism_authority.charged_a100_microseconds
        ),
        precalibration_determinism_authority_sha256=(
            None
            if precalibration_determinism_authority is None
            else precalibration_determinism_authority.receipt_sha256
        ),
    )
    envelope = {
        "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
        "payload": asdict(evidence),
        "receipt_sha256": evidence.receipt_sha256,
        "schema": "weft1_gtok_v2_calibration_projection_evidence",
    }
    path = root / "calibration-projection-evidence.json"
    if path.exists():
        raw, stored = load_canonical_json_snapshot(path)
        if raw != canonical_json_bytes(envelope) + b"\n":
            raise GTokCampaignV2Error(
                "stored calibration projection evidence differs on resume"
            )
        physical = hashlib.sha256(raw).hexdigest()
    else:
        physical = _exclusive_write(path, envelope)
    return evidence, physical


def build_preflight_projection_v2(
    calibrations: tuple[ArmCalibrationProjectionV2, ...],
    *,
    prior_campaign_a100_microseconds: int = 0,
    prior_event_ledger_sha256: str | None = None,
    scope: str = "base_screen",
    recovered_attempt_a100_microseconds: int = 0,
    precalibration_replay_attempts: tuple[
        PrecalibrationReplayAttemptReceiptV2, ...
    ] = (),
    precalibration_replay_plan_set_sha256: str | None = None,
    precalibration_replay_receipt_sha256s: tuple[str, ...] = (),
    precalibration_replay_authority_sha256: str | None = None,
) -> PreflightProjectionReceiptV2:
    return PreflightProjectionReceiptV2(
        scope=scope,
        prior_campaign_a100_microseconds=prior_campaign_a100_microseconds,
        prior_event_ledger_sha256=prior_event_ledger_sha256,
        calibrations=calibrations,
        projected_campaign_a100_microseconds=(
            prior_campaign_a100_microseconds
            + recovered_attempt_a100_microseconds
            + sum(
                item.consumed_a100_microseconds
                for item in precalibration_replay_attempts
            )
            + sum(item.projected_scope_a100_microseconds for item in calibrations)
        ),
        recovered_attempt_a100_microseconds=recovered_attempt_a100_microseconds,
        precalibration_replay_attempts=precalibration_replay_attempts,
        precalibration_replay_plan_set_sha256=(
            precalibration_replay_plan_set_sha256
        ),
        precalibration_replay_receipt_sha256s=(
            precalibration_replay_receipt_sha256s
        ),
        precalibration_replay_authority_sha256=(
            precalibration_replay_authority_sha256
        ),
    )


def _attempt_id(kind: str, vocab_size: int, seed: int | None = None) -> str:
    suffix = "arm" if seed is None else str(seed)
    return f"base-{kind}-v{vocab_size}-s{suffix}"


def _physical_initialization_rows_v2() -> tuple[InitializationSeedStateV2, ...]:
    """Build fresh CPU models for every governed seed/arm in CPU precompute."""

    rows: list[InitializationSeedStateV2] = []
    for training_seed, initialization_seed, _data_seed in GTOK_GOVERNED_SEED_ROWS_V2:
        arms: list[InitializationArmStateV2] = []
        for vocab_size in GTOK_VOCABULARY_ARMS:
            model = build_gtok_proxy_model_v2(
                vocab_size=vocab_size,
                initialization_seed=initialization_seed,
                run_seed=training_seed,
            )
            try:
                arms.append(
                    InitializationArmStateV2(
                        vocab_size=vocab_size,
                        shared_nonvocabulary_state_sha256=(
                            shared_nonvocabulary_state_sha256_v2(model)
                        ),
                    )
                )
            finally:
                del model
        rows.append(
            InitializationSeedStateV2(
                training_seed=training_seed,
                initialization_seed=initialization_seed,
                arms=tuple(arms),
            )
        )
    return tuple(rows)


def _physical_initialization_equality_evidence_v2(
    *,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    offline_network_receipt_sha256: str,
) -> InitializationEqualityEvidenceV2:
    return InitializationEqualityEvidenceV2(
        rows=_physical_initialization_rows_v2(),
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
    )


def _persist_initialization_equality_v2(
    root: Path,
    evidence: InitializationEqualityEvidenceV2,
) -> None:
    envelope = {
        "payload": asdict(evidence),
        "receipt_sha256": evidence.receipt_sha256,
        "schema": "weft1_gtok_v2_initialization_equality_evidence",
    }
    path = root / "initialization-equality-evidence.json"
    if path.exists():
        raw, _stored = load_canonical_json_snapshot(path)
        if raw != canonical_json_bytes(envelope) + b"\n":
            raise GTokCampaignV2Error(
                "stored initialization-equality evidence differs on resume"
            )
    else:
        _exclusive_write(path, envelope)


def _persist_precalibration_cpu_evidence_v2(
    root: Path,
    evidence: PreCalibrationCpuEvidenceV2,
) -> None:
    """Persist the literal A2-R6 boundary before the first charged burst.

    Stream planning and seed-invariant tokenizer scans are CPU-only corpus work.
    They happen before the calibration burst starts, execute no accelerator
    operator, and therefore sit outside the A2-R6 A100 meter.  The immutable
    evidence makes that exclusion explicit and gives every full run the same
    once-per-arm tokenizer metrics instead of repeating the 4 GB T scan per seed.
    """

    envelope = {
        "payload": asdict(evidence),
        "receipt_sha256": evidence.receipt_sha256,
        "schema": "weft1_gtok_v2_precalibration_cpu_evidence",
    }
    path = root / "precalibration-cpu-evidence.json"
    if path.exists():
        raw, _stored = load_canonical_json_snapshot(path)
        if raw != canonical_json_bytes(envelope) + b"\n":
            raise GTokCampaignV2Error(
                "stored pre-calibration CPU evidence differs on resume"
            )
    else:
        _exclusive_write(path, envelope)


def build_precalibration_cpu_evidence_v2(
    *,
    corpus: FrozenScreenCorpusV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    code_closure_receipt_sha256: str,
    cpu_runtime_identity_sha256: str,
    offline_network_policy_sha256: str,
    offline_network_receipt_sha256: str,
    generator_script_sha256: str,
) -> PreCalibrationCpuEvidenceV2:
    """Perform the expensive CPU-only planning/metric pass before A100 allocation."""

    if source.physical_d6_evidence_sha256 != corpus.d6_physical_evidence_sha256:
        raise GTokCampaignV2Error("pre-calibration source differs from frozen D6")
    if source.training_raw_bytes != corpus.training_realized_bytes:
        raise GTokCampaignV2Error("pre-calibration T bytes differ from the freeze")
    if source.heldout_raw_bytes_by_stratum != corpus.heldout_denominator_signature:
        raise GTokCampaignV2Error("pre-calibration H strata differ from the freeze")
    if tuple(arm.receipt.vocab_size for arm in tokenizer_arms) != GTOK_VOCABULARY_ARMS:
        raise GTokCampaignV2Error("pre-calibration tokenizer arms are not canonical")
    if (
        len(code_closure_receipt_sha256) != 64
        or any(character not in _HEX for character in code_closure_receipt_sha256)
    ):
        raise GTokCampaignV2Error("pre-calibration code closure is not SHA-256")
    tokenizers = {arm.receipt.vocab_size: arm.load() for arm in tokenizer_arms}
    plans: dict[tuple[int, int], TrainingPlanV2] = {}
    metrics: dict[int, TokenizerCorpusMetricsV2] = {}
    for vocab_size in GTOK_VOCABULARY_ARMS:
        tokenizer = tokenizers[vocab_size]
        for seed in GTOK_GOVERNED_TRAINING_SEEDS_V2:
            plans[(vocab_size, seed)] = plan_training_stream_v2(
                lambda seed=seed: source.training_documents(seed),
                tokenizer=tokenizer,
                expected_realized_raw_bytes=corpus.training_realized_bytes,
            )
        if len(
            {
                plans[(vocab_size, seed)].optimizer_steps
                for seed in GTOK_GOVERNED_TRAINING_SEEDS_V2
            }
        ) != 1:
            raise GTokCampaignV2Error("seed order changed an arm's total optimizer steps")
        metrics[vocab_size] = measure_tokenizer_corpus_metrics_v2(
            tokenizer=tokenizer,
            training_document_factory=(
                lambda seed=GTOK_GOVERNED_TRAINING_SEEDS_V2[0]: source.training_documents(seed)
            ),
            heldout_factory=source.heldout_documents,
        )
    return PreCalibrationCpuEvidenceV2(
        plan_rows=tuple(
            PreCalibrationPlanRowV2(
                vocab_size=vocab_size,
                training_seed=seed,
                plan=plans[(vocab_size, seed)],
            )
            for vocab_size in GTOK_VOCABULARY_ARMS
            for seed in GTOK_GOVERNED_TRAINING_SEEDS_V2
        ),
        arm_metrics=tuple(
            PreCalibrationArmMetricsV2(
                vocab_size=vocab_size,
                tokenizer_receipt_sha256=next(
                    arm.receipt.receipt_sha256
                    for arm in tokenizer_arms
                    if arm.receipt.vocab_size == vocab_size
                ),
                tokenizer_corpus=metrics[vocab_size],
            )
            for vocab_size in GTOK_VOCABULARY_ARMS
        ),
        initialization_rows=_physical_initialization_rows_v2(),
        frozen_screen_corpus_sha256=corpus.receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        cpu_runtime_identity_sha256=cpu_runtime_identity_sha256,
        offline_network_policy_sha256=offline_network_policy_sha256,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
        generator_script_sha256=generator_script_sha256,
    )


def write_precalibration_cpu_evidence_v2(
    path: Path,
    evidence: PreCalibrationCpuEvidenceV2,
) -> str:
    if not isinstance(path, Path) or not isinstance(evidence, PreCalibrationCpuEvidenceV2):
        raise TypeError("pre-calibration writer requires pathlib.Path and typed evidence")
    return _exclusive_write(
        path,
        {
            "payload": asdict(evidence),
            "receipt_sha256": evidence.receipt_sha256,
            "schema": "weft1_gtok_v2_precalibration_cpu_evidence",
        },
    )


def load_precalibration_cpu_evidence_v2(path: Path) -> PreCalibrationCpuEvidenceV2:
    resolved = assert_no_symlink_ancestors(path).resolve(strict=True)
    raw, envelope = load_canonical_json_snapshot(resolved)
    if (
        raw != canonical_json_bytes(envelope) + b"\n"
        or not isinstance(envelope, Mapping)
        or set(envelope) != {"payload", "receipt_sha256", "schema"}
        or envelope.get("schema") != "weft1_gtok_v2_precalibration_cpu_evidence"
        or not isinstance(envelope.get("payload"), Mapping)
    ):
        raise GTokCampaignV2Error("pre-calibration CPU evidence envelope drifted")
    payload = dict(envelope["payload"])
    try:
        plan_rows = tuple(
            PreCalibrationPlanRowV2(
                vocab_size=row["vocab_size"],
                training_seed=row["training_seed"],
                plan=TrainingPlanV2(**row["plan"]),
            )
            for row in payload.pop("plan_rows")
        )
        arm_metrics = []
        for row in payload.pop("arm_metrics"):
            metric_payload = dict(row["tokenizer_corpus"])
            metric_payload["strata"] = tuple(
                StratumCompressionMetricsV2(**stratum)
                for stratum in metric_payload["strata"]
            )
            arm_metrics.append(
                PreCalibrationArmMetricsV2(
                    vocab_size=row["vocab_size"],
                    tokenizer_receipt_sha256=row["tokenizer_receipt_sha256"],
                    tokenizer_corpus=TokenizerCorpusMetricsV2(**metric_payload),
                )
            )
        initialization_rows = tuple(
            InitializationSeedStateV2(
                training_seed=row["training_seed"],
                initialization_seed=row["initialization_seed"],
                arms=tuple(
                    InitializationArmStateV2(**arm) for arm in row["arms"]
                ),
            )
            for row in payload.pop("initialization_rows")
        )
        evidence = PreCalibrationCpuEvidenceV2(
            plan_rows=plan_rows,
            arm_metrics=tuple(arm_metrics),
            initialization_rows=initialization_rows,
            **payload,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error("pre-calibration CPU evidence payload is invalid") from error
    if envelope.get("receipt_sha256") != evidence.receipt_sha256:
        raise GTokCampaignV2Error("pre-calibration CPU evidence identity drifted")
    return evidence


def validate_precalibration_generation_v2(
    evidence: PreCalibrationCpuEvidenceV2,
    *,
    offline_parent_receipt_path: Path,
) -> None:
    """Authenticate the parent-owned offline process that minted CPU evidence."""

    if not isinstance(evidence, PreCalibrationCpuEvidenceV2):
        raise TypeError("pre-calibration generation validation requires typed evidence")
    try:
        receipt, physical_sha256 = load_offline_parent_receipt_v2(
            offline_parent_receipt_path
        )
        expected_script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "precompute_weft1_gtok_cpu_v2.py"
        ).resolve(strict=True)
        observed_script = assert_no_symlink_ancestors(
            Path(receipt.campaign_script)
        ).resolve(strict=True)
    except (GTokOfflineV2Error, OSError, TypeError, ValueError) as error:
        raise GTokCampaignV2Error(
            "pre-calibration offline generation receipt is absent or invalid"
        ) from error
    script_sha256 = hashlib.sha256(observed_script.read_bytes()).hexdigest()
    if (
        observed_script != expected_script
        or script_sha256 != receipt.campaign_script_sha256
        or evidence.generator_script_sha256 != script_sha256
        or evidence.offline_network_receipt_sha256 != physical_sha256
        or evidence.offline_network_policy_sha256 != receipt.policy_sha256
    ):
        raise GTokCampaignV2Error(
            "pre-calibration evidence differs from its authenticated offline generator"
        )


def _projection_from_measurement_v2(
    *,
    scope: str,
    vocab_size: int,
    attempt_id: str,
    measurement: CalibrationMeasurementV2,
    charged_a100_microseconds: int,
) -> ArmCalibrationProjectionV2:
    projected_training = math.ceil(
        measurement.measured_a100_microseconds
        * measurement.planned_tokens_per_run
        / measurement.measured_tokens
    )
    projected = projected_training + (
        measurement.measured_heldout_evaluation_a100_microseconds
        * measurement.heldout_evaluations_per_full_run
    ) + (
        measurement.measured_output_surface_a100_microseconds
        * measurement.output_surface_benchmarks_per_full_run
    )
    return ArmCalibrationProjectionV2(
        scope=scope,
        vocab_size=vocab_size,
        calibration_attempt_id=attempt_id,
        calibration_steps=measurement.steps,
        measured_tokens=measurement.measured_tokens,
        measured_a100_microseconds=measurement.measured_a100_microseconds,
        planned_tokens_per_run=measurement.planned_tokens_per_run,
        projected_run_a100_microseconds=projected,
        charged_calibration_a100_microseconds=charged_a100_microseconds,
        measured_heldout_evaluation_a100_microseconds=(
            measurement.measured_heldout_evaluation_a100_microseconds
        ),
        heldout_evaluations_per_full_run=(
            measurement.heldout_evaluations_per_full_run
        ),
        measured_output_surface_a100_microseconds=(
            measurement.measured_output_surface_a100_microseconds
        ),
        output_surface_benchmarks_per_full_run=(
            measurement.output_surface_benchmarks_per_full_run
        ),
    )


def _default_calibration_executor(
    *,
    source: V4CorpusSourceV2,
    device: torch.device,
    microbatch_sequences: int,
) -> CalibrationExecutorV2:
    def execute(
        *,
        vocab_size: int,
        tokenizer: Tokenizer,
        plan: TrainingPlanV2,
        initialization_seed: int,
        run_seed: int,
        document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    ) -> CalibrationMeasurementV2:
        model = build_gtok_proxy_model_v2(
            vocab_size=vocab_size,
            initialization_seed=initialization_seed,
            run_seed=run_seed,
        )
        try:
            return calibrate_arm_v2(
                model=model,
                tokenizer=tokenizer,
                document_factory=document_factory,
                plan=plan,
                device=device,
                microbatch_sequences=microbatch_sequences,
                heldout_factory=source.heldout_documents,
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return execute


def _default_full_executor(
    *,
    corpus: FrozenScreenCorpusV2,
    source: V4CorpusSourceV2,
    device: torch.device,
    microbatch_sequences: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    gpu_uuid_provenance: str,
    tokenizer_corpus_metrics_by_vocab: Mapping[int, TokenizerCorpusMetricsV2],
) -> FullRunExecutorV2:
    def execute(
        *,
        vocab_size: int,
        seed: int,
        tokenizer: Tokenizer,
        tokenizer_receipt: TokenizerArmReceiptV2,
        plan: TrainingPlanV2,
        initialization_seed: int,
        data_order_seed: int,
        data_order_sha256: str,
        compute_attempt_id: str,
        watchdog_limit_a100_microseconds: int,
        prior_campaign_a100_microseconds: int,
        document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    ) -> FullRunMeasurementV2:
        model = build_gtok_proxy_model_v2(
            vocab_size=vocab_size,
            initialization_seed=initialization_seed,
            run_seed=seed,
        )
        try:
            return execute_full_run_v2(
                model=model,
                tokenizer=tokenizer,
                tokenizer_receipt=tokenizer_receipt,
                corpus=corpus,
                document_factory=document_factory,
                heldout_factory=source.heldout_documents,
                plan=plan,
                seed=seed,
                initialization_seed=initialization_seed,
                data_order_seed=data_order_seed,
                data_order_sha256=data_order_sha256,
                compute_attempt_id=compute_attempt_id,
                training_runtime_receipt_sha256=training_runtime_receipt_sha256,
                code_closure_receipt_sha256=code_closure_receipt_sha256,
                gpu_uuid_provenance=gpu_uuid_provenance,
                watchdog_limit_a100_microseconds=watchdog_limit_a100_microseconds,
                prior_campaign_a100_microseconds=prior_campaign_a100_microseconds,
                campaign_tripwire_a100_microseconds=GTOK_TRIPWIRE_A100_MICROSECONDS,
                device=device,
                microbatch_sequences=microbatch_sequences,
                tokenizer_corpus_metrics=tokenizer_corpus_metrics_by_vocab[vocab_size],
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return execute


def _write_stop(
    root: Path,
    *,
    reason: str,
    cumulative: int,
    attempts: tuple[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2, ...
    ],
    pending: tuple[str, ...],
    running: tuple[str, ...],
    calibration_projection_evidence_receipt_sha256: str | None = None,
    calibration_projection_evidence_physical_sha256: str | None = None,
) -> CampaignStopArtifactV2:
    if attempts:
        validate_sqlite_event_ledger_v2(root, attempts)
    active = tuple(sorted(set(pending) | set(running)))
    artifact = CampaignStopArtifactV2(
        reason=reason,
        cumulative_a100_microseconds=cumulative,
        attempts=attempts,
        pending_attempt_ids=pending,
        running_attempt_ids=running,
        hard_abort_attempt_ids=active,
        hard_abort_and_report=True,
        return_to_strategy=True,
        calibration_projection_evidence_receipt_sha256=(
            calibration_projection_evidence_receipt_sha256
        ),
        calibration_projection_evidence_physical_sha256=(
            calibration_projection_evidence_physical_sha256
        ),
    )
    _exclusive_write(
        root / "campaign-stop.json",
        {
            "payload": asdict(artifact),
            "schema": "weft1_gtok_v2_campaign_stop",
        },
    )
    return artifact


def run_base_campaign_v2(
    *,
    corpus: FrozenScreenCorpusV2,
    source: V4CorpusSourceV2,
    tokenizer_arms: tuple[TokenizerExecutionArmV2, ...],
    seeds: tuple[int, int],
    initialization_seeds: tuple[int, int],
    data_order_seeds: tuple[int, int],
    output_root: Path,
    device: torch.device,
    microbatch_sequences: int,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    code_closure_receipt: GTokCodeClosureReceiptV2 | None = None,
    repository_root: Path | None = None,
    offline_network_receipt_sha256: str | None = None,
    offline_network_policy_sha256: str | None = None,
    gpu_uuid_provenance: str | None = None,
    precalibration_cpu_evidence: PreCalibrationCpuEvidenceV2 | None = None,
    precalibration_offline_parent_receipt_path: Path | None = None,
    cpu_runtime_identity_sha256: str | None = None,
    calibration_executor: CalibrationExecutorV2 | None = None,
    full_run_executor: FullRunExecutorV2 | None = None,
) -> BaseCampaignResultV2 | DryRunCampaignResultV2:
    """Execute the four calibrations then the exact eight-run base matrix."""

    if not isinstance(corpus, FrozenScreenCorpusV2):
        raise TypeError("campaign requires a frozen P-B corpus")
    for name, value in (
        ("training-runtime", training_runtime_receipt_sha256),
        ("code-closure", code_closure_receipt_sha256),
    ):
        if len(value) != 64 or any(character not in _HEX for character in value):
            raise GTokCampaignV2Error(f"campaign requires one exact {name} receipt")
    if source.physical_d6_evidence_sha256 != corpus.d6_physical_evidence_sha256:
        raise GTokCampaignV2Error("physical source differs from frozen D6 evidence")
    if source.training_raw_bytes != corpus.training_realized_bytes:
        raise GTokCampaignV2Error("physical T bytes differ from the P-B freeze")
    if source.heldout_raw_bytes_by_stratum != corpus.heldout_denominator_signature:
        raise GTokCampaignV2Error("physical H strata differ from the P-B freeze")
    if tuple(arm.receipt.vocab_size for arm in tokenizer_arms) != GTOK_VOCABULARY_ARMS:
        raise GTokCampaignV2Error("campaign tokenizer arms are not canonical")
    if any(
        arm.receipt.full_corpus_manifest_sha256
        != corpus.full_corpus_manifest_sha256
        or arm.receipt.fit_stream_sha256 != corpus.training_stream_sha256
        for arm in tokenizer_arms
    ):
        raise GTokCampaignV2Error("tokenizer panel differs from the frozen T stream")
    if tuple(seeds) != GTOK_GOVERNED_TRAINING_SEEDS_V2:
        raise GTokCampaignV2Error("training seeds differ from the A2 governed rows")
    if tuple(initialization_seeds) != GTOK_GOVERNED_INITIALIZATION_SEEDS_V2:
        raise GTokCampaignV2Error("initialization seeds differ from the A2 derivation")
    if tuple(data_order_seeds) != GTOK_GOVERNED_DATA_ORDER_SEEDS_V2:
        raise GTokCampaignV2Error("data-order seeds differ from the A2 derivation")
    if len(set(seeds)) != GTOK_SEED_COUNT:
        raise GTokCampaignV2Error("campaign requires two distinct governed seed rows")
    if tuple(seeds) != tuple(
        seed for seed, _, _ in source.training_order_receipts
    ):
        raise GTokCampaignV2Error("campaign seeds differ from physical V4 consumer orders")
    physical_data_seeds = tuple(
        data_seed for _, data_seed, _ in source.training_order_receipts
    )
    if physical_data_seeds != tuple(data_order_seeds):
        raise GTokCampaignV2Error(
            "campaign data-order seeds differ from physical V4 consumer orders"
        )
    if (calibration_executor is None) != (full_run_executor is None):
        raise GTokCampaignV2Error(
            "calibration and full-run executors must be both physical or both injected"
        )
    authoritative_execution = calibration_executor is None
    replay_plan_bindings: tuple[DeterminismReplayPlanBindingV2, ...] = ()
    replay_plan_set_sha256: str | None = None
    if authoritative_execution:
        require_resolved_confirmation_semantics_v2()
        if microbatch_sequences != GTOK_MICROBATCH_SEQUENCES_V2:
            raise GTokCampaignV2Error(
                "authoritative P-C requires microbatch_sequences=8"
            )
        if (
            not isinstance(offline_network_receipt_sha256, str)
            or len(offline_network_receipt_sha256) != 64
            or any(
                character not in _HEX
                for character in offline_network_receipt_sha256
            )
        ):
            raise GTokCampaignV2Error(
                "authoritative P-C requires a parent-probed offline receipt"
            )
        if (
            not isinstance(offline_network_policy_sha256, str)
            or len(offline_network_policy_sha256) != 64
            or any(
                character not in _HEX
                for character in offline_network_policy_sha256
            )
        ):
            raise GTokCampaignV2Error(
                "authoritative P-C requires a stable offline isolation policy"
            )
        if not isinstance(code_closure_receipt, GTokCodeClosureReceiptV2):
            raise GTokCampaignV2Error(
                "authoritative execution requires the full code-closure receipt"
            )
        if code_closure_receipt.receipt_sha256 != code_closure_receipt_sha256:
            raise GTokCampaignV2Error("code-closure payload differs from its identity")
        if not isinstance(repository_root, Path):
            raise TypeError("authoritative execution requires pathlib repository_root")
        if (
            not isinstance(gpu_uuid_provenance, str)
            or not gpu_uuid_provenance.startswith("GPU-")
            or len(gpu_uuid_provenance) <= 4
        ):
            raise GTokCampaignV2Error(
                "authoritative P-C requires the physical NVIDIA GPU UUID as provenance"
            )
        if not isinstance(precalibration_cpu_evidence, PreCalibrationCpuEvidenceV2):
            raise GTokCampaignV2Error(
                "authoritative P-C requires a separately materialized CPU precompute receipt"
            )
        if not isinstance(precalibration_offline_parent_receipt_path, Path):
            raise TypeError(
                "authoritative P-C requires the precompute offline parent receipt path"
            )
        validate_precalibration_generation_v2(
            precalibration_cpu_evidence,
            offline_parent_receipt_path=precalibration_offline_parent_receipt_path,
        )
        if (
            not isinstance(cpu_runtime_identity_sha256, str)
            or cpu_runtime_identity_sha256
            != precalibration_cpu_evidence.cpu_runtime_identity_sha256
        ):
            raise GTokCampaignV2Error(
                "A100 training venv differs from CPU precompute environment"
            )
        if (
            precalibration_cpu_evidence.frozen_screen_corpus_sha256
            != corpus.receipt_sha256
            or precalibration_cpu_evidence.code_closure_receipt_sha256
            != code_closure_receipt_sha256
            or precalibration_cpu_evidence.microbatch_sequences
            != microbatch_sequences
            or tuple(
                row.tokenizer_receipt_sha256
                for row in precalibration_cpu_evidence.arm_metrics
            )
            != tuple(arm.receipt.receipt_sha256 for arm in tokenizer_arms)
        ):
            raise GTokCampaignV2Error(
                "CPU precompute receipt differs from this corpus/code/tokenizer launch"
            )
        replay_plan_bindings = canonical_determinism_replay_plan_bindings_v2(
            {
                (row.vocab_size, row.training_seed): row.plan
                for row in precalibration_cpu_evidence.plan_rows
            },
            vocabularies=GTOK_VOCABULARY_ARMS,
            training_seeds=tuple(seeds),
            initialization_seeds=tuple(initialization_seeds),
        )
        replay_plan_set_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_precalibration_replay_plan_set",
            replay_plan_bindings,
        )

        def revalidate_code_closure() -> None:
            validate_gtok_code_closure_v2(
                code_closure_receipt,
                repository_root=repository_root,
            )

        revalidate_code_closure()
    else:
        def revalidate_code_closure() -> None:
            return None
    root = assert_no_symlink_ancestors(output_root)
    resuming = root.exists()
    if root.exists():
        if not authoritative_execution:
            raise FileExistsError("synthetic campaign output root must be new")
        root = root.resolve(strict=True)
        if (root / "campaign-stop.json").exists():
            raise GTokV2Stop(
                "a governed STOP artifact exists; automatic resume is prohibited"
            )
        if not _lifecycle_ledger_path(root).exists():
            raise GTokCampaignV2Error(
                "existing campaign root lacks the durable lifecycle ledger"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
    root = root.resolve(strict=True)
    authority_payload = {
        "authority_status": (
            "AUTHORITATIVE_PHYSICAL_DEFAULT_EXECUTORS"
            if authoritative_execution
            else "NON_AUTHORITATIVE_INJECTED_EXECUTORS"
        ),
        "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
        "code_closure_receipt_sha256": code_closure_receipt_sha256,
        "corpus_receipt_sha256": corpus.receipt_sha256,
        "governed_seed_rows": GTOK_GOVERNED_SEED_ROWS_V2,
        "microbatch_sequences": microbatch_sequences,
        "gradient_accumulation_slices": GTOK_ACCUMULATION_SLICES_V2,
        "offline_network_policy_sha256": offline_network_policy_sha256,
        "gpu_uuid_policy": "RECORDED_PER_ATTEMPT_NOT_RUNTIME_IDENTITY",
        "cpu_runtime_identity_sha256": cpu_runtime_identity_sha256,
        "tokenizer_receipt_sha256s": tuple(
            arm.receipt.receipt_sha256 for arm in tokenizer_arms
        ),
        "precalibration_cpu_evidence_receipt_sha256": (
            None
            if precalibration_cpu_evidence is None
            else precalibration_cpu_evidence.receipt_sha256
        ),
        "precalibration_determinism_policy_sha256": (
            CUDA_DETERMINISM_POLICY_SHA256_V2
            if authoritative_execution
            else None
        ),
        "precalibration_replay_plan_binding_sha256s": tuple(
            row.receipt_sha256 for row in replay_plan_bindings
        ),
        "precalibration_replay_plan_set_sha256": replay_plan_set_sha256,
        "training_runtime_receipt_sha256": training_runtime_receipt_sha256,
    }
    authority_envelope = {
        "payload": authority_payload,
        "receipt_sha256": gtok_v2_bound_sha256(
            "weft1_gtok_v2_campaign_authority_header",
            authority_payload,
        ),
        "schema": "weft1_gtok_v2_campaign_authority_header",
    }
    authority_path = root / "campaign-authority.json"
    if authority_path.exists():
        raw, stored = load_canonical_json_snapshot(authority_path)
        if raw != canonical_json_bytes(authority_envelope) + b"\n":
            raise GTokCampaignV2Error(
                "campaign authority header differs; authority cannot be upgraded or substituted"
            )
    else:
        _exclusive_write(authority_path, authority_envelope)
    if resuming:
        recover_orphaned_lifecycle_attempts_v2(root)
    initialization_equality: InitializationEqualityEvidenceV2 | None = None
    if authoritative_execution:
        revalidate_code_closure()
        try:
            initialization_equality = InitializationEqualityEvidenceV2(
                rows=precalibration_cpu_evidence.initialization_rows,
                training_runtime_receipt_sha256=training_runtime_receipt_sha256,
                code_closure_receipt_sha256=code_closure_receipt_sha256,
                offline_network_receipt_sha256=str(offline_network_policy_sha256),
            )
            _persist_initialization_equality_v2(root, initialization_equality)
        except (GTokCampaignV2Error, TypeError, ValueError) as error:
            pending = tuple(
                _attempt_id("calibration", arm) for arm in GTOK_VOCABULARY_ARMS
            ) + tuple(
                _attempt_id("run", arm, seed)
                for arm in GTOK_VOCABULARY_ARMS
                for seed in seeds
            )
            _write_stop(
                root,
                reason="INITIALIZATION_EQUALITY_PREFLIGHT_FAILED",
                cumulative=0,
                attempts=(),
                pending=pending,
                running=(),
            )
            raise GTokV2Stop(
                "non-vocabulary initialization differs before any calibration"
            ) from error
        revalidate_code_closure()

    def validate_full_initialization(
        measurement: FullRunMeasurementV2,
        *,
        vocab_size: int,
        training_seed: int,
    ) -> None:
        if initialization_equality is None:
            return
        _validate_full_initialization_v2(
            measurement,
            evidence=initialization_equality,
            vocab_size=vocab_size,
            training_seed=training_seed,
        )

    def validate_calibration_prefix(
        measurement: CalibrationMeasurementV2,
        *,
        vocab_size: int,
    ) -> CalibrationMeasurementV2:
        if authoritative_execution:
            validate_calibration_prefix_v2(
                measurement,
                plan=plans[(vocab_size, seeds[0])],
            )
        return measurement

    tokenizers = {arm.receipt.vocab_size: arm.load() for arm in tokenizer_arms}
    arm_receipts = {arm.receipt.vocab_size: arm.receipt for arm in tokenizer_arms}
    orders = {
        training_seed: order_sha256
        for training_seed, _, order_sha256 in source.training_order_receipts
    }
    plans: dict[tuple[int, int], TrainingPlanV2] = {}
    tokenizer_corpus_metrics_by_vocab: dict[int, TokenizerCorpusMetricsV2] = {}
    precalibration_evidence = precalibration_cpu_evidence
    if authoritative_execution:
        if not isinstance(precalibration_evidence, PreCalibrationCpuEvidenceV2):
            raise GTokCampaignV2Error(
                "authoritative P-C requires the durable CPU precompute evidence"
            )
        if (
            precalibration_evidence.frozen_screen_corpus_sha256
            != corpus.receipt_sha256
            or precalibration_evidence.code_closure_receipt_sha256
            != code_closure_receipt_sha256
        ):
            raise GTokCampaignV2Error(
                "CPU precompute evidence differs from corpus or code closure"
            )
        if any(
            row.tokenizer_receipt_sha256
            != arm_receipts[row.vocab_size].receipt_sha256
            for row in precalibration_evidence.arm_metrics
        ):
            raise GTokCampaignV2Error(
                "CPU precompute evidence differs from tokenizer panel"
            )
        plans = {
            (row.vocab_size, row.training_seed): row.plan
            for row in precalibration_evidence.plan_rows
        }
        tokenizer_corpus_metrics_by_vocab = {
            row.vocab_size: row.tokenizer_corpus
            for row in precalibration_evidence.arm_metrics
        }
        if any(
            plan.realized_raw_bytes != corpus.training_realized_bytes
            for plan in plans.values()
        ):
            raise GTokCampaignV2Error(
                "CPU precompute plans differ from frozen realized T bytes"
            )
        _persist_precalibration_cpu_evidence_v2(root, precalibration_evidence)
        revalidate_code_closure()
    else:
        for vocab_size in GTOK_VOCABULARY_ARMS:
            tokenizer = tokenizers[vocab_size]
            for seed in seeds:
                plans[(vocab_size, seed)] = plan_training_stream_v2(
                    lambda seed=seed: source.training_documents(seed),
                    tokenizer=tokenizer,
                    expected_realized_raw_bytes=corpus.training_realized_bytes,
                )
            if len({plans[(vocab_size, seed)].optimizer_steps for seed in seeds}) != 1:
                raise GTokCampaignV2Error("seed order changed an arm's total optimizer steps")

    calibration_executor = calibration_executor or _default_calibration_executor(
        source=source,
        device=device,
        microbatch_sequences=microbatch_sequences,
    )
    full_run_executor = full_run_executor or _default_full_executor(
        corpus=corpus,
        source=source,
        device=device,
        microbatch_sequences=microbatch_sequences,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        gpu_uuid_provenance=str(gpu_uuid_provenance),
        tokenizer_corpus_metrics_by_vocab=tokenizer_corpus_metrics_by_vocab,
    )
    attempts: list[
        ComputeAttemptReceiptV2 | PrecalibrationReplayAttemptReceiptV2
    ] = list(_load_persisted_attempts_v2(root))
    precalibration_determinism_authority: (
        PrecalibrationDeterminismAuthorityV2 | None
    ) = None
    precalibration_replay_receipts: tuple[
        PrecalibrationDeterminismReplayReceiptV2, ...
    ] = ()
    if authoritative_execution:
        if not replay_plan_bindings or replay_plan_set_sha256 is None:
            raise GTokCampaignV2Error(
                "authoritative P-C lacks its canonical replay plan set"
            )
        (
            precalibration_determinism_authority,
            precalibration_replay_receipts,
        ) = _run_precalibration_determinism_gate_v2(
            root=root,
            replay_plan_bindings=replay_plan_bindings,
            plans=plans,
            device=device,
            microbatch_sequences=microbatch_sequences,
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
            gpu_uuid_provenance=str(gpu_uuid_provenance),
            offline_network_receipt_sha256=str(offline_network_receipt_sha256),
            attempts=attempts,
            revalidate_code_closure=revalidate_code_closure,
            training_seeds=tuple(seeds),
        )
        if (
            precalibration_determinism_authority.replay_plan_set_sha256
            != replay_plan_set_sha256
        ):
            raise GTokCampaignV2Error(
                "executed replay authority differs from campaign plan-set header"
            )
    lifecycle_events = (
        validate_lifecycle_ledger_v2(root)
        if _lifecycle_ledger_path(root).exists()
        else ()
    )
    projections: list[ArmCalibrationProjectionV2] = []
    calibration_measurements: dict[int, CalibrationMeasurementV2] = {}
    for vocab_size in GTOK_VOCABULARY_ARMS:
        revalidate_code_closure()
        logical_attempt_id = _attempt_id("calibration", vocab_size)
        lifecycle_events = (
            validate_lifecycle_ledger_v2(root)
            if _lifecycle_ledger_path(root).exists()
            else ()
        )
        completed_events = tuple(
            event
            for event in lifecycle_events
            if event.logical_attempt_id == logical_attempt_id
            and event.phase == "TERMINAL"
            and event.terminal_status == "completed"
        )
        if len(completed_events) > 1:
            raise GTokCampaignV2Error("calibration logical attempt completed twice")
        if completed_events:
            completed = completed_events[0]
            measurement = validate_calibration_prefix(
                _measurement_from_lifecycle_v2(completed),
                vocab_size=vocab_size,
            )
            calibration_measurements[vocab_size] = measurement
            projection = _projection_from_measurement_v2(
                scope="base_screen",
                vocab_size=vocab_size,
                attempt_id=completed.attempt_id,
                measurement=measurement,
                charged_a100_microseconds=completed.charged_a100_microseconds,
            )
            for orphan in tuple(
                event
                for event in lifecycle_events
                if event.logical_attempt_id == logical_attempt_id
                and event.phase == "TERMINAL"
                and event.terminal_status == "preempted"
                and event.attempt_id not in {row.attempt_id for row in attempts}
            ):
                recovered_attempt = ComputeAttemptReceiptV2(
                    attempt_id=orphan.attempt_id,
                    scope="base_screen",
                    kind="calibration",
                    vocab_size=vocab_size,
                    seed=None,
                    consumed_a100_microseconds=orphan.charged_a100_microseconds,
                    status="preempted",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=(
                        projection.projected_run_a100_microseconds
                    ),
                    watchdog_limit_a100_microseconds=(
                        2 * projection.projected_run_a100_microseconds
                    ),
                )
                attempts.append(recovered_attempt)
                _persist_attempt(root, len(attempts) - 1, recovered_attempt)
            existing = next(
                (
                    row
                    for row in attempts
                    if row.attempt_id == completed.attempt_id
                ),
                None,
            )
            if existing is None:
                existing = ComputeAttemptReceiptV2(
                    attempt_id=completed.attempt_id,
                    scope="base_screen",
                    kind="calibration",
                    vocab_size=vocab_size,
                    seed=None,
                    consumed_a100_microseconds=completed.charged_a100_microseconds,
                    status="completed",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=(
                        projection.projected_run_a100_microseconds
                    ),
                    watchdog_limit_a100_microseconds=(
                        2 * projection.projected_run_a100_microseconds
                    ),
                )
                attempts.append(existing)
                _persist_attempt(root, len(attempts) - 1, existing)
            elif (
                existing.status != "completed"
                or existing.calibration_projection_sha256
                != projection.receipt_sha256
            ):
                raise GTokCampaignV2Error(
                    "completed calibration differs from terminal attempt ledger"
                )
            projections.append(projection)
            continue
        attempt_id = _next_physical_attempt_id_v2(
            logical_attempt_id,
            lifecycle_events,
        )
        calibration_started = time.perf_counter_ns()
        try:
            measurement, lifecycle_charge = _execute_with_lifecycle_v2(
                root=root,
                logical_attempt_id=logical_attempt_id,
                attempt_id=attempt_id,
                scope="base_screen",
                kind="calibration",
                operation=lambda vocab_size=vocab_size: validate_calibration_prefix(
                    calibration_executor(
                        vocab_size=vocab_size,
                        tokenizer=tokenizers[vocab_size],
                        plan=plans[(vocab_size, seeds[0])],
                        initialization_seed=initialization_seeds[0],
                        run_seed=seeds[0],
                        document_factory=(
                            lambda seed=seeds[0]: source.training_documents(seed)
                        ),
                    ),
                    vocab_size=vocab_size,
                ),
                success_charge=lambda value: value.charged_a100_microseconds,
                success_payload=lambda value: asdict(value),
                gpu_uuid_provenance=gpu_uuid_provenance,
                offline_network_launch_receipt_sha256=(
                    offline_network_receipt_sha256
                ),
            )
        except BaseException as error:
            consumed = int(
                getattr(
                    error,
                    "_gtok_lifecycle_charge_v2",
                    max(
                        1,
                        math.ceil(
                            (time.perf_counter_ns() - calibration_started) / 1_000
                        ),
                    ),
                )
            )
            failure_projection_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "attempt_id": attempt_id,
                        "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
                        "failure_type": type(error).__name__,
                        "status": "FAILED_BEFORE_PREFLIGHT_PROJECTION",
                    }
                )
            ).hexdigest()
            attempt = ComputeAttemptReceiptV2(
                attempt_id=attempt_id,
                scope="base_screen",
                kind="calibration",
                vocab_size=vocab_size,
                seed=None,
                consumed_a100_microseconds=consumed,
                status="failed",
                calibration_projection_sha256=failure_projection_sha256,
                projected_run_a100_microseconds=consumed,
                watchdog_limit_a100_microseconds=2 * consumed,
            )
            attempts.append(attempt)
            _persist_attempt(root, len(attempts) - 1, attempt)
            remaining = tuple(
                _attempt_id("calibration", later)
                for later in GTOK_VOCABULARY_ARMS
                if later > vocab_size
            ) + tuple(
                _attempt_id("run", arm, seed)
                for arm in GTOK_VOCABULARY_ARMS
                for seed in seeds
            )
            _write_stop(
                root,
                reason=f"CALIBRATION_FAILED:{type(error).__name__}",
                cumulative=sum(item.consumed_a100_microseconds for item in attempts),
                attempts=tuple(attempts),
                pending=remaining,
                running=(),
            )
            raise
        projection = _projection_from_measurement_v2(
            scope="base_screen",
            vocab_size=vocab_size,
            attempt_id=attempt_id,
            measurement=measurement,
            charged_a100_microseconds=lifecycle_charge,
        )
        calibration_measurements[vocab_size] = measurement
        projections.append(projection)
        for orphan in tuple(
            event
            for event in lifecycle_events
            if event.logical_attempt_id == logical_attempt_id
            and event.phase == "TERMINAL"
            and event.terminal_status == "preempted"
            and event.attempt_id not in {row.attempt_id for row in attempts}
        ):
            recovered_attempt = ComputeAttemptReceiptV2(
                attempt_id=orphan.attempt_id,
                scope="base_screen",
                kind="calibration",
                vocab_size=vocab_size,
                seed=None,
                consumed_a100_microseconds=orphan.charged_a100_microseconds,
                status="preempted",
                calibration_projection_sha256=projection.receipt_sha256,
                projected_run_a100_microseconds=(
                    projection.projected_run_a100_microseconds
                ),
                watchdog_limit_a100_microseconds=(
                    2 * projection.projected_run_a100_microseconds
                ),
            )
            attempts.append(recovered_attempt)
            _persist_attempt(root, len(attempts) - 1, recovered_attempt)
        attempt = ComputeAttemptReceiptV2(
            attempt_id=attempt_id,
            scope="base_screen",
            kind="calibration",
            vocab_size=vocab_size,
            seed=None,
            consumed_a100_microseconds=lifecycle_charge,
            status="completed",
            calibration_projection_sha256=projection.receipt_sha256,
            projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
            watchdog_limit_a100_microseconds=(
                GTOK_PER_RUN_WATCHDOG_MULTIPLIER
                * projection.projected_run_a100_microseconds
            ),
        )
        attempts.append(attempt)
        _persist_attempt(root, len(attempts) - 1, attempt)
        calibration_cumulative = sum(
            item.consumed_a100_microseconds for item in attempts
        )
        if calibration_cumulative > GTOK_TRIPWIRE_A100_MICROSECONDS:
            pending = tuple(
                _attempt_id("calibration", later)
                for later in GTOK_VOCABULARY_ARMS
                if later > vocab_size
            ) + tuple(
                _attempt_id("run", arm, seed)
                for arm in GTOK_VOCABULARY_ARMS
                for seed in seeds
            )
            _write_stop(
                root,
                reason="CUMULATIVE_TRIPWIRE_DURING_CALIBRATION",
                cumulative=calibration_cumulative,
                attempts=tuple(attempts),
                pending=pending,
                running=(),
            )
            raise GTokV2Stop("calibration crossed 12 A100-hours; return to strategy")
    if initialization_equality is not None:
        seed_zero = initialization_equality.rows[0]
        expected_hashes = {
            row.vocab_size: row.shared_nonvocabulary_state_sha256
            for row in seed_zero.arms
        }
        if any(
            calibration_measurements[vocab_size].shared_initial_state_sha256
            != expected_hashes[vocab_size]
            for vocab_size in GTOK_VOCABULARY_ARMS
        ):
            pending = tuple(
                _attempt_id("run", arm, seed)
                for arm in GTOK_VOCABULARY_ARMS
                for seed in seeds
            )
            _write_stop(
                root,
                reason="INITIALIZATION_EQUALITY_DRIFT_DURING_CALIBRATION",
                cumulative=sum(item.consumed_a100_microseconds for item in attempts),
                attempts=tuple(attempts),
                pending=pending,
                running=(),
            )
            raise GTokV2Stop(
                "calibration model initialization differs from pre-spend evidence"
            )
    projection_evidence, projection_evidence_physical = (
        _persist_calibration_projection_evidence(
            root,
            projections=tuple(projections),
            measurements=tuple(
                calibration_measurements[vocab_size]
                for vocab_size in GTOK_VOCABULARY_ARMS
            ),
            attempts=tuple(
                item
                for item in attempts
                if isinstance(item, ComputeAttemptReceiptV2)
                and item.kind == "calibration"
            ),
            precalibration_determinism_authority=(
                precalibration_determinism_authority
            ),
        )
    )
    selected_calibration_ids = {
        item.calibration_attempt_id for item in projections
    }
    recovered_calibration_charge = sum(
        item.consumed_a100_microseconds
        for item in attempts
        if item.kind == "calibration"
        and item.attempt_id not in selected_calibration_ids
    )
    try:
        preflight = build_preflight_projection_v2(
            tuple(projections),
            recovered_attempt_a100_microseconds=recovered_calibration_charge,
            precalibration_replay_attempts=tuple(
                item
                for item in attempts
                if isinstance(item, PrecalibrationReplayAttemptReceiptV2)
            ),
            precalibration_replay_plan_set_sha256=(
                None
                if precalibration_determinism_authority is None
                else precalibration_determinism_authority.replay_plan_set_sha256
            ),
            precalibration_replay_receipt_sha256s=tuple(
                row.receipt_sha256 for row in precalibration_replay_receipts
            ),
            precalibration_replay_authority_sha256=(
                None
                if precalibration_determinism_authority is None
                else precalibration_determinism_authority.receipt_sha256
            ),
        )
    except GTokV2Stop as error:
        pending = tuple(
            _attempt_id("run", arm, seed)
            for arm in GTOK_VOCABULARY_ARMS
            for seed in seeds
        )
        _write_stop(
            root,
            reason="PREFLIGHT_PROJECTION_EXCEEDS_12_A100_HOURS",
            cumulative=sum(item.consumed_a100_microseconds for item in attempts),
            attempts=tuple(attempts),
            pending=pending,
            running=(),
            calibration_projection_evidence_receipt_sha256=(
                projection_evidence.receipt_sha256
            ),
            calibration_projection_evidence_physical_sha256=(
                projection_evidence_physical
            ),
        )
        raise GTokV2Stop(
            "preflight projection exceeds 12 A100-hours; no full run launched"
        ) from error
    preflight_envelope = {
        "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
        "payload": asdict(preflight),
        "calibration_projection_evidence_physical_sha256": (
            projection_evidence_physical
        ),
        "calibration_projection_evidence_receipt_sha256": (
            projection_evidence.receipt_sha256
        ),
        "receipt_sha256": preflight.receipt_sha256,
        "schema": "weft1_gtok_v2_preflight_projection_event",
    }
    preflight_path = root / "preflight-projection.json"
    if preflight_path.exists():
        raw, stored = load_canonical_json_snapshot(preflight_path)
        if raw != canonical_json_bytes(preflight_envelope) + b"\n":
            raise GTokCampaignV2Error("stored preflight differs on resume")
    else:
        _exclusive_write(preflight_path, preflight_envelope)

    runs: list[GTokRunReceiptV2] = []
    measurements: list[FullRunMeasurementV2] = []
    cumulative = sum(item.consumed_a100_microseconds for item in attempts)
    pending = tuple(
        _attempt_id("run", vocab_size, seed)
        for vocab_size in GTOK_VOCABULARY_ARMS
        for seed in seeds
    )
    for vocab_size in GTOK_VOCABULARY_ARMS:
        projection = next(item for item in projections if item.vocab_size == vocab_size)
        for seed_index, seed in enumerate(seeds):
            revalidate_code_closure()
            logical_attempt_id = _attempt_id("run", vocab_size, seed)
            lifecycle_events = validate_lifecycle_ledger_v2(root)
            completed_events = tuple(
                event
                for event in lifecycle_events
                if event.logical_attempt_id == logical_attempt_id
                and event.phase == "TERMINAL"
                and event.terminal_status == "completed"
            )
            if len(completed_events) > 1:
                raise GTokCampaignV2Error("full-run logical attempt completed twice")
            for orphan in tuple(
                event
                for event in lifecycle_events
                if event.logical_attempt_id == logical_attempt_id
                and event.phase == "TERMINAL"
                and event.terminal_status == "preempted"
                and event.attempt_id not in {row.attempt_id for row in attempts}
            ):
                recovered_attempt = ComputeAttemptReceiptV2(
                    attempt_id=orphan.attempt_id,
                    scope="base_screen",
                    kind="full_run",
                    vocab_size=vocab_size,
                    seed=seed,
                    consumed_a100_microseconds=orphan.charged_a100_microseconds,
                    status="preempted",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=(
                        projection.projected_run_a100_microseconds
                    ),
                    watchdog_limit_a100_microseconds=(
                        2 * projection.projected_run_a100_microseconds
                    ),
                )
                attempts.append(recovered_attempt)
                _persist_attempt(root, len(attempts) - 1, recovered_attempt)
                cumulative += orphan.charged_a100_microseconds
            recovered_full_charge = sum(
                item.consumed_a100_microseconds
                for item in attempts
                if item.kind == "full_run" and item.status == "preempted"
            )
            if (
                cumulative > GTOK_TRIPWIRE_A100_MICROSECONDS
                or preflight.projected_campaign_a100_microseconds
                + recovered_full_charge
                > GTOK_TRIPWIRE_A100_MICROSECONDS
            ):
                _write_stop(
                    root,
                    reason="RESUME_METER_OR_PROJECTION_EXCEEDS_12_A100_HOURS",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(),
                )
                raise GTokV2Stop(
                    "checkpoint-free resume exceeds the 12 A100-hour tripwire"
                )
            if completed_events:
                completed = completed_events[0]
                measurement = _full_measurement_from_lifecycle_v2(completed)
                validate_full_initialization(
                    measurement,
                    vocab_size=vocab_size,
                    training_seed=seed,
                )
                run = measurement.run
                if (
                    run.vocab_size != vocab_size
                    or run.seed != seed
                    or run.training_runtime_receipt_sha256
                    != training_runtime_receipt_sha256
                    or run.code_closure_receipt_sha256
                    != code_closure_receipt_sha256
                ):
                    raise GTokCampaignV2Error(
                        "completed run differs from its governed campaign row"
                    )
                existing = next(
                    (
                        row
                        for row in attempts
                        if row.attempt_id == completed.attempt_id
                    ),
                    None,
                )
                if existing is None:
                    existing = ComputeAttemptReceiptV2(
                        attempt_id=completed.attempt_id,
                        scope="base_screen",
                        kind="full_run",
                        vocab_size=vocab_size,
                        seed=seed,
                        consumed_a100_microseconds=completed.charged_a100_microseconds,
                        status="completed",
                        calibration_projection_sha256=projection.receipt_sha256,
                        projected_run_a100_microseconds=(
                            projection.projected_run_a100_microseconds
                        ),
                        watchdog_limit_a100_microseconds=(
                            2 * projection.projected_run_a100_microseconds
                        ),
                    )
                    attempts.append(existing)
                    _persist_attempt(root, len(attempts) - 1, existing)
                    cumulative += existing.consumed_a100_microseconds
                elif (
                    existing.status != "completed"
                    or existing.consumed_a100_microseconds
                    != run.measured_a100_microseconds
                ):
                    raise GTokCampaignV2Error(
                        "completed run differs from terminal attempt ledger"
                    )
                runs.append(run)
                measurements.append(measurement)
                pending = tuple(
                    item for item in pending if item != logical_attempt_id
                )
                continue
            attempt_id = _next_physical_attempt_id_v2(
                logical_attempt_id,
                lifecycle_events,
            )
            pending = tuple(item for item in pending if item != logical_attempt_id)
            started = time.perf_counter_ns()

            def execute_full_and_validate_initialization() -> FullRunMeasurementV2:
                value = full_run_executor(
                    vocab_size=vocab_size,
                    seed=seed,
                    tokenizer=tokenizers[vocab_size],
                    tokenizer_receipt=arm_receipts[vocab_size],
                    plan=plans[(vocab_size, seed)],
                    initialization_seed=initialization_seeds[seed_index],
                    data_order_seed=data_order_seeds[seed_index],
                    data_order_sha256=orders[seed],
                    compute_attempt_id=attempt_id,
                    watchdog_limit_a100_microseconds=(
                        GTOK_PER_RUN_WATCHDOG_MULTIPLIER
                        * projection.projected_run_a100_microseconds
                    ),
                    prior_campaign_a100_microseconds=cumulative,
                    document_factory=lambda seed=seed: source.training_documents(seed),
                )
                validate_full_initialization(
                    value,
                    vocab_size=vocab_size,
                    training_seed=seed,
                )
                return value

            try:
                measurement, lifecycle_charge = _execute_with_lifecycle_v2(
                    root=root,
                    logical_attempt_id=logical_attempt_id,
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="full_run",
                    operation=execute_full_and_validate_initialization,
                    success_charge=lambda value: value.run.measured_a100_microseconds,
                    success_payload=lambda value: asdict(value),
                    gpu_uuid_provenance=gpu_uuid_provenance,
                    offline_network_launch_receipt_sha256=(
                        offline_network_receipt_sha256
                    ),
                )
                if measurement.run.measured_a100_microseconds != lifecycle_charge:
                    measurement = replace(
                        measurement,
                        run=replace(
                            measurement.run,
                            measured_a100_microseconds=lifecycle_charge,
                        ),
                    )
                if (
                    measurement.training_runtime_receipt_sha256
                    != training_runtime_receipt_sha256
                ):
                    raise GTokCampaignV2Error(
                        "full-run runtime differs across campaign arms or seeds"
                    )
                if measurement.code_closure_receipt_sha256 != code_closure_receipt_sha256:
                    raise GTokCampaignV2Error(
                        "full-run code closure differs across campaign arms or seeds"
                    )
                if measurement.run.gpu_uuid_provenance != gpu_uuid_provenance:
                    raise GTokCampaignV2Error(
                        "full-run GPU provenance differs from the campaign launch"
                    )
            except GTokRunWatchdogV2 as error:
                consumed = int(
                    getattr(
                        error,
                        "_gtok_lifecycle_charge_v2",
                        error.consumed_a100_microseconds,
                    )
                )
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="full_run",
                    vocab_size=vocab_size,
                    seed=seed,
                    consumed_a100_microseconds=consumed,
                    status="aborted_watchdog",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                    watchdog_limit_a100_microseconds=(
                        2 * projection.projected_run_a100_microseconds
                    ),
                    hard_abort_issued=True,
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += attempt.consumed_a100_microseconds
                _write_stop(
                    root,
                    reason="PER_RUN_WATCHDOG_STRICTLY_ABOVE_2X",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(attempt_id,),
                )
                raise GTokV2Stop("per-run watchdog fired; hard abort and return to strategy") from error
            except GTokCampaignTripwireV2 as error:
                consumed = int(
                    getattr(
                        error,
                        "_gtok_lifecycle_charge_v2",
                        max(1, error.consumed_a100_microseconds - cumulative),
                    )
                )
                watchdog = 2 * projection.projected_run_a100_microseconds
                if consumed > watchdog:
                    status = "aborted_watchdog"
                    hard_abort = True
                else:
                    status = "preempted"
                    hard_abort = False
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="full_run",
                    vocab_size=vocab_size,
                    seed=seed,
                    consumed_a100_microseconds=consumed,
                    status=status,
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                    watchdog_limit_a100_microseconds=watchdog,
                    hard_abort_issued=hard_abort,
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += consumed
                _write_stop(
                    root,
                    reason="CUMULATIVE_TRIPWIRE_STRICTLY_ABOVE_12_A100_HOURS",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(attempt_id,),
                )
                raise GTokV2Stop("campaign tripwire fired; hard abort and return to strategy") from error
            except BaseException as error:
                consumed = int(
                    getattr(
                        error,
                        "_gtok_lifecycle_charge_v2",
                        max(1, math.ceil((time.perf_counter_ns() - started) / 1_000)),
                    )
                )
                # A generic failed attempt is still charged and preserved.  It
                # cannot be coerced into a green campaign receipt.
                watchdog = 2 * projection.projected_run_a100_microseconds
                exceeded_watchdog = consumed > watchdog
                attempt = ComputeAttemptReceiptV2(
                    attempt_id=attempt_id,
                    scope="base_screen",
                    kind="full_run",
                    vocab_size=vocab_size,
                    seed=seed,
                    consumed_a100_microseconds=consumed,
                    status="aborted_watchdog" if exceeded_watchdog else "failed",
                    calibration_projection_sha256=projection.receipt_sha256,
                    projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                    watchdog_limit_a100_microseconds=watchdog,
                    hard_abort_issued=exceeded_watchdog,
                )
                attempts.append(attempt)
                _persist_attempt(root, len(attempts) - 1, attempt)
                cumulative += attempt.consumed_a100_microseconds
                _write_stop(
                    root,
                    reason=f"FULL_RUN_FAILED:{type(error).__name__}",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(),
                )
                raise
            run = measurement.run
            attempt = ComputeAttemptReceiptV2(
                attempt_id=attempt_id,
                scope="base_screen",
                kind="full_run",
                vocab_size=vocab_size,
                seed=seed,
                consumed_a100_microseconds=run.measured_a100_microseconds,
                status="completed",
                calibration_projection_sha256=projection.receipt_sha256,
                projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
                watchdog_limit_a100_microseconds=2 * projection.projected_run_a100_microseconds,
            )
            attempts.append(attempt)
            _persist_attempt(root, len(attempts) - 1, attempt)
            cumulative += attempt.consumed_a100_microseconds
            if cumulative > GTOK_TRIPWIRE_A100_MICROSECONDS:
                _write_stop(
                    root,
                    reason="CUMULATIVE_TRIPWIRE_STRICTLY_ABOVE_12_A100_HOURS",
                    cumulative=cumulative,
                    attempts=tuple(attempts),
                    pending=pending,
                    running=(),
                )
                raise GTokV2Stop("campaign crossed 12 A100-hours; return to strategy")
            runs.append(run)
            measurements.append(measurement)

    attempt_tuple = tuple(attempts)
    ledger_sha256 = validate_sqlite_event_ledger_v2(root, attempt_tuple)
    runtime = RuntimeTripwireSnapshotV2(
        event_ledger_sha256=ledger_sha256,
        cumulative_a100_microseconds=cumulative,
        pending_attempt_ids=(),
        running_attempt_ids=(),
        hard_abort_attempt_ids=(),
        hard_abort_and_report=False,
        return_to_strategy=False,
    )
    compute = CampaignComputeReceiptV2(
        scope="base_screen",
        predecessor_campaign_sha256=None,
        preflight=preflight,
        attempts=attempt_tuple,
        event_ledger_sha256=ledger_sha256,
        consumed_a100_microseconds=cumulative,
        selected_run_a100_microseconds=sum(
            item.measured_a100_microseconds for item in runs
        ),
        runtime_snapshot=runtime,
        all_attempts_accounted=True,
    )
    terminal_gpu_rows = tuple(
        sorted(
            (
                event.attempt_id,
                str(event.gpu_uuid_provenance),
            )
            for event in validate_lifecycle_ledger_v2(root)
            if event.phase == "TERMINAL"
            and event.gpu_uuid_provenance is not None
        )
    )
    if authoritative_execution and (
        len(terminal_gpu_rows) != len(attempt_tuple)
        or any(value == "None" for _attempt_id_value, value in terminal_gpu_rows)
    ):
        raise GTokCampaignV2Error(
            "authoritative attempt ledger lacks physical GPU UUID provenance"
        )
    terminal_offline_rows = tuple(
        sorted(
            (
                event.attempt_id,
                str(event.offline_network_launch_receipt_sha256),
            )
            for event in validate_lifecycle_ledger_v2(root)
            if event.phase == "TERMINAL"
            and event.offline_network_launch_receipt_sha256 is not None
        )
    )
    if authoritative_execution and (
        len(terminal_offline_rows) != len(attempt_tuple)
        or any(value == "None" for _attempt_id_value, value in terminal_offline_rows)
    ):
        raise GTokCampaignV2Error(
            "authoritative attempt ledger lacks physical offline launch provenance"
        )
    revalidate_code_closure()
    if not authoritative_execution:
        result = DryRunCampaignResultV2(
            preflight=preflight,
            compute=compute,
            runs=tuple(runs),
            measurements=tuple(measurements),
            plans=tuple(
                (vocab_size, seed, plans[(vocab_size, seed)])
                for vocab_size in GTOK_VOCABULARY_ARMS
                for seed in seeds
            ),
            training_runtime_receipt_sha256=training_runtime_receipt_sha256,
            code_closure_receipt_sha256=code_closure_receipt_sha256,
            offline_network_receipt_sha256=offline_network_receipt_sha256,
            gpu_uuid_provenance=gpu_uuid_provenance,
            cpu_runtime_identity_sha256=cpu_runtime_identity_sha256,
            microbatch_sequences=microbatch_sequences,
        )
        _exclusive_write(
            root / "non-authoritative-dry-run-receipt.json",
            {
                "authority_status": result.authority_status,
                "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
                "compute": asdict(compute),
                "compute_receipt_sha256": compute.receipt_sha256,
                "runs": tuple(
                    {"payload": asdict(run), "receipt_sha256": run.receipt_sha256}
                    for run in result.runs
                ),
                "measurements": tuple(
                    {
                        "flop_ledger_receipt_sha256": item.flop_ledger.receipt_sha256,
                        "measurement_panel_receipt_sha256": (
                            item.measurement_panel.receipt_sha256
                        ),
                        "payload": asdict(item),
                    }
                    for item in result.measurements
                ),
                "schema": "weft1_gtok_v2_non_authoritative_dry_run",
            },
        )
        return result
    matrix = validate_complete_gtok_matrix_v2(
        tuple(runs),
        corpus=corpus,
        tokenizers=tuple(arm.receipt for arm in tokenizer_arms),
        compute=compute,
    )
    result = BaseCampaignResultV2(
        preflight=preflight,
        compute=compute,
        runs=tuple(runs),
        measurements=tuple(measurements),
        matrix=matrix,
        plans=tuple(
            (vocab_size, seed, plans[(vocab_size, seed)])
            for vocab_size in GTOK_VOCABULARY_ARMS
            for seed in seeds
        ),
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        offline_network_receipt_sha256=str(offline_network_policy_sha256),
        microbatch_sequences=microbatch_sequences,
        cpu_runtime_identity_sha256=str(cpu_runtime_identity_sha256),
        gpu_uuid_provenance_by_attempt=terminal_gpu_rows,
        offline_network_receipt_sha256_by_attempt=terminal_offline_rows,
        precalibration_determinism_authority_sha256=(
            None
            if precalibration_determinism_authority is None
            else precalibration_determinism_authority.receipt_sha256
        ),
        precalibration_determinism_replay_plan_set_sha256=(
            None
            if precalibration_determinism_authority is None
            else precalibration_determinism_authority.replay_plan_set_sha256
        ),
        precalibration_determinism_replay_receipt_sha256s=tuple(
            row.receipt_sha256 for row in precalibration_replay_receipts
        ),
    )
    base_envelope = {
            "campaign_binding_sha256": CAMPAIGN_BINDING_SHA256_V2,
            "compute": asdict(compute),
            "compute_receipt_sha256": compute.receipt_sha256,
            "matrix_receipt_sha256": matrix.receipt_sha256,
            "training_runtime_receipt_sha256": training_runtime_receipt_sha256,
            "code_closure_receipt_sha256": code_closure_receipt_sha256,
            "offline_network_policy_sha256": offline_network_policy_sha256,
            "offline_network_receipt_sha256_by_attempt": terminal_offline_rows,
            "gpu_uuid_provenance_by_attempt": terminal_gpu_rows,
            "cpu_runtime_identity_sha256": cpu_runtime_identity_sha256,
            "microbatch_sequences": microbatch_sequences,
            "gradient_accumulation_slices": GTOK_ACCUMULATION_SLICES_V2,
            "precalibration_determinism_authority_sha256": (
                result.precalibration_determinism_authority_sha256
            ),
            "precalibration_determinism_replay_plan_set_sha256": (
                result.precalibration_determinism_replay_plan_set_sha256
            ),
            "precalibration_determinism_replay_receipt_sha256s": (
                result.precalibration_determinism_replay_receipt_sha256s
            ),
            "initialization_equality_receipt_sha256": (
                None
                if initialization_equality is None
                else initialization_equality.receipt_sha256
            ),
            "precalibration_cpu_evidence_receipt_sha256": (
                None
                if precalibration_evidence is None
                else precalibration_evidence.receipt_sha256
            ),
            "plans": tuple(
                {
                    "plan": asdict(plan),
                    "receipt_sha256": plan.receipt_sha256,
                    "seed": seed,
                    "vocab_size": vocab_size,
                }
                for vocab_size, seed, plan in result.plans
            ),
            "runs": tuple(
                {"payload": asdict(run), "receipt_sha256": run.receipt_sha256}
                for run in result.runs
            ),
            "measurements": tuple(
                {
                    "flop_ledger_receipt_sha256": item.flop_ledger.receipt_sha256,
                    "measurement_panel_receipt_sha256": (
                        item.measurement_panel.receipt_sha256
                    ),
                    "payload": asdict(item),
                }
                for item in result.measurements
            ),
            "schema": "weft1_gtok_v2_base_campaign_receipt",
        }
    base_path = root / "base-campaign-receipt.json"
    if base_path.exists():
        raw, stored = load_canonical_json_snapshot(base_path)
        if raw != canonical_json_bytes(base_envelope) + b"\n":
            raise GTokCampaignV2Error("stored base campaign receipt differs on resume")
    else:
        _exclusive_write(base_path, base_envelope)
    return result


__all__ = [
    "BaseCampaignResultV2",
    "CampaignLifecycleEventV2",
    "DryRunCampaignResultV2",
    "CAMPAIGN_BINDING_SHA256_V2",
    "CalibrationProjectionEvidenceV2",
    "CampaignStopArtifactV2",
    "GTOK_GOVERNED_DATA_ORDER_SEEDS_V2",
    "GTOK_GOVERNED_INITIALIZATION_SEEDS_V2",
    "GTOK_GOVERNED_SEED_ROWS_V2",
    "GTOK_GOVERNED_TRAINING_SEEDS_V2",
    "GTokCampaignV2Error",
    "HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2",
    "InitializationArmStateV2",
    "InitializationEqualityEvidenceV2",
    "InitializationSeedStateV2",
    "PreCalibrationArmMetricsV2",
    "PreCalibrationCpuEvidenceV2",
    "PreCalibrationPlanRowV2",
    "PrecalibrationDeterminismAuthorityV2",
    "TokenizerExecutionArmV2",
    "build_preflight_projection_v2",
    "build_precalibration_cpu_evidence_v2",
    "load_precalibration_cpu_evidence_v2",
    "load_tokenizer_execution_panel_v2",
    "recover_orphaned_lifecycle_attempts_v2",
    "run_base_campaign_v2",
    "validate_lifecycle_ledger_v2",
    "validate_sqlite_event_ledger_v2",
    "write_precalibration_cpu_evidence_v2",
]
