from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import AddedToken, models, pre_tokenizers

import scripts.run_weft1_gtok_campaign_v2 as campaign_cli
import training.weft1_gtok_campaign_v2 as campaign
import training.weft1_gtok_determinism_v2 as determinism
import training.weft1_gtok_offline_v2 as offline
from training.weft1_gtok_code_closure_v2 import (
    CodeArtifactV2,
    GTokCodeClosureReceiptV2,
)
from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_gtok_contract import (
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
)
from training.weft1_corpus_pa import DEFAULT_REQUIREMENTS_LOCK_SHA256
from training.weft1_gtok_tokenizer_a2 import (
    new_a2_tokenizer,
    special_token_strings,
    tokenizer_artifact_sha256,
    tokenizer_inventory_sha256,
    tokenizer_merges_sha256,
)
from training.weft1_gtok_tokenizer_v2 import (
    DOUBLE_FIT_SCHEMA_V2,
    FIT_WORKER_SCHEMA_V2,
    FitWorkerReceiptV2,
    _pretokenizer_regex_sha256,
    _reserved_inventory_sha256,
    tokenizer_byte_round_trip_receipt_v2,
)
from training.weft1_gtok_training_v2 import (
    AnalyticUnsupportedFlopRowV2,
    ArmMeasurementPanelV2,
    CalibrationMeasurementV2,
    CompleteFlopLedgerV2,
    FLOP_BINDING_SHA256_V2,
    FullRunMeasurementV2,
    GTokRunWatchdogV2,
    INITIALIZATION_RECIPE_SHA256_V2,
    OutputSurfacePerformanceV2,
    PACKING_BINDING_SHA256_V2,
    PhysicalShapeFlopReceiptV2,
    ProfilerOperatorFlopRowV2,
    SCHEDULE_BINDING_SHA256_V2,
    StratumCompressionMetricsV2,
    TokenizerCorpusMetricsV2,
    TrainingPlanV2,
    VocabularyFractionRowV2,
)
from training.weft1_seed import derive_module_seed
from training.weft1_gtok_v2_contract import (
    A2FirstFitGroupReceiptV2,
    A2FirstFitScreenReceiptV2,
    BpbMilestoneReceiptV2,
    FrozenScreenCorpusV2,
    GTOK_FIRST_BOUNDARY_BYTES,
    GTOK_SECOND_BOUNDARY_BYTES,
    GTokRunReceiptV2,
    GTokV2Stop,
    TokenizerArmReceiptV2,
    gtok_v2_bound_sha256,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _tokenizer_offline_receipt(
    tmp_path: Path,
) -> tuple[Path, offline.OfflineParentLaunchReceiptV2, str]:
    tokenizer_cli = (
        Path(campaign.__file__).resolve().parents[1]
        / "scripts"
        / "run_weft1_gtok_v2.py"
    ).resolve(strict=True)
    receipt = offline.OfflineParentLaunchReceiptV2(
        parent_network_namespace="net:[1234]",
        unshare_executable=str(offline.LINUX_UNSHARE_PATH_V1),
        unshare_executable_sha256=offline.LINUX_UNSHARE_SHA256_V1,
        python_executable=str(Path(sys.executable).resolve()),
        python_executable_sha256=_hash("python"),
        campaign_script=str(tokenizer_cli),
        campaign_script_sha256=hashlib.sha256(tokenizer_cli.read_bytes()).hexdigest(),
    )
    path = tmp_path / "tokenizer-offline-parent-receipt.json"
    raw = canonical_json_bytes(asdict(receipt)) + b"\n"
    path.write_bytes(raw)
    return path, receipt, hashlib.sha256(raw).hexdigest()


def _precompute_offline_receipt(
    tmp_path: Path,
) -> tuple[Path, offline.OfflineParentLaunchReceiptV2, str]:
    script = (
        Path(campaign.__file__).resolve().parents[1]
        / "scripts"
        / "precompute_weft1_gtok_cpu_v2.py"
    ).resolve(strict=True)
    receipt = offline.OfflineParentLaunchReceiptV2(
        parent_network_namespace="net:[5678]",
        unshare_executable=str(offline.LINUX_UNSHARE_PATH_V1),
        unshare_executable_sha256=offline.LINUX_UNSHARE_SHA256_V1,
        python_executable=str(Path(sys.executable).resolve()),
        python_executable_sha256=_hash("python"),
        campaign_script=str(script),
        campaign_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    )
    path = tmp_path / "precompute-offline-parent-receipt.json"
    raw = canonical_json_bytes(asdict(receipt)) + b"\n"
    path.write_bytes(raw)
    return path, receipt, hashlib.sha256(raw).hexdigest()


def _synthetic_measurement_evidence(
    vocab_size: int,
    plan: TrainingPlanV2,
) -> tuple[CompleteFlopLedgerV2, ArmMeasurementPanelV2]:
    full_slots = 256 * 2_048
    terminal_slots = plan.compute_token_slots - (
        (plan.optimizer_steps - 1) * full_slots
    )
    terminal_rows = terminal_slots // 2_048
    shapes_list = [
        PhysicalShapeFlopReceiptV2(
            batch_rows=256,
            sequence_length=2_048,
            optimizer_phase="initial",
            occurrences=1,
            profiler_rows=(ProfilerOperatorFlopRowV2("aten::mm", vocab_size),),
            unsupported_rows=(
                AnalyticUnsupportedFlopRowV2("synthetic", 1, "fixture=1"),
            ),
            zero_flop_profiler_operators=("aten::view",),
        ),
    ]
    steady_full_occurrences = plan.optimizer_steps - (
        1 if terminal_rows == 256 else 2
    )
    if steady_full_occurrences:
        shapes_list.append(
            PhysicalShapeFlopReceiptV2(
                batch_rows=256,
                sequence_length=2_048,
                optimizer_phase="steady",
                occurrences=steady_full_occurrences,
                profiler_rows=(ProfilerOperatorFlopRowV2("aten::mm", vocab_size),),
                unsupported_rows=(
                    AnalyticUnsupportedFlopRowV2("synthetic", 1, "fixture=1"),
                ),
                zero_flop_profiler_operators=("aten::view",),
            )
        )
    if terminal_rows != 256:
        shapes_list.append(
            PhysicalShapeFlopReceiptV2(
                batch_rows=terminal_rows,
                sequence_length=2_048,
                optimizer_phase="steady",
                occurrences=1,
                profiler_rows=(ProfilerOperatorFlopRowV2("aten::mm", vocab_size),),
                unsupported_rows=(
                    AnalyticUnsupportedFlopRowV2("synthetic", 1, "fixture=1"),
                ),
                zero_flop_profiler_operators=("aten::view",),
            )
        )
    shapes = tuple(shapes_list)
    ledger = CompleteFlopLedgerV2(
        shapes=shapes,
        optimizer_steps=plan.optimizer_steps,
        compute_token_slots=plan.compute_token_slots,
    )
    tokenizer_metrics = TokenizerCorpusMetricsV2(
        strata=tuple(
            StratumCompressionMetricsV2(
                stratum=stratum,
                raw_bytes=10,
                encoded_tokens=5,
                document_count=1,
                coverage_2048_raw_bytes_p50=10,
                coverage_2048_raw_bytes_p95=10,
            )
            for stratum in GTOK_STRATA
        ),
        nonreserved_row_count=vocab_size - 64,
        undertrained_row_count=1,
        undertrained_threshold=1_000,
        training_encoded_tokens=10,
        scanned_training_documents=1,
        scanned_heldout_documents=4,
        exact_byte_round_trip_passed=True,
    )
    full = OutputSurfacePerformanceV2(
        scope="full_softmax_throughput",
        batch_size=1,
        context_tokens=2_048,
        decode_tokens=0,
        warmup_trials=20,
        timed_trials=100,
        timed_token_count=204_800,
        measured_microseconds=1,
    )
    decode = OutputSurfacePerformanceV2(
        scope="output_projection_full_softmax_decode_surface_latency",
        batch_size=1,
        context_tokens=2_048,
        decode_tokens=128,
        warmup_trials=20,
        timed_trials=100,
        timed_token_count=12_800,
        measured_microseconds=1,
    )
    fractions = (
        VocabularyFractionRowV2("proxy", vocab_size * 512, vocab_size * 512 + 1),
        VocabularyFractionRowV2(
            "target_a",
            vocab_size * 1_024,
            302_900_000 + (vocab_size - 32_768) * 1_024,
        ),
        VocabularyFractionRowV2(
            "target_b",
            vocab_size * 1_024,
            305_800_000 + (vocab_size - 32_768) * 1_024,
        ),
    )
    return ledger, ArmMeasurementPanelV2(
        tokenizer_corpus=tokenizer_metrics,
        full_softmax=full,
        decode=decode,
        vocabulary_fractions=fractions,
        complete_measured_flops=ledger.measured_flops,
        flop_ledger_sha256=ledger.receipt_sha256,
    )


def _group(stream: str, stratum: str, target: int) -> A2FirstFitGroupReceiptV2:
    return A2FirstFitGroupReceiptV2(
        stream=stream,
        stratum=stratum,
        target_bytes=target,
        realized_bytes=target,
        deficit_bytes=0,
        document_count=10,
        ordered_raw_content_ids_sha256=_hash(f"{stream}-{stratum}-order"),
    )


def _corpus() -> FrozenScreenCorpusV2:
    first_fit = A2FirstFitScreenReceiptV2(
        groups=(
            *tuple(
                _group("T", name, target)
                for name, target in GTOK_SCREEN_TRAIN_STRATUM_TARGETS
            ),
            *tuple(
                _group("H", name, target)
                for name, target in GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
            ),
        ),
        training_framed_stream_sha256=_hash("T-stream"),
        heldout_framed_stream_sha256=_hash("H-stream"),
        document_overlap_count=0,
        cluster_overlap_count=0,
    )
    return FrozenScreenCorpusV2(
        full_corpus_manifest_sha256=_hash("full"),
        screen_submanifest_sha256=_hash("screen"),
        d6_physical_evidence_sha256=_hash("physical-d6"),
        corpus_freeze_receipt_sha256=_hash("freeze"),
        d1_d6_gate_bundle_sha256=_hash("gates"),
        decontamination_receipt_sha256=_hash("decon"),
        first_fit=first_fit,
    )


def _double_fit_core(
    vocab: int,
    offline_sha256: str | None = None,
    offline_policy_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "first_process_id": 17,
        "first_worker_receipt_sha256": _hash(f"first-worker-{vocab}"),
        "second_process_id": 18,
        "second_worker_receipt_sha256": _hash(f"second-worker-{vocab}"),
        "offline_network_receipt_sha256": (
            offline_sha256 or _hash("tokenizer-offline-network")
        ),
        "offline_network_policy_sha256": (
            offline_policy_sha256 or _hash("tokenizer-offline-policy")
        ),
        "status": "PARENT_REHASHED_SUBPROCESSES_MATCH",
        "tokenizer_json_sha256": _hash(f"artifact-{vocab}"),
        "vocab_size": vocab,
    }


def _tokenizer_receipts(
    offline_sha256: str | None = None,
    offline_policy_sha256: str | None = None,
) -> tuple[TokenizerArmReceiptV2, ...]:
    return tuple(
        TokenizerArmReceiptV2(
            vocab_size=vocab,
            tokenizer_json_sha256=_hash(f"artifact-{vocab}"),
            merges_sha256=_hash(f"merges-{vocab}"),
            token_inventory_sha256=_hash(f"inventory-{vocab}"),
            reserved_inventory_sha256=_hash("reserved"),
            pretokenizer_regex_sha256=_hash("regex"),
            fit_stream_sha256=_hash("T-stream"),
            full_corpus_manifest_sha256=_hash("full"),
            double_fit_receipt_sha256=gtok_v2_bound_sha256(
                DOUBLE_FIT_SCHEMA_V2,
                _double_fit_core(vocab, offline_sha256, offline_policy_sha256),
            ),
            byte_round_trip_receipt_sha256=_hash("round-trip"),
            token_inventory_count=vocab,
        )
        for vocab in GTOK_VOCABULARY_ARMS
    )


def _fixture_tokenizer_payload(vocab_size: int) -> bytes:
    protocol = special_token_strings()
    tokens = list(protocol)
    for token in sorted(pre_tokenizers.ByteLevel.alphabet()):
        if token not in tokens:
            tokens.append(token)
    index = 0
    while len(tokens) < vocab_size:
        token = f"<|fixture_{index:05d}|>"
        index += 1
        if token not in tokens:
            tokens.append(token)
    tokenizer = new_a2_tokenizer()
    tokenizer.model = models.BPE(
        vocab={token: token_id for token_id, token in enumerate(tokens)},
        merges=[],
    )
    tokenizer.add_special_tokens(
        [
            AddedToken(
                token,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=True,
            )
            for token in protocol
        ]
    )
    return tokenizer.to_str(pretty=False).encode("utf-8")


def _materialize_tokenizer_producer_fixture(
    *,
    artifact_root: Path,
    corpus: FrozenScreenCorpusV2,
    offline_receipt: offline.OfflineParentLaunchReceiptV2,
    offline_sha256: str,
) -> tuple[list[dict[str, object]], tuple[TokenizerArmReceiptV2, ...]]:
    rows: list[dict[str, object]] = []
    receipts: list[TokenizerArmReceiptV2] = []
    for offset, vocab_size in enumerate(GTOK_VOCABULARY_ARMS):
        arm_root = artifact_root / f"vocab-{vocab_size}"
        first_root = arm_root / "fit-a"
        second_root = arm_root / "fit-b"
        first_root.mkdir(parents=True)
        second_root.mkdir(parents=True)
        payload = _fixture_tokenizer_payload(vocab_size)
        round_trip = tokenizer_byte_round_trip_receipt_v2(payload)
        artifact_sha256 = tokenizer_artifact_sha256(payload)
        common = {
            "vocab_size": vocab_size,
            "tokenizer_json_sha256": artifact_sha256,
            "merges_sha256": tokenizer_merges_sha256(payload),
            "token_inventory_sha256": tokenizer_inventory_sha256(payload),
            "reserved_inventory_sha256": _reserved_inventory_sha256(),
            "pretokenizer_regex_sha256": _pretokenizer_regex_sha256(),
            "fit_stream_sha256": corpus.training_stream_sha256,
            "full_corpus_manifest_sha256": corpus.full_corpus_manifest_sha256,
            "screen_submanifest_sha256": corpus.screen_submanifest_sha256,
            "physical_d6_evidence_sha256": corpus.d6_physical_evidence_sha256,
            "tokenizer_fit_input_receipt_sha256": _hash("fit-input"),
            "bpe_safety_receipt_sha256": _hash("bpe-safety"),
            "byte_round_trip_receipt_sha256": round_trip["receipt_sha256"],
            "executable_sha256": offline_receipt.python_executable_sha256,
            "dependency_lock_sha256": DEFAULT_REQUIREMENTS_LOCK_SHA256,
            "environment_identity_sha256": _hash("pa-environment"),
            "offline_network_receipt_sha256": offline_sha256,
            "offline_network_policy_sha256": offline_receipt.policy_sha256,
            "tokenizers_version": "0.22.2",
        }
        common["runtime_attestation_receipt_sha256"] = gtok_v2_bound_sha256(
            "weft1_gtok_v2_tokenizer_runtime_attestation",
            {
                "dependency_lock_sha256": common["dependency_lock_sha256"],
                "environment_identity_sha256": common[
                    "environment_identity_sha256"
                ],
                "executable_sha256": common["executable_sha256"],
            },
        )
        workers = []
        for process_id, worker_root in (
            (101 + 2 * offset, first_root),
            (102 + 2 * offset, second_root),
        ):
            (worker_root / "tokenizer.json").write_bytes(payload)
            worker = FitWorkerReceiptV2(
                process_id=process_id,
                output_root=str(worker_root.resolve()),
                **common,
            )
            envelope = {
                "payload": asdict(worker),
                "receipt_sha256": worker.receipt_sha256,
                "schema": FIT_WORKER_SCHEMA_V2,
            }
            (worker_root / "fit-worker-receipt.json").write_bytes(
                canonical_json_bytes(envelope) + b"\n"
            )
            workers.append(worker)
        first, second = workers
        double_core = {
            "first_process_id": first.process_id,
            "first_worker_receipt_sha256": first.receipt_sha256,
            "offline_network_receipt_sha256": offline_sha256,
            "offline_network_policy_sha256": offline_receipt.policy_sha256,
            "second_process_id": second.process_id,
            "second_worker_receipt_sha256": second.receipt_sha256,
            "status": "PARENT_REHASHED_SUBPROCESSES_MATCH",
            "tokenizer_json_sha256": artifact_sha256,
            "vocab_size": vocab_size,
        }
        double_fit = {
            **double_core,
            "receipt_sha256": gtok_v2_bound_sha256(
                DOUBLE_FIT_SCHEMA_V2, double_core
            ),
        }
        arm = TokenizerArmReceiptV2(
            vocab_size=vocab_size,
            tokenizer_json_sha256=artifact_sha256,
            merges_sha256=common["merges_sha256"],
            token_inventory_sha256=common["token_inventory_sha256"],
            reserved_inventory_sha256=common["reserved_inventory_sha256"],
            pretokenizer_regex_sha256=common["pretokenizer_regex_sha256"],
            fit_stream_sha256=corpus.training_stream_sha256,
            full_corpus_manifest_sha256=corpus.full_corpus_manifest_sha256,
            double_fit_receipt_sha256=double_fit["receipt_sha256"],
            byte_round_trip_receipt_sha256=round_trip["receipt_sha256"],
            token_inventory_count=vocab_size,
        )
        arm_payload = asdict(arm)
        evidence = {
            "arm": arm_payload,
            "arm_receipt_sha256": arm.receipt_sha256,
            "double_fit": double_fit,
            "offline_network_receipt_sha256": offline_sha256,
            "offline_network_policy_sha256": offline_receipt.policy_sha256,
            "selected_artifact_relative_path": "fit-a/tokenizer.json",
        }
        (arm_root / "tokenizer-arm-receipt.json").write_bytes(
            canonical_json_bytes(evidence) + b"\n"
        )
        rows.append(
            {
                "arm": arm_payload,
                "arm_receipt_sha256": arm.receipt_sha256,
                "corpus_receipt_sha256": corpus.receipt_sha256,
                "evidence": evidence,
                "offline_network_receipt_sha256": offline_sha256,
                "offline_network_policy_sha256": offline_receipt.policy_sha256,
                "output_root": str(arm_root.resolve()),
            }
        )
        receipts.append(arm)
    return rows, tuple(receipts)


@dataclass
class FakeSource:
    physical_d6_evidence_sha256: str
    training_raw_bytes: int
    heldout_raw_bytes_by_stratum: tuple[tuple[str, int], ...]
    training_order_receipts: tuple[tuple[int, int, str], ...]

    def training_documents(self, seed: int):
        raise AssertionError("planning is injected in this fixture")

    def heldout_documents(self, stratum: str):
        raise AssertionError("execution is injected in this fixture")


def _observations(corpus: FrozenScreenCorpusV2, bpb_shift: float) -> tuple[BpbMilestoneReceiptV2, ...]:
    rows = tuple(
        StratumNllReceipt(
            stratum=row.stratum,
            nll_nats=(1.0 + bpb_shift) * row.realized_bytes,
            raw_byte_count=row.realized_bytes,
        )
        for row in corpus.first_fit.heldout
    )
    return (
        BpbMilestoneReceiptV2(
            label="after_1b",
            optimizer_step=1,
            previous_training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES - 1,
            training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=rows,
        ),
        BpbMilestoneReceiptV2(
            label="after_2b",
            optimizer_step=2,
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES - 1,
            training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=rows,
        ),
        BpbMilestoneReceiptV2(
            label="terminal_realized_T",
            optimizer_step=3,
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES,
            training_raw_bytes=corpus.training_realized_bytes,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=rows,
        ),
    )


def _synthetic_executors(
    frozen: FrozenScreenCorpusV2,
    seeds: tuple[int, int],
    plan: TrainingPlanV2,
    *,
    runtime_sha256: str = _hash("training-runtime"),
    code_closure_sha256: str = _hash("code-closure"),
):
    def calibrate(**kwargs):
        return CalibrationMeasurementV2(
            steps=100,
            warmup_steps=20,
            measured_steps=80,
            measured_tokens=plan.compute_token_slots,
            measured_a100_microseconds=100_000,
            charged_a100_microseconds=110_000,
            measured_heldout_evaluation_a100_microseconds=5_000,
            heldout_evaluations_per_full_run=3,
            measured_output_surface_a100_microseconds=2_000,
            output_surface_benchmarks_per_full_run=1,
            planned_tokens_per_run=plan.compute_token_slots,
            shared_initial_state_sha256=_hash("shared-0"),
        )

    def full(**kwargs):
        vocab = kwargs["vocab_size"]
        seed = kwargs["seed"]
        seed_index = seeds.index(seed)
        flop_ledger, measurement_panel = _synthetic_measurement_evidence(vocab, plan)
        # The attempt-boundary fixture retains a smaller synthetic replay shape,
        # but its governed run receipt must still encode complete 256x2048
        # optimizer batches under the post-S2 Q2 contract.
        trained_tokens = plan.optimizer_steps * 256 * 2_048
        run = GTokRunReceiptV2(
            vocab_size=vocab,
            seed=seed,
            frozen_screen_corpus_sha256=frozen.receipt_sha256,
            tokenizer_receipt_sha256=kwargs["tokenizer_receipt"].receipt_sha256,
            initialization_recipe_sha256=INITIALIZATION_RECIPE_SHA256_V2,
            initialization_seed=kwargs["initialization_seed"],
            shared_initial_state_sha256=_hash(f"shared-{seed_index}"),
            data_order_seed=kwargs["data_order_seed"],
            data_order_sha256=kwargs["data_order_sha256"],
            training_runtime_receipt_sha256=runtime_sha256,
            code_closure_receipt_sha256=code_closure_sha256,
            gpu_uuid_provenance="GPU-11111111-2222-3333-4444-555555555555",
            compute_attempt_id=kwargs["compute_attempt_id"],
            measured_a100_microseconds=50_000,
            measured_flops=flop_ledger.measured_flops,
            optimizer=a1_flat_adamw_recipe(),
            observations=_observations(
                frozen,
                vocab / 1_000_000 + seed_index / 10_000,
            ),
            stream_bytes=frozen.training_realized_bytes,
            stream_docs=plan.document_count,
            stream_tokens=trained_tokens,
            trained_tokens=trained_tokens,
            dropped_tokens=0,
            trained_bytes=plan.realized_raw_bytes,
            dropped_bytes=0,
            trained_docs_full=plan.document_count,
            dropped_docs=0,
        )
        return FullRunMeasurementV2(
            run=run,
            training_runtime_receipt_sha256=runtime_sha256,
            code_closure_receipt_sha256=code_closure_sha256,
            training_plan_sha256=plan.receipt_sha256,
            packing_binding_sha256=PACKING_BINDING_SHA256_V2,
            schedule_binding_sha256=SCHEDULE_BINDING_SHA256_V2,
            flop_binding_sha256=FLOP_BINDING_SHA256_V2,
            flop_ledger=flop_ledger,
            measurement_panel=measurement_panel,
        )

    return calibrate, full


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    frozen = _corpus()
    seeds = campaign.GTOK_GOVERNED_TRAINING_SEEDS_V2
    source = FakeSource(
        physical_d6_evidence_sha256=frozen.d6_physical_evidence_sha256,
        training_raw_bytes=frozen.training_realized_bytes,
        heldout_raw_bytes_by_stratum=tuple(
            (row.stratum, row.realized_bytes) for row in frozen.first_fit.heldout
        ),
        training_order_receipts=tuple(
            (seed, data_seed, _hash(f"order-{seed}"))
            for seed, data_seed in zip(
                seeds,
                campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
                strict=True,
            )
        ),
    )
    plan = TrainingPlanV2(
        optimizer_steps=3,
        compute_token_slots=(2 * 256 + 128) * 2_048,
        valid_prediction_count=1_000,
        realized_raw_bytes=frozen.training_realized_bytes,
        document_count=10,
        packed_stream_sha256=_hash("packed"),
    )
    monkeypatch.setattr(campaign, "plan_training_stream_v2", lambda *args, **kwargs: plan)
    monkeypatch.setattr(campaign.TokenizerExecutionArmV2, "load", lambda self: object())
    arms = tuple(
        campaign.TokenizerExecutionArmV2(
            receipt=receipt,
            tokenizer_json_path=tmp_path / "unused",
            offline_network_receipt_sha256=_hash("tokenizer-offline-network"),
            offline_network_policy_sha256=_hash("tokenizer-offline-policy"),
        )
        for receipt in _tokenizer_receipts()
    )
    return frozen, seeds, source, plan, arms


def _initialization_rows() -> tuple[campaign.InitializationSeedStateV2, ...]:
    return tuple(
        campaign.InitializationSeedStateV2(
            training_seed=training_seed,
            initialization_seed=initialization_seed,
            arms=tuple(
                campaign.InitializationArmStateV2(
                    vocab_size=vocab_size,
                    shared_nonvocabulary_state_sha256=_hash(
                        f"shared-{seed_index}"
                    ),
                )
                for vocab_size in GTOK_VOCABULARY_ARMS
            ),
        )
        for seed_index, (
            training_seed,
            initialization_seed,
            _data_seed,
        ) in enumerate(campaign.GTOK_GOVERNED_SEED_ROWS_V2)
    )


def _install_synthetic_physical_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise campaign replay persistence without allocating the proxy graph."""

    attestation = determinism.CudaDeterminismAttestationV2(
        policy=determinism.CUDA_DETERMINISM_POLICY_V2,
        torch_version=str(torch.__version__),
        device_type="cuda",
        authority_status="AUTHORITATIVE_A100_FLASH_ONLY",
    )
    monkeypatch.setattr(
        campaign,
        "apply_and_attest_cuda_determinism_policy_v2",
        lambda **_kwargs: attestation,
    )

    def execute(**kwargs):
        binding = kwargs["replay_plan_binding"]
        plan = kwargs["plan"]
        shapes = (
            determinism.ReplayBatchShapeV2(
                label="full",
                batch_rows=256,
                sequence_length=2_048,
                valid_prediction_count=1,
                batch_sha256=_hash(f"full-{binding.vocab_size}"),
            ),
            determinism.ReplayBatchShapeV2(
                label="terminal_partial",
                batch_rows=binding.terminal_rows,
                sequence_length=2_048,
                valid_prediction_count=1,
                batch_sha256=_hash(
                    f"tail-{binding.vocab_size}-{binding.terminal_rows}"
                ),
            ),
        )
        by_shape = tuple(
            (row.label, _hash(f"state-{binding.receipt_sha256}-{row.label}"))
            for row in shapes
        )
        fingerprint = determinism.DeterminismReplayFingerprintV2(
            policy_receipt_sha256=attestation.policy.receipt_sha256,
            replay_plan_binding_sha256=binding.receipt_sha256,
            training_plan_sha256=plan.receipt_sha256,
            vocab_size=binding.vocab_size,
            initialization_seed=binding.representative_initialization_seed,
            run_seed=binding.representative_training_seed,
            microbatch_sequences=kwargs["microbatch_sequences"],
            shapes=shapes,
            model_state_sha256_by_shape=by_shape,
            optimizer_state_sha256_by_shape=by_shape,
            evaluation_output_sha256_by_shape=by_shape,
            fused_backend_operator_names=tuple(
                sorted(
                    (
                        "aten::_scaled_dot_product_flash_attention",
                        "aten::_scaled_dot_product_flash_attention_backward",
                    )
                )
            ),
        )
        return determinism.DeterminismReplayReplicaV2(
            replica_index=kwargs["replica_index"],
            fingerprint=fingerprint,
            charged_device_microseconds=100,
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )

    monkeypatch.setattr(
        campaign,
        "execute_precalibration_determinism_replay_replica_v2",
        execute,
    )


def test_governed_seed_rows_equal_the_a2_namespaced_derivation() -> None:
    expected = tuple(
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
        for training_seed in campaign.GTOK_GOVERNED_TRAINING_SEEDS_V2
    )
    assert expected == campaign.GTOK_GOVERNED_SEED_ROWS_V2 == (
        (4_069_725_298_476_216_533, 9_305_630_768_498_788_030, 10_666_192_988_433_719_740),
        (13_256_058_689_613_801_745, 12_171_684_496_048_357_438, 4_197_282_192_878_334_768),
    )


def test_confirmation_seed_registry_binds_all_direct_roots_and_fresh_roles() -> None:
    rows = campaign.GTOK_CONFIRMATION_SEED_ROWS_V2
    assert tuple((row.vocab_size, row.seed_slot) for row in rows) == tuple(
        (vocab_size, slot)
        for vocab_size in GTOK_VOCABULARY_ARMS
        for slot in (0, 1)
    )
    assert tuple(row.run_seed for row in rows) == (
        14_491_391_970_410_640_762,
        11_241_563_954_874_861_528,
        15_081_792_657_614_179_907,
        18_192_772_026_849_707_115,
        9_884_118_125_684_999_954,
        7_190_589_679_906_404_951,
        1_571_914_625_861_595_228,
        2_644_639_369_611_050_861,
    )
    assert all(
        row.run_seed
        == int.from_bytes(
            hashlib.sha256(row.registry_key.encode("ascii")).digest()[:8],
            "big",
        )
        for row in rows
    )
    assert all(
        row.initialization_seed
        == derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.init.shared.{row.run_seed}",
        )
        and row.data_order_seed
        == derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{row.run_seed}",
        )
        for row in rows
    )
    assert len({row.run_seed for row in rows}) == 8
    assert len({row.initialization_seed for row in rows}) == 8
    assert len({row.data_order_seed for row in rows}) == 8
    assert {row.run_seed for row in rows}.isdisjoint(
        campaign.GTOK_GOVERNED_TRAINING_SEEDS_V2
    )
    assert {row.initialization_seed for row in rows}.isdisjoint(
        campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2
    )
    assert {row.data_order_seed for row in rows}.isdisjoint(
        campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2
    )
    assert all(
        campaign.confirmation_seed_rows_for_vocabulary_v2(vocab_size)
        == tuple(row for row in rows if row.vocab_size == vocab_size)
        for vocab_size in GTOK_VOCABULARY_ARMS
    )
    assert len(campaign.GTOK_CONFIRMATION_SEED_BINDING_SHA256_V2) == 64


def test_confirmation_seed_registry_rejects_tampering() -> None:
    row = campaign.GTOK_CONFIRMATION_SEED_ROWS_V2[0]
    with pytest.raises(ValueError, match="direct SHA root"):
        replace(row, run_seed=row.run_seed + 1)
    with pytest.raises(ValueError, match="initialization seed"):
        replace(row, initialization_seed=row.initialization_seed + 1)
    with pytest.raises(ValueError, match="data-order seed"):
        replace(row, data_order_seed=row.data_order_seed + 1)
    with pytest.raises(ValueError, match="unregistered vocabulary"):
        campaign.confirmation_seed_rows_for_vocabulary_v2(65_536)


def test_seed_one_all_arm_initialization_substitution_is_rejected() -> None:
    frozen = _corpus()
    seeds = campaign.GTOK_GOVERNED_TRAINING_SEEDS_V2
    plan = TrainingPlanV2(
        optimizer_steps=3,
        compute_token_slots=3 * 256 * 2_048,
        valid_prediction_count=1_000,
        realized_raw_bytes=frozen.training_realized_bytes,
        document_count=10,
        packed_stream_sha256=_hash("packed"),
    )
    _calibrate, full = _synthetic_executors(frozen, seeds, plan)
    evidence = campaign.InitializationEqualityEvidenceV2(
        rows=_initialization_rows(),
        training_runtime_receipt_sha256=_hash("training-runtime"),
        code_closure_receipt_sha256=_hash("code-closure"),
        offline_network_receipt_sha256=_hash("offline-policy"),
    )
    substituted = _hash("self-consistent-seed-one-all-arms")
    seed = seeds[1]
    for vocab_size, tokenizer_receipt in zip(
        GTOK_VOCABULARY_ARMS,
        _tokenizer_receipts(),
        strict=True,
    ):
        measurement = full(
            vocab_size=vocab_size,
            seed=seed,
            tokenizer=object(),
            tokenizer_receipt=tokenizer_receipt,
            plan=plan,
            initialization_seed=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2[1],
            data_order_seed=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2[1],
            data_order_sha256=_hash("order"),
            compute_attempt_id=f"substitute-{vocab_size}",
            watchdog_limit_a100_microseconds=1_000_000,
            prior_campaign_a100_microseconds=0,
            document_factory=lambda: iter(()),
        )
        measurement = replace(
            measurement,
            run=replace(
                measurement.run,
                shared_initial_state_sha256=substituted,
            ),
        )
        with pytest.raises(campaign.GTokCampaignV2Error, match="pre-spend"):
            campaign._validate_full_initialization_v2(
                measurement,
                evidence=evidence,
                vocab_size=vocab_size,
                training_seed=seed,
            )
def test_precalibration_cpu_evidence_round_trip_and_mutation_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen, seeds, _source, plan, arms = _fixture(monkeypatch, tmp_path)
    offline_path, offline_receipt, offline_sha256 = _precompute_offline_receipt(
        tmp_path
    )
    evidence = campaign.PreCalibrationCpuEvidenceV2(
        plan_rows=tuple(
            campaign.PreCalibrationPlanRowV2(vocab_size, seed, plan)
            for vocab_size in GTOK_VOCABULARY_ARMS
            for seed in seeds
        ),
        arm_metrics=tuple(
            campaign.PreCalibrationArmMetricsV2(
                vocab_size=vocab_size,
                tokenizer_receipt_sha256=next(
                    arm.receipt.receipt_sha256
                    for arm in arms
                    if arm.receipt.vocab_size == vocab_size
                ),
                tokenizer_corpus=(
                    _synthetic_measurement_evidence(vocab_size, plan)[1].tokenizer_corpus
                ),
            )
            for vocab_size in GTOK_VOCABULARY_ARMS
        ),
        initialization_rows=_initialization_rows(),
        frozen_screen_corpus_sha256=frozen.receipt_sha256,
        code_closure_receipt_sha256=_hash("code-closure"),
        cpu_runtime_identity_sha256=_hash("cpu-runtime"),
        offline_network_policy_sha256=offline_receipt.policy_sha256,
        offline_network_receipt_sha256=offline_sha256,
        generator_script_sha256=offline_receipt.campaign_script_sha256,
    )
    path = tmp_path / "precalibration.json"
    campaign.write_precalibration_cpu_evidence_v2(path, evidence)
    assert campaign.load_precalibration_cpu_evidence_v2(path) == evidence
    envelope = json.loads(path.read_bytes())
    envelope["payload"]["cpu_runtime_identity_sha256"] = _hash("substituted")
    path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    with pytest.raises(campaign.GTokCampaignV2Error, match="identity drifted"):
        campaign.load_precalibration_cpu_evidence_v2(path)

    forged = replace(
        evidence,
        offline_network_receipt_sha256=_hash("self-consistent-forged-launch"),
    )
    forged_path = tmp_path / "precalibration-forged.json"
    campaign.write_precalibration_cpu_evidence_v2(forged_path, forged)
    loaded_forged = campaign.load_precalibration_cpu_evidence_v2(forged_path)
    with pytest.raises(campaign.GTokCampaignV2Error, match="authenticated offline"):
        campaign.validate_precalibration_generation_v2(
            loaded_forged,
            offline_parent_receipt_path=offline_path,
        )


@pytest.mark.parametrize("substitution", ("training", "initialization", "data"))
def test_campaign_rejects_seed_substitution_before_calibration(
    substitution: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen, seeds, source, _plan, arms = _fixture(monkeypatch, tmp_path)
    initialization = campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2
    data = campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2
    if substitution == "training":
        seeds = tuple(reversed(seeds))
    elif substitution == "initialization":
        initialization = (initialization[0] + 1, initialization[1])
    else:
        data = (data[0] + 1, data[1])
    called = False

    def calibration(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("seed substitution must fail before calibration")

    with pytest.raises(campaign.GTokCampaignV2Error, match="seeds differ"):
        campaign.run_base_campaign_v2(
            corpus=frozen,
            source=source,  # type: ignore[arg-type]
            tokenizer_arms=arms,
            seeds=seeds,  # type: ignore[arg-type]
            initialization_seeds=initialization,
            data_order_seeds=data,
            output_root=tmp_path / f"seed-substitution-{substitution}",
            device=torch.device("cpu"),
            microbatch_sequences=1,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            calibration_executor=calibration,
        )
    assert not called
    assert not (tmp_path / f"seed-substitution-{substitution}").exists()


def test_production_cli_has_no_seed_override_and_passes_only_governed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _corpus()
    panel = tmp_path / "tokenizer-panel.json"
    panel.write_bytes(b"{}\n")
    output = tmp_path / "campaign-output"
    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class Process:
        executable_sha256: str
        dependency_lock_sha256: str
        environment_identity_sha256: str
        process_id: int
        output_root: str

    class Runtime:
        environment_payload = {"status": "fixture"}
        receipt_sha256 = _hash("training-runtime")

    class Closure:
        artifacts = ()
        git_commit = "0" * 40
        receipt_sha256 = _hash("code-closure")
        schema = "weft1_gtok_code_closure_v2"
        status = "CLEAN_EXACT_CODE_CLOSURE"

    def attest(**kwargs):
        calls["runtime_attestation"] = kwargs
        return Runtime()

    monkeypatch.setattr(campaign_cli, "attest_gtok_training_runtime_v2", attest)
    monkeypatch.setattr(
        campaign_cli, "require_resolved_confirmation_semantics_v2", lambda: None
    )
    monkeypatch.setattr(
        campaign_cli,
        "cpu_runtime_identity_sha256_from_payload_v2",
        lambda *_: _hash("cpu-runtime"),
    )
    monkeypatch.setattr(
        campaign_cli,
        "gpu_uuid_provenance_v2",
        lambda **_: "GPU-11111111-2222-3333-4444-555555555555",
    )
    monkeypatch.setattr(
        campaign_cli, "capture_gtok_code_closure_v2", lambda *_: Closure()
    )
    monkeypatch.setattr(
        campaign_cli,
        "assert_offline_campaign_child_v2",
        lambda *_: _hash("offline-network"),
    )
    monkeypatch.setattr(
        campaign_cli,
        "load_offline_parent_receipt_v2",
        lambda *_: (
            SimpleNamespace(policy_sha256=_hash("offline-policy")),
            _hash("offline-network"),
        ),
    )
    monkeypatch.setattr(
        campaign_cli, "validate_gtok_code_closure_v2", lambda *_, **__: None
    )
    monkeypatch.setattr(campaign_cli.torch.cuda, "set_device", lambda device: calls.setdefault("device", device))
    monkeypatch.setattr(campaign_cli.torch.cuda, "get_device_name", lambda device: "NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(campaign_cli, "require_production_a100_v2", lambda device: calls.setdefault("a100", device))
    monkeypatch.setattr(campaign_cli, "load_frozen_screen_corpus_v2", lambda **_: frozen)
    monkeypatch.setattr(
        campaign_cli,
        "load_tokenizer_execution_panel_v2",
        lambda **_: (
            SimpleNamespace(
                offline_network_receipt_sha256=_hash("tokenizer-offline-network"),
                offline_network_policy_sha256=_hash("tokenizer-offline-policy"),
            ),
        ),
    )
    precalibration = SimpleNamespace(
        receipt_sha256=_hash("precalibration"),
        cpu_runtime_identity_sha256=_hash("cpu-runtime"),
    )
    monkeypatch.setattr(
        campaign_cli,
        "load_precalibration_cpu_evidence_v2",
        lambda *_: precalibration,
    )
    monkeypatch.setattr(campaign_cli, "load_v4_corpus_source_v2", lambda *_, **__: "source")

    def run(**kwargs):
        calls["run"] = kwargs
        kwargs["output_root"].mkdir()
        return SimpleNamespace(
            matrix=SimpleNamespace(receipt_sha256=_hash("matrix")),
            offline_network_receipt_sha256_by_attempt=(
                ("attempt", _hash("offline-network")),
            ),
        )

    monkeypatch.setattr(campaign_cli, "run_base_campaign_v2", run)
    assert "seed" not in {action.dest for action in campaign_cli._parser()._actions}
    assert campaign_cli.main(
        [
            "--corpus-root", str(tmp_path / "corpus"),
            "--freeze-receipt", str(tmp_path / "freeze"),
            "--gate-bundle", str(tmp_path / "gates"),
            "--c2-evidence", str(tmp_path / "c2"),
            "--decon-receipt", str(tmp_path / "decon"),
            "--training-requirements-lock", str(tmp_path / "lock"),
            "--runtime-build-receipt", str(tmp_path / "runtime-build"),
            "--pa-runtime-build-receipt", str(tmp_path / "pa-runtime-build"),
            "--training-runtime-binding", str(tmp_path / "runtime-binding"),
            "--offline-network-receipt", str(tmp_path / "offline-network"),
            "--precalibration-cpu-evidence", str(tmp_path / "precalibration"),
            "--precalibration-offline-network-receipt", str(tmp_path / "precompute-offline"),
            "--tokenizer-panel-receipt", str(panel),
            "--tokenizer-artifact-root", str(tmp_path / "tokenizers"),
            "--tokenizer-offline-network-receipt", str(tmp_path / "tokenizer-offline"),
            "--output-root", str(output),
            "--microbatch-sequences", "8",
        ]
    ) == 0
    run_args = calls["run"]
    assert run_args["seeds"] == campaign.GTOK_GOVERNED_TRAINING_SEEDS_V2
    assert run_args["initialization_seeds"] == campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2
    assert run_args["data_order_seeds"] == campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2
    assert calls["runtime_attestation"]["pa_runtime_build_receipt"] == tmp_path / "pa-runtime-build"
    assert calls["a100"] == torch.device("cuda:0")
    assert (output / "production-launch-receipt.json").is_file()


def test_production_launch_receipt_is_idempotent_but_not_replaceable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "launch.json"
    first = campaign_cli._exclusive_json(path, {"status": "PASS"})
    assert campaign_cli._exclusive_json(path, {"status": "PASS"}) == first
    with pytest.raises(RuntimeError, match="differs"):
        campaign_cli._exclusive_json(path, {"status": "DRIFT"})


def test_confirmation_semantics_authority_is_complete_but_does_not_reorder_pb() -> None:
    campaign.require_resolved_confirmation_semantics_v2()
    assert campaign.CONFIRMATION_SEMANTICS_AUTHORITY_STATUS_V2 == (
        "RESOLVED_PARENT_PLUS_S1_PLUS_S2_BUILD_AXIS;GPU_REMAINS_BEHIND_PB"
    )
    assert tuple(path for path, _sha256 in campaign.CONFIRMATION_SEMANTICS_AUTHORITY_V2) == (
        "docs/STRATEGY_GTOK_CONFIRMATION_SEMANTICS_20260831.md",
        "docs/STRATEGY_GTOK_SEMANTICS_AMENDMENT_S1_20260831.md",
        "docs/STRATEGY_GTOK_SEMANTICS_AMENDMENT_S2_20260831.md",
    )


def test_tokenizer_panel_loader_revalidates_receipts_and_rejects_row_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _corpus()
    artifact_root = tmp_path / "tokenizers"
    artifact_root.mkdir()
    offline_path, offline_receipt, tokenizer_offline_sha256 = (
        _tokenizer_offline_receipt(tmp_path)
    )
    rows, receipts = _materialize_tokenizer_producer_fixture(
        artifact_root=artifact_root,
        corpus=frozen,
        offline_receipt=offline_receipt,
        offline_sha256=tokenizer_offline_sha256,
    )
    panel = {
        "arms": rows,
        "offline_network_receipt_sha256": tokenizer_offline_sha256,
        "offline_network_policy_sha256": offline_receipt.policy_sha256,
        "schema": "weft1_gtok_v2_tokenizer_panel",
        "vocabularies": GTOK_VOCABULARY_ARMS,
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_bytes(canonical_json_bytes(panel) + b"\n")
    loaded = campaign.load_tokenizer_execution_panel_v2(
        panel_receipt_path=panel_path,
        artifact_root=artifact_root,
        offline_parent_receipt_path=offline_path,
        corpus=frozen,
    )
    assert tuple(row.receipt.vocab_size for row in loaded) == GTOK_VOCABULARY_ARMS

    substituted = dict(panel)
    substituted["arms"] = list(reversed(rows))
    substituted_path = tmp_path / "substituted-panel.json"
    substituted_path.write_bytes(canonical_json_bytes(substituted) + b"\n")
    with pytest.raises(campaign.GTokCampaignV2Error, match="order"):
        campaign.load_tokenizer_execution_panel_v2(
            panel_receipt_path=substituted_path,
            artifact_root=artifact_root,
            offline_parent_receipt_path=offline_path,
            corpus=frozen,
        )

    substituted_offline = dict(panel)
    substituted_offline["offline_network_receipt_sha256"] = _hash(
        "self-consistent-but-not-physical"
    )
    substituted_offline["arms"] = [
        {
            **row,
            "offline_network_receipt_sha256": substituted_offline[
                "offline_network_receipt_sha256"
            ],
            "evidence": {
                **row["evidence"],
                "offline_network_receipt_sha256": substituted_offline[
                    "offline_network_receipt_sha256"
                ],
                "double_fit": {
                    **row["evidence"]["double_fit"],
                    "offline_network_receipt_sha256": substituted_offline[
                        "offline_network_receipt_sha256"
                    ],
                },
            },
        }
        for row in rows
    ]
    substituted_offline_path = tmp_path / "substituted-offline-panel.json"
    substituted_offline_path.write_bytes(
        canonical_json_bytes(substituted_offline) + b"\n"
    )
    with pytest.raises(campaign.GTokCampaignV2Error, match="physical launch receipt"):
        campaign.load_tokenizer_execution_panel_v2(
            panel_receipt_path=substituted_offline_path,
            artifact_root=artifact_root,
            offline_parent_receipt_path=offline_path,
            corpus=frozen,
        )

    first_artifact = artifact_root / "vocab-16384" / "fit-a" / "tokenizer.json"
    original_artifact = first_artifact.read_bytes()
    for mutation in ("normalizer", "regex"):
        value = json.loads(original_artifact)
        if mutation == "normalizer":
            value["normalizer"] = {"type": "NFC"}
        else:
            value["pre_tokenizer"]["pretokenizers"][0]["pattern"] = {
                "Regex": "(?s).+"
            }
        first_artifact.write_bytes(canonical_json_bytes(value))
        with pytest.raises(campaign.GTokCampaignV2Error, match="fit-a evidence"):
            campaign.load_tokenizer_execution_panel_v2(
                panel_receipt_path=panel_path,
                artifact_root=artifact_root,
                offline_parent_receipt_path=offline_path,
                corpus=frozen,
            )
        first_artifact.write_bytes(original_artifact)

    same_size = original_artifact.replace(b"fixture_00000", b"fixture_90000", 1)
    assert len(same_size) == len(original_artifact) and same_size != original_artifact
    first_artifact.write_bytes(same_size)
    with pytest.raises(campaign.GTokCampaignV2Error, match="artifact-local"):
        campaign.load_tokenizer_execution_panel_v2(
            panel_receipt_path=panel_path,
            artifact_root=artifact_root,
            offline_parent_receipt_path=offline_path,
            corpus=frozen,
        )
    first_artifact.write_bytes(original_artifact)

    first_worker = artifact_root / "vocab-16384" / "fit-a" / "fit-worker-receipt.json"
    worker_envelope = json.loads(first_worker.read_bytes())
    worker_envelope["payload"]["environment_identity_sha256"] = _hash(
        "self-consistent-worker-substitution"
    )
    runtime_core = {
        "dependency_lock_sha256": worker_envelope["payload"][
            "dependency_lock_sha256"
        ],
        "environment_identity_sha256": worker_envelope["payload"][
            "environment_identity_sha256"
        ],
        "executable_sha256": worker_envelope["payload"]["executable_sha256"],
    }
    worker_envelope["payload"]["runtime_attestation_receipt_sha256"] = (
        gtok_v2_bound_sha256(
            "weft1_gtok_v2_tokenizer_runtime_attestation", runtime_core
        )
    )
    substituted_worker = FitWorkerReceiptV2(**worker_envelope["payload"])
    worker_envelope["receipt_sha256"] = substituted_worker.receipt_sha256
    first_worker.write_bytes(canonical_json_bytes(worker_envelope) + b"\n")
    with pytest.raises(campaign.GTokCampaignV2Error, match="producer receipts"):
        campaign.load_tokenizer_execution_panel_v2(
            panel_receipt_path=panel_path,
            artifact_root=artifact_root,
            offline_parent_receipt_path=offline_path,
            corpus=frozen,
        )


def test_campaign_rejects_physical_d6_substitution_before_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, seeds, source, _plan, arms = _fixture(monkeypatch, tmp_path)
    called = False

    def calibration(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("substituted physical evidence must fail before calibration")

    with pytest.raises(campaign.GTokCampaignV2Error, match="frozen D6"):
        campaign.run_base_campaign_v2(
            corpus=frozen,
            source=replace(source, physical_d6_evidence_sha256=_hash("substitute")),
            tokenizer_arms=arms,
            seeds=seeds,
            initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
            data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
            output_root=tmp_path / "must-not-exist",
            device=torch.device("cpu"),
            microbatch_sequences=1,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            calibration_executor=calibration,
        )
    assert not called
    assert not (tmp_path / "must-not-exist").exists()


def test_campaign_rejects_physical_data_order_seed_substitution_before_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, seeds, source, _plan, arms = _fixture(monkeypatch, tmp_path)
    rows = list(source.training_order_receipts)
    training_seed, data_seed, order_sha = rows[0]
    rows[0] = (training_seed, data_seed + 1, order_sha)
    called = False

    def calibration(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("physical data-seed substitution must fail before execution")

    with pytest.raises(campaign.GTokCampaignV2Error, match="physical V4 consumer"):
        campaign.run_base_campaign_v2(
            corpus=frozen,
            source=replace(source, training_order_receipts=tuple(rows)),
            tokenizer_arms=arms,
            seeds=seeds,
            initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
            data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
            output_root=tmp_path / "must-not-exist-data-seed",
            device=torch.device("cpu"),
            microbatch_sequences=1,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            calibration_executor=calibration,
        )
    assert not called
    assert not (tmp_path / "must-not-exist-data-seed").exists()


def test_lifecycle_sqlite_replay_tamper_and_duplicate_terminal_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lifecycle"
    root.mkdir()
    start = campaign.CampaignLifecycleEventV2(
        logical_attempt_id="base-run-v16384-s1",
        attempt_id="base-run-v16384-s1",
        scope="base_screen",
        kind="full_run",
        phase="START",
        charged_a100_microseconds=1,
        terminal_status=None,
    )
    heartbeat = replace(start, phase="HEARTBEAT", charged_a100_microseconds=30_000_001)
    terminal = replace(
        start,
        phase="TERMINAL",
        charged_a100_microseconds=31_000_000,
        terminal_status="preempted",
    )
    campaign._append_lifecycle_event_v2(root, start)
    campaign._append_lifecycle_event_v2(root, heartbeat)
    assert campaign.validate_lifecycle_ledger_v2(root) == (start, heartbeat)
    campaign._append_lifecycle_event_v2(root, terminal)
    assert campaign.validate_lifecycle_ledger_v2(root)[-1] == terminal
    with pytest.raises(campaign.GTokCampaignV2Error, match="terminal"):
        campaign._append_lifecycle_event_v2(root, terminal)
    mirror = next((root / "lifecycle-events").glob("000001-*.json"))
    mirror.write_bytes(mirror.read_bytes().replace(b"30000001", b"30000002"))
    with pytest.raises(campaign.GTokCampaignV2Error, match="mirror"):
        campaign.validate_lifecycle_ledger_v2(root)


def test_replay_success_watchdog_prices_outer_lifecycle_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0, 10_000_000, 10_000_000))
    monkeypatch.setattr(campaign.time, "perf_counter_ns", lambda: next(clock))
    root = tmp_path / "outer-lifecycle-watchdog"
    root.mkdir()

    with pytest.raises(
        determinism.GTokDeterminismV2Stop,
        match="DETERMINISM_REPLAY_WATCHDOG",
    ):
        campaign._execute_with_lifecycle_v2(
            root=root,
            logical_attempt_id="replay-outer",
            attempt_id="replay-outer",
            scope="base_screen",
            kind="determinism_replay",
            operation=lambda: SimpleNamespace(inner_microseconds=10),
            success_charge=lambda value: value.inner_microseconds,
            success_payload=lambda value: {
                "inner_microseconds": value.inner_microseconds
            },
            success_watchdog_limit_a100_microseconds=5_000,
        )

    terminal = campaign.validate_lifecycle_ledger_v2(root)[-1]
    assert terminal.phase == "TERMINAL"
    assert terminal.terminal_status == "aborted_watchdog"
    assert terminal.charged_a100_microseconds == 10_000
    assert terminal.completion_payload is None


def test_hard_kill_relaunch_closes_orphan_and_uses_distinct_fresh_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hard-kill"
    root.mkdir()
    logical = "base-run-v24576-s4069725298476216533"
    start = campaign.CampaignLifecycleEventV2(
        logical_attempt_id=logical,
        attempt_id=logical,
        scope="base_screen",
        kind="full_run",
        phase="START",
        charged_a100_microseconds=1,
        terminal_status=None,
    )
    heartbeat = replace(
        start,
        phase="HEARTBEAT",
        charged_a100_microseconds=campaign.HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2,
    )
    campaign._append_lifecycle_event_v2(root, start)
    campaign._append_lifecycle_event_v2(root, heartbeat)
    recovered = campaign.recover_orphaned_lifecycle_attempts_v2(root)
    assert len(recovered) == 1
    assert recovered[0].terminal_status == "preempted"
    assert recovered[0].charged_a100_microseconds == (
        heartbeat.charged_a100_microseconds
        + campaign.HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2
    )
    events = campaign.validate_lifecycle_ledger_v2(root)
    retry = campaign._next_physical_attempt_id_v2(logical, events)
    assert retry == f"{logical}.retry-1"
    retry_start = replace(start, attempt_id=retry)
    campaign._append_lifecycle_event_v2(root, retry_start)
    retry_terminal = replace(
        retry_start,
        phase="TERMINAL",
        charged_a100_microseconds=10,
        terminal_status="completed",
        completion_payload={"receipt": "synthetic"},
    )
    campaign._append_lifecycle_event_v2(root, retry_terminal)
    completed = {
        event.logical_attempt_id
        for event in campaign.validate_lifecycle_ledger_v2(root)
        if event.phase == "TERMINAL" and event.terminal_status == "completed"
    }
    assert completed == {logical}


def test_preheartbeat_hard_kills_each_charge_one_full_cadence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "preheartbeat-kills"
    root.mkdir()
    logical = "calibration-v16384"
    first = campaign.CampaignLifecycleEventV2(
        logical_attempt_id=logical,
        attempt_id=logical,
        scope="base_screen",
        kind="calibration",
        phase="START",
        charged_a100_microseconds=1,
        terminal_status=None,
    )
    campaign._append_lifecycle_event_v2(root, first)
    first_recovered = campaign.recover_orphaned_lifecycle_attempts_v2(root)[0]
    retry = replace(first, attempt_id=f"{logical}.retry-1")
    campaign._append_lifecycle_event_v2(root, retry)
    second_recovered = campaign.recover_orphaned_lifecycle_attempts_v2(root)[0]
    expected = campaign.HEARTBEAT_INTERVAL_A100_MICROSECONDS_V2 + 1
    assert first_recovered.charged_a100_microseconds == expected
    assert second_recovered.charged_a100_microseconds == expected
    assert sum(
        event.charged_a100_microseconds
        for event in campaign.validate_lifecycle_ledger_v2(root)
        if event.phase == "TERMINAL"
    ) == 2 * expected


def test_injected_campaign_is_strictly_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, seeds, source, plan, arms = _fixture(monkeypatch, tmp_path)

    def calibrate(**kwargs):
        return CalibrationMeasurementV2(
            steps=100,
            warmup_steps=20,
            measured_steps=80,
            measured_tokens=plan.compute_token_slots,
            measured_a100_microseconds=100_000,
            charged_a100_microseconds=110_000,
            measured_heldout_evaluation_a100_microseconds=5_000,
            heldout_evaluations_per_full_run=3,
            measured_output_surface_a100_microseconds=2_000,
            output_surface_benchmarks_per_full_run=1,
            planned_tokens_per_run=plan.compute_token_slots,
            shared_initial_state_sha256=_hash("shared-calibration"),
        )

    def full(**kwargs):
        vocab = kwargs["vocab_size"]
        seed = kwargs["seed"]
        seed_index = seeds.index(seed)
        flop_ledger, measurement_panel = _synthetic_measurement_evidence(vocab, plan)
        run = GTokRunReceiptV2(
            vocab_size=vocab,
            seed=seed,
            frozen_screen_corpus_sha256=frozen.receipt_sha256,
            tokenizer_receipt_sha256=kwargs["tokenizer_receipt"].receipt_sha256,
            initialization_recipe_sha256=_hash("init-recipe"),
            initialization_seed=kwargs["initialization_seed"],
            shared_initial_state_sha256=_hash(f"shared-{seed_index}"),
                data_order_seed=kwargs["data_order_seed"],
                data_order_sha256=kwargs["data_order_sha256"],
                training_runtime_receipt_sha256=_hash("training-runtime"),
                code_closure_receipt_sha256=_hash("code-closure"),
            compute_attempt_id=kwargs["compute_attempt_id"],
            measured_a100_microseconds=50_000,
            measured_flops=flop_ledger.measured_flops,
            optimizer=a1_flat_adamw_recipe(),
            observations=_observations(frozen, vocab / 1_000_000 + seed_index / 10_000),
        )
        return FullRunMeasurementV2(
            run=run,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            training_plan_sha256=plan.receipt_sha256,
            packing_binding_sha256=PACKING_BINDING_SHA256_V2,
            schedule_binding_sha256=SCHEDULE_BINDING_SHA256_V2,
            flop_binding_sha256=FLOP_BINDING_SHA256_V2,
            flop_ledger=flop_ledger,
            measurement_panel=measurement_panel,
        )

    result = campaign.run_base_campaign_v2(
        corpus=frozen,
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        seeds=seeds,
        initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
        data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
        output_root=tmp_path / "campaign",
        device=torch.device("cpu"),
        microbatch_sequences=1,
        training_runtime_receipt_sha256=_hash("training-runtime"),
        code_closure_receipt_sha256=_hash("code-closure"),
        calibration_executor=calibrate,
        full_run_executor=full,
    )
    assert isinstance(result, campaign.DryRunCampaignResultV2)
    assert not hasattr(result, "matrix")
    assert result.authority_status == "NON_AUTHORITATIVE_INJECTED_EXECUTORS"
    assert len(result.runs) == 8
    assert len(result.compute.attempts) == 12
    assert not (tmp_path / "campaign" / "base-campaign-receipt.json").exists()
    assert (tmp_path / "campaign" / "non-authoritative-dry-run-receipt.json").is_file()
    assert result.compute.consumed_a100_microseconds >= 840_000
    assert len(tuple((tmp_path / "campaign" / "events").glob("*.json"))) == 12
    assert not tuple((tmp_path / "campaign").rglob("*.pt"))


def test_authoritative_attempt_boundary_relaunch_skips_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock_ns = 0

    def deterministic_lifecycle_clock() -> int:
        nonlocal clock_ns
        clock_ns += 1_000_000
        return clock_ns

    # Make replica lifecycle overhead stable: the synthetic replay reports an
    # inner 100 us while every outer lifecycle interval is exactly 1,000 us.
    monkeypatch.setattr(
        campaign.time,
        "perf_counter_ns",
        deterministic_lifecycle_clock,
    )
    frozen, seeds, source, plan, arms = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        campaign, "require_resolved_confirmation_semantics_v2", lambda: None
    )
    closure_receipt = GTokCodeClosureReceiptV2(
        git_commit="1" * 40,
        artifacts=(
            CodeArtifactV2(
                relative_path="governed.py",
                bytes=1,
                sha256=_hash("governed-source"),
            ),
        ),
    )
    runtime_sha = _hash("training-runtime")
    init_hash = _hash("shared-calibration")
    calibrate, full = _synthetic_executors(
        frozen,
        seeds,
        plan,
        runtime_sha256=runtime_sha,
        code_closure_sha256=closure_receipt.receipt_sha256,
    )
    raw_calibrate = calibrate

    def calibrate(**kwargs):
        # Keep the synthetic per-run watchdog above the governed 30-second
        # conservative orphan charge used by the hard-kill resume branch.
        return replace(
            raw_calibrate(**kwargs),
            measured_a100_microseconds=1_000_000_000,
            charged_a100_microseconds=1_100_000_000,
            measured_heldout_evaluation_a100_microseconds=5_000_000,
            measured_output_surface_a100_microseconds=2_000_000,
        )
    monkeypatch.setattr(campaign, "validate_gtok_code_closure_v2", lambda *_, **__: None)
    # This test isolates attempt-boundary resume with synthetic three-step
    # executors; production calibration-prefix validation has its own physical
    # 100-step mutation tests.
    monkeypatch.setattr(campaign, "validate_calibration_prefix_v2", lambda *_, **__: None)
    monkeypatch.setattr(
        campaign,
        "_physical_initialization_equality_evidence_v2",
        lambda **kwargs: campaign.InitializationEqualityEvidenceV2(
            rows=tuple(
                campaign.InitializationSeedStateV2(
                    training_seed=training_seed,
                    initialization_seed=initialization_seed,
                    arms=tuple(
                        campaign.InitializationArmStateV2(vocab_size, init_hash)
                        for vocab_size in GTOK_VOCABULARY_ARMS
                    ),
                )
                for training_seed, initialization_seed, _data_seed in (
                    campaign.GTOK_GOVERNED_SEED_ROWS_V2
                )
            ),
            training_runtime_receipt_sha256=kwargs[
                "training_runtime_receipt_sha256"
            ],
            code_closure_receipt_sha256=kwargs["code_closure_receipt_sha256"],
            offline_network_receipt_sha256=kwargs[
                "offline_network_receipt_sha256"
            ],
        ),
    )
    monkeypatch.setattr(campaign, "_default_calibration_executor", lambda **_: calibrate)
    monkeypatch.setattr(campaign, "_default_full_executor", lambda **_: full)
    _install_synthetic_physical_replay(monkeypatch)
    precompute_offline_path, precompute_offline, precompute_offline_sha256 = (
        _precompute_offline_receipt(tmp_path)
    )
    precalibration = campaign.PreCalibrationCpuEvidenceV2(
        plan_rows=tuple(
            campaign.PreCalibrationPlanRowV2(vocab_size, seed, plan)
            for vocab_size in GTOK_VOCABULARY_ARMS
            for seed in seeds
        ),
        arm_metrics=tuple(
            campaign.PreCalibrationArmMetricsV2(
                vocab_size=vocab_size,
                tokenizer_receipt_sha256=next(
                    arm.receipt.receipt_sha256
                    for arm in arms
                    if arm.receipt.vocab_size == vocab_size
                ),
                tokenizer_corpus=(
                    _synthetic_measurement_evidence(vocab_size, plan)[1].tokenizer_corpus
                ),
            )
            for vocab_size in GTOK_VOCABULARY_ARMS
        ),
        initialization_rows=_initialization_rows(),
        frozen_screen_corpus_sha256=frozen.receipt_sha256,
        code_closure_receipt_sha256=closure_receipt.receipt_sha256,
        cpu_runtime_identity_sha256=_hash("cpu-runtime"),
        offline_network_policy_sha256=precompute_offline.policy_sha256,
        offline_network_receipt_sha256=precompute_offline_sha256,
        generator_script_sha256=precompute_offline.campaign_script_sha256,
    )
    output = tmp_path / "authoritative-resume"
    common = dict(
        corpus=frozen,
        source=source,
        tokenizer_arms=arms,
        seeds=seeds,
        initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
        data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
        output_root=output,
        device=torch.device("cpu"),
        microbatch_sequences=campaign.GTOK_MICROBATCH_SEQUENCES_V2,
        training_runtime_receipt_sha256=runtime_sha,
        code_closure_receipt_sha256=closure_receipt.receipt_sha256,
        code_closure_receipt=closure_receipt,
        repository_root=tmp_path,
        offline_network_receipt_sha256=_hash("offline-network"),
        offline_network_policy_sha256=_hash("offline-policy"),
        gpu_uuid_provenance="GPU-11111111-2222-3333-4444-555555555555",
        precalibration_cpu_evidence=precalibration,
        precalibration_offline_parent_receipt_path=precompute_offline_path,
        cpu_runtime_identity_sha256=_hash("cpu-runtime"),
    )
    first = campaign.run_base_campaign_v2(**common)
    assert isinstance(first, campaign.BaseCampaignResultV2)
    assert len(first.runs) == 8
    assert len(first.measurements) == 8
    base_receipt = json.loads((output / "base-campaign-receipt.json").read_text())
    base_receipt_bytes = (output / "base-campaign-receipt.json").read_bytes()
    replay_authority_bytes = (
        output / "precalibration-determinism-authority.json"
    ).read_bytes()
    replay_receipt_bytes = tuple(
        path.read_bytes()
        for path in sorted((output / "determinism-replay").rglob("receipt.json"))
    )
    assert len(base_receipt["measurements"]) == 8
    for row, measurement in zip(base_receipt["measurements"], first.measurements):
        assert canonical_json_bytes(row["payload"]) == canonical_json_bytes(
            asdict(measurement)
        )
        assert (
            row["flop_ledger_receipt_sha256"]
            == measurement.flop_ledger.receipt_sha256
        )
        assert (
            row["measurement_panel_receipt_sha256"]
            == measurement.measurement_panel.receipt_sha256
        )

    def must_not_execute(**_kwargs):
        raise AssertionError("completed attempt must be skipped on relaunch")

    monkeypatch.setattr(
        campaign,
        "_default_calibration_executor",
        lambda **_: must_not_execute,
    )
    monkeypatch.setattr(campaign, "_default_full_executor", lambda **_: must_not_execute)
    monkeypatch.setattr(
        campaign,
        "execute_precalibration_determinism_replay_replica_v2",
        must_not_execute,
    )
    second = campaign.run_base_campaign_v2(
        **{
            **common,
            "gpu_uuid_provenance": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "offline_network_receipt_sha256": _hash("offline-network-new-vm"),
        }
    )
    assert isinstance(second, campaign.BaseCampaignResultV2)
    assert second.matrix.receipt_sha256 == first.matrix.receipt_sha256
    assert second.compute.receipt_sha256 == first.compute.receipt_sha256
    assert second.measurements == first.measurements
    assert (output / "base-campaign-receipt.json").read_bytes() == base_receipt_bytes
    assert (
        output / "precalibration-determinism-authority.json"
    ).read_bytes() == replay_authority_bytes
    assert tuple(
        path.read_bytes()
        for path in sorted((output / "determinism-replay").rglob("receipt.json"))
    ) == replay_receipt_bytes
    assert {
        gpu_uuid for _attempt_id, gpu_uuid in second.gpu_uuid_provenance_by_attempt
    } == {"GPU-11111111-2222-3333-4444-555555555555"}
    assert {
        receipt_sha256
        for _attempt_id, receipt_sha256 in (
            second.offline_network_receipt_sha256_by_attempt
        )
    } == {_hash("offline-network")}
    assert len(tuple((output / "events").glob("*.json"))) == (
        12 + 2 * len(first.precalibration_determinism_replay_receipt_sha256s)
    )

    # Rebuild the temp-only append ledgers with exactly one full row left as a
    # START-only hard kill.  The governed relaunch must authenticate/reuse the
    # already-green replay authority, conservatively close the orphan, retry
    # only that row from a fresh model, and skip the other seven completed rows.
    selected_logical = campaign._attempt_id("run", GTOK_VOCABULARY_ARMS[0], seeds[0])
    persisted_attempts = campaign._load_persisted_attempts_v2(output)
    retained_attempts = tuple(
        row for row in persisted_attempts if row.attempt_id != selected_logical
    )
    lifecycle = campaign.validate_lifecycle_ledger_v2(output)
    retained_lifecycle = tuple(
        row for row in lifecycle if row.logical_attempt_id != selected_logical
    )
    for suffix in ("", "-wal", "-shm"):
        (output / f"campaign-events.sqlite3{suffix}").unlink(missing_ok=True)
        (output / f"campaign-lifecycle.sqlite3{suffix}").unlink(missing_ok=True)
    for path in (output / "events").glob("*.json"):
        path.unlink()
    for path in (output / "lifecycle-events").glob("*.json"):
        path.unlink()
    for index, row in enumerate(retained_attempts):
        campaign._persist_attempt(output, index, row)
    for row in retained_lifecycle:
        campaign._append_lifecycle_event_v2(output, row)
    campaign._append_lifecycle_event_v2(
        output,
        campaign.CampaignLifecycleEventV2(
            logical_attempt_id=selected_logical,
            attempt_id=selected_logical,
            scope="base_screen",
            kind="full_run",
            phase="START",
            charged_a100_microseconds=1,
            terminal_status=None,
            gpu_uuid_provenance="GPU-11111111-2222-3333-4444-555555555555",
            offline_network_launch_receipt_sha256=_hash("offline-network"),
        ),
    )
    (output / "base-campaign-receipt.json").unlink()
    retry_calls: list[tuple[int, int]] = []

    def retry_one(**kwargs):
        retry_calls.append((kwargs["vocab_size"], kwargs["seed"]))
        measurement = full(**kwargs)
        return replace(
            measurement,
            run=replace(
                measurement.run,
                gpu_uuid_provenance=(
                    "GPU-ffffffff-1111-2222-3333-444444444444"
                ),
            ),
        )

    monkeypatch.setattr(campaign, "_default_full_executor", lambda **_: retry_one)
    third = campaign.run_base_campaign_v2(
        **{
            **common,
            "gpu_uuid_provenance": "GPU-ffffffff-1111-2222-3333-444444444444",
            "offline_network_receipt_sha256": _hash("offline-network-third-vm"),
        }
    )
    assert isinstance(third, campaign.BaseCampaignResultV2)
    assert retry_calls == [(GTOK_VOCABULARY_ARMS[0], seeds[0])]
    assert (
        output / "precalibration-determinism-authority.json"
    ).read_bytes() == replay_authority_bytes
    assert tuple(
        path.read_bytes()
        for path in sorted((output / "determinism-replay").rglob("receipt.json"))
    ) == replay_receipt_bytes
    selected_attempts = tuple(
        row
        for row in third.compute.attempts
        if row.attempt_id == selected_logical
        or row.attempt_id.startswith(f"{selected_logical}.retry-")
    )
    assert tuple(row.status for row in selected_attempts) == (
        "preempted",
        "completed",
    )
    assert not tuple(output.rglob("*.pt"))


def test_watchdog_failure_writes_stop_and_never_mints_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, seeds, source, plan, arms = _fixture(monkeypatch, tmp_path)

    def calibrate(**kwargs):
        return CalibrationMeasurementV2(
            steps=100,
            warmup_steps=20,
            measured_steps=80,
            measured_tokens=plan.compute_token_slots,
            measured_a100_microseconds=1_000,
            charged_a100_microseconds=1_100,
            measured_heldout_evaluation_a100_microseconds=50,
            heldout_evaluations_per_full_run=3,
            measured_output_surface_a100_microseconds=10,
            output_surface_benchmarks_per_full_run=1,
            planned_tokens_per_run=plan.compute_token_slots,
            shared_initial_state_sha256=_hash("shared"),
        )

    def full(**kwargs):
        raise GTokRunWatchdogV2(2_301)

    with pytest.raises(GTokV2Stop, match="watchdog"):
        campaign.run_base_campaign_v2(
            corpus=frozen,
            source=source,  # type: ignore[arg-type]
            tokenizer_arms=arms,
            seeds=seeds,
            initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
            data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
            output_root=tmp_path / "campaign-stop",
            device=torch.device("cpu"),
            microbatch_sequences=1,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            calibration_executor=calibrate,
            full_run_executor=full,
        )
    assert (tmp_path / "campaign-stop" / "campaign-stop.json").is_file()
    assert not (tmp_path / "campaign-stop" / "base-campaign-receipt.json").exists()


def test_over_tripwire_projection_persists_calibrations_and_stops_before_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, seeds, source, plan, arms = _fixture(monkeypatch, tmp_path)
    full_called = False

    def calibrate(**kwargs):
        return CalibrationMeasurementV2(
            steps=100,
            warmup_steps=20,
            measured_steps=80,
            measured_tokens=plan.compute_token_slots,
            measured_a100_microseconds=3_960_000_000,  # 1.1 A100-hours per arm
            charged_a100_microseconds=3_960_000_101,
            measured_heldout_evaluation_a100_microseconds=100,
            heldout_evaluations_per_full_run=3,
            measured_output_surface_a100_microseconds=1,
            output_surface_benchmarks_per_full_run=1,
            planned_tokens_per_run=plan.compute_token_slots,
            shared_initial_state_sha256=_hash("shared"),
        )

    def full(**kwargs):
        nonlocal full_called
        full_called = True
        raise AssertionError("preflight must stop before a full run")

    with pytest.raises(GTokV2Stop, match="preflight projection"):
        campaign.run_base_campaign_v2(
            corpus=frozen,
            source=source,  # type: ignore[arg-type]
            tokenizer_arms=arms,
            seeds=seeds,
            initialization_seeds=campaign.GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
            data_order_seeds=campaign.GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
            output_root=tmp_path / "projected-stop",
            device=torch.device("cpu"),
            microbatch_sequences=1,
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            calibration_executor=calibrate,
            full_run_executor=full,
        )
    assert not full_called
    assert len(tuple((tmp_path / "projected-stop" / "events").glob("*.json"))) == 4
    projection_path = tmp_path / "projected-stop" / "calibration-projection-evidence.json"
    projection_raw = projection_path.read_bytes()
    projection = json.loads(projection_raw)
    assert len(projection["payload"]["calibrations"]) == 4
    assert len(projection["payload"]["calibration_attempts"]) == 4
    assert all(
            row["calibration_steps"] == 100
            and row["measured_tokens"] == plan.compute_token_slots
            and row["planned_tokens_per_run"] == plan.compute_token_slots
            and row["measured_a100_microseconds"] == 3_960_000_000
            and row["charged_calibration_a100_microseconds"] == 3_960_000_101
            and row["measured_heldout_evaluation_a100_microseconds"] == 100
            and row["heldout_evaluations_per_full_run"] == 3
            and row["measured_output_surface_a100_microseconds"] == 1
            and row["output_surface_benchmarks_per_full_run"] == 1
        for row in projection["payload"]["calibrations"]
    )
    assert projection["payload"]["projected_campaign_a100_microseconds"] == 47_520_002_812
    assert projection["payload"]["full_run_launch_count"] == 0
    stop = json.loads(
        (tmp_path / "projected-stop" / "campaign-stop.json").read_bytes()
    )
    assert stop["payload"]["reason"] == "PREFLIGHT_PROJECTION_EXCEEDS_12_A100_HOURS"
    assert (
        stop["payload"]["calibration_projection_evidence_receipt_sha256"]
        == projection["receipt_sha256"]
    )
    assert (
        stop["payload"]["calibration_projection_evidence_physical_sha256"]
        == hashlib.sha256(projection_raw).hexdigest()
    )
    assert not (tmp_path / "projected-stop" / "base-campaign-receipt.json").exists()
