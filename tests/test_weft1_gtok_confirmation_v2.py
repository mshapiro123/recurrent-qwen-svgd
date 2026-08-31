from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import training.weft1_gtok_confirmation_v2 as confirmation
import training.weft1_gtok_campaign_v2 as campaign
import scripts.run_weft1_gtok_full_campaign_v2 as full_campaign_cli
from tests.test_weft1_gtok_v2_contract import (
    _confirmation_compute,
    _confirmation_evidence_closure,
    _confirmation_runs,
    _matrix,
    _strata,
    SEEDS,
)
from training.weft1_gtok_campaign_v2 import (
    BaseCampaignResultV2,
    GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
    GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
    GTOK_GOVERNED_TRAINING_SEEDS_V2,
    TokenizerExecutionArmV2,
)
from training.weft1_corpus_materialize_a3 import ConfirmationConsumerOrderV4
from training.weft1_gtok_training_v2 import (
    AnalyticUnsupportedFlopRowV2,
    CompleteFlopLedgerV2,
    ConfirmationTrainingPlanV2,
    PhysicalShapeFlopReceiptV2,
    ProfilerOperatorFlopRowV2,
)
from training.weft1_gtok_v2_contract import (
    ArmCalibrationProjectionV2,
    BpbMilestoneReceiptV2,
    ConfirmationEvidenceClosureV2,
    ConfirmationFreshEvidenceJoinV2,
    ComputeConfirmationRunV2,
    GTokV2Stop,
    select_vocabulary_v2,
    validate_compute_confirmation_v2,
)


_BASE_GPU = "GPU-00000000-0000-0000-0000-000000000001"
_CONFIRMATION_GPU = "GPU-00000000-0000-0000-0000-000000000002"


def _full_campaign_argv(tmp_path: Path) -> list[str]:
    panel = tmp_path / "tokenizer-panel.json"
    panel.write_bytes(b"synthetic-panel\n")
    paths = {
        "corpus-root": tmp_path / "corpus",
        "freeze-receipt": tmp_path / "freeze.json",
        "gate-bundle": tmp_path / "gates.json",
        "c2-evidence": tmp_path / "c2.json",
        "decon-receipt": tmp_path / "decon.json",
        "training-requirements-lock": tmp_path / "requirements.lock",
        "runtime-build-receipt": tmp_path / "runtime-build.json",
        "pa-runtime-build-receipt": tmp_path / "pa-runtime-build.json",
        "training-runtime-binding": tmp_path / "runtime-binding.json",
        "offline-network-receipt": tmp_path / "offline.json",
        "precalibration-cpu-evidence": tmp_path / "precalibration.json",
        "precalibration-offline-network-receipt": tmp_path / "precompute-offline.json",
        "tokenizer-panel-receipt": panel,
        "tokenizer-artifact-root": tmp_path / "tokenizers",
        "tokenizer-offline-network-receipt": tmp_path / "tokenizer-offline.json",
        "output-root": tmp_path / "campaign",
    }
    result: list[str] = []
    for name, path in paths.items():
        result.extend((f"--{name}", str(path)))
    result.extend(("--microbatch-sequences", "8"))
    return result


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_OFFLINE_POLICY = _hash("offline-policy")
_OFFLINE_A = _hash("offline-physical-a")
_OFFLINE_B = _hash("offline-physical-b")


def _base(matrix) -> BaseCampaignResultV2:
    return BaseCampaignResultV2(
        preflight=matrix.compute.preflight,
        compute=matrix.compute,
        runs=matrix.runs,
        measurements=(),
        matrix=matrix,
        plans=(),
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        offline_network_receipt_sha256=_OFFLINE_POLICY,
        gpu_uuid_provenance_by_attempt=(("base-synthetic", _BASE_GPU),),
    )


def _evidence(matrix, *, reachable: bool = True):
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    pair = selection.compute_confirmation_pair
    common = min(
        run.measured_flops
        for run in matrix.runs
        if run.vocab_size in pair
    )
    result = []
    by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    for vocab_size in pair:
        for seed in matrix.seeds:
            run = by_key[(vocab_size, seed)]
            if vocab_size == min(pair):
                costs = [1] * 99 + [common - 99]
            elif reachable:
                costs = [1] * 99 + [common - 99, run.measured_flops - common]
            else:
                costs = [1] * 99 + [common - 98, run.measured_flops - common - 1]
            result.append(
                confirmation.BaseRunFlopEvidenceV2(
                    vocab_size=vocab_size,
                    seed=seed,
                    base_run_receipt_sha256=run.receipt_sha256,
                    base_compute_attempt_id=run.compute_attempt_id,
                    flop_ledger_sha256=_hash(f"ledger-{vocab_size}-{seed}"),
                    steps=tuple(
                        confirmation.BaseStepFlopV2(
                            optimizer_step=index,
                            batch_rows=256,
                            sequence_length=2_048,
                            optimizer_phase="initial" if index == 1 else "steady",
                            measured_flops=value,
                        )
                        for index, value in enumerate(costs, start=1)
                    ),
                    measured_flops=run.measured_flops,
                )
            )
    return tuple(result)


class _Source:
    def __init__(self, matrix) -> None:
        self.root = Path(".")
        self.physical_d6_evidence_sha256 = matrix.corpus.d6_physical_evidence_sha256
        self.training_raw_bytes = matrix.corpus.training_realized_bytes
        self.heldout_raw_bytes_by_stratum = matrix.corpus.heldout_denominator_signature
        self.training_order_receipts = tuple(
            (seed, 20_000 + seed, _hash(f"data-order-{seed}"))
            for seed in matrix.seeds
        )

    def training_documents(self, seed: int):
        raise AssertionError("prefix planning is replaced by a bound synthetic plan")

    def confirmation_training_documents(self, order_receipt):
        raise AssertionError("physical training is replaced by a synthetic executor")


def _arms(matrix, tmp_path: Path):
    return tuple(
        TokenizerExecutionArmV2(
            receipt=row,
            tokenizer_json_path=tmp_path / f"unused-{row.vocab_size}",
            offline_network_receipt_sha256=_hash("tokenizer-offline-network"),
            offline_network_policy_sha256=_hash("tokenizer-offline-policy"),
        )
        for row in matrix.tokenizers
    )


def _order(matrix, *, run_seed: int, data_order_seed: int) -> ConfirmationConsumerOrderV4:
    return ConfirmationConsumerOrderV4(
        confirmation_run_seed=run_seed,
        data_order_seed=data_order_seed,
        physical_d6_evidence_sha256=matrix.corpus.d6_physical_evidence_sha256,
        document_multiset_sha256=_hash("confirmation-multiset"),
        ordered_raw_content_ids_sha256=_hash(f"confirmation-order-{run_seed}"),
        framed_payload_sha256=_hash(f"confirmation-payload-{run_seed}"),
        document_count=10,
        retained_text_bytes=matrix.corpus.training_realized_bytes,
    )


def _plan(matrix, order_receipt, *, optimizer_steps: int = 100) -> ConfirmationTrainingPlanV2:
    trained_tokens = optimizer_steps * 256 * 2_048
    trained_bytes = matrix.corpus.training_realized_bytes
    return ConfirmationTrainingPlanV2(
        confirmation_order_receipt_sha256=order_receipt.receipt_sha256,
        optimizer_steps=optimizer_steps,
        global_batch_sequences=256,
        sequence_length=2_048,
        compute_token_slots=trained_tokens,
        valid_prediction_count=1,
        trained_bytes=trained_bytes,
        trained_tokens=trained_tokens,
        trained_docs_full=10,
        boundary_doc_id=None,
        boundary_doc_consumed_tokens=None,
        stream_bytes=trained_bytes,
        stream_tokens=trained_tokens,
        stream_docs=10,
        dropped_bytes=0,
        dropped_tokens=0,
        dropped_docs=0,
        packed_stream_sha256=_hash("prefix-plan"),
        calibration_prefix_compute_token_slots=100 * 256 * 2_048,
        calibration_prefix_valid_prediction_count=1,
        calibration_prefix_realized_raw_bytes=trained_bytes // 4,
        calibration_prefix_document_count=1,
        calibration_prefix_packed_stream_sha256=_hash("calibration-prefix"),
        bpb_checkpoint_steps=(25, 50, optimizer_steps),
    )


def _confirmation_observations(matrix, bpb: float, execution_plan):
    total = execution_plan.training_plan.trained_bytes
    quarter = (total + 3) // 4
    half = (total + 1) // 2
    points = (
        ("after_1b", execution_plan.heldout_evaluation_steps[0], quarter - 1, quarter, bpb + 0.2),
        ("after_2b", execution_plan.heldout_evaluation_steps[1], half - 1, half, bpb + 0.1),
        (
            "terminal_realized_T",
            execution_plan.heldout_evaluation_steps[2],
            total - 1,
            total,
            bpb,
        ),
    )
    return tuple(
        BpbMilestoneReceiptV2(
            label=label,
            optimizer_step=step,
            previous_training_raw_bytes=previous,
            training_raw_bytes=current,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            strata=_strata(matrix.corpus, point_bpb),
        )
        for label, step, previous, current, point_bpb in points
    )


def _burst_receipt(execution_plan):
    attributed = (
        execution_plan.arm_mean_flops
        + execution_plan.byte_matched_optimizer_steps // 2
    ) // execution_plan.byte_matched_optimizer_steps
    return confirmation.ConfirmationBurstFlopReceiptV2(
        ordered_step_flops=(attributed,) * 100,
        prelaunch_arm_mean_flops=execution_plan.arm_mean_flops,
        byte_matched_optimizer_steps=execution_plan.byte_matched_optimizer_steps,
    )


def _flop_ledger(
    execution_plan,
    *,
    measured_flops: int | None = None,
) -> CompleteFlopLedgerV2:
    """Build exact synthetic FLOP evidence without weakening the production type."""

    steps = execution_plan.training_plan.optimizer_steps
    total = execution_plan.target_flops if measured_flops is None else measured_flops
    steady_occurrences = steps - 1
    initial_total = total - 2 * steady_occurrences
    assert initial_total >= 2
    common = {
        "batch_rows": 256,
        "sequence_length": 2_048,
        "zero_flop_profiler_operators": (),
    }
    shapes = [
        PhysicalShapeFlopReceiptV2(
            **common,
            optimizer_phase="initial",
            occurrences=1,
            profiler_rows=(
                ProfilerOperatorFlopRowV2(
                    operator="synthetic.initial",
                    flops_per_occurrence=initial_total - 1,
                ),
            ),
            unsupported_rows=(
                AnalyticUnsupportedFlopRowV2(
                    family="synthetic.initial.unsupported",
                    flops_per_occurrence=1,
                    derivation="synthetic=1",
                ),
            ),
        )
    ]
    if steady_occurrences:
        shapes.append(
            PhysicalShapeFlopReceiptV2(
                **common,
                optimizer_phase="steady",
                occurrences=steady_occurrences,
                profiler_rows=(
                    ProfilerOperatorFlopRowV2(
                        operator="synthetic.steady",
                        flops_per_occurrence=1,
                    ),
                ),
                unsupported_rows=(
                    AnalyticUnsupportedFlopRowV2(
                        family="synthetic.steady.unsupported",
                        flops_per_occurrence=1,
                        derivation="synthetic=1",
                    ),
                ),
            )
        )
    return CompleteFlopLedgerV2(
        shapes=tuple(shapes),
        optimizer_steps=steps,
        compute_token_slots=execution_plan.training_plan.compute_token_slots,
    )


def _confirmation_closure(matrix, runs, compute) -> ConfirmationEvidenceClosureV2:
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    return _confirmation_evidence_closure(matrix, selection, compute, runs)


def _synthetic_measurement(matrix, pair, execution_plan, kwargs):
    bpb = 0.90 + execution_plan.seed_slot * 0.01
    if execution_plan.vocab_size == pair[1]:
        bpb += 0.05
    plan = execution_plan.training_plan
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
        physical_d6_evidence_sha256=execution_plan.physical_d6_evidence_sha256,
        document_multiset_sha256=execution_plan.document_multiset_sha256,
        framed_payload_sha256=execution_plan.framed_payload_sha256,
        execution_plan_sha256=execution_plan.receipt_sha256,
        training_plan_sha256=plan.receipt_sha256,
        base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
        compute_attempt_id=kwargs["compute_attempt_id"],
        common_flop_budget=execution_plan.common_flop_budget,
        measured_flops=execution_plan.common_flop_budget,
        heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
        observations=_confirmation_observations(matrix, bpb, execution_plan),
        measured_a100_microseconds=100_000,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
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
        gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
    )
    ledger = _flop_ledger(execution_plan)
    return confirmation.ConfirmationPhysicalMeasurementV2(
        run=run,
        flop_ledger=ledger,
        execution_plan_sha256=execution_plan.receipt_sha256,
        base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
        training_plan_sha256=plan.receipt_sha256,
        heldout_evaluation_steps=execution_plan.heldout_evaluation_steps,
        burst_flop_receipt=_burst_receipt(execution_plan),
        physical_flop_ledger_sha256=(
            confirmation.confirmation_physical_flop_ledger_evidence_sha256_v2(
                compute_attempt_id=run.compute_attempt_id,
                execution_plan_sha256=execution_plan.receipt_sha256,
                flop_ledger=ledger,
            )
        ),
        physical_optimizer_steps=plan.optimizer_steps,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
    )


def test_rung_b_admissibility_adjusts_total_parameters_per_vocabulary() -> None:
    rows = confirmation.build_rung_b_admissibility_v2()
    assert tuple(row.vocab_size for row in rows) == (16_384, 24_576, 32_768, 49_152)
    assert tuple(row.vocabulary_parameter_count for row in rows) == tuple(
        vocab * 1_024 for vocab in (16_384, 24_576, 32_768, 49_152)
    )
    assert tuple(row.target_parameter_count for row in rows) == tuple(
        305_800_000 + (vocab - 32_768) * 1_024
        for vocab in (16_384, 24_576, 32_768, 49_152)
    )
    assert all(row.admissible for row in rows)


def test_budget_uses_minimum_floor_arm_mean_and_exact_prelaunch_horizon() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    evidence = _evidence(matrix)
    receipt = confirmation.build_confirmation_budget_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=evidence,
    )
    expected_arm_means = {
        vocab_size: sum(
            row.measured_flops for row in evidence if row.vocab_size == vocab_size
        )
        // 2
        for vocab_size in selection.compute_confirmation_pair
    }
    assert receipt.target_flops == min(expected_arm_means.values())
    for row in receipt.rows:
        assert row.arm_mean_flops == expected_arm_means[row.vocab_size]
        assert row.planned_optimizer_steps == (
            receipt.target_flops * row.byte_matched_optimizer_steps
        ) // row.arm_mean_flops


def test_confirmation_gpu_is_recorded_without_equality_binding_to_base_gpu() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    runs = tuple(
        replace(row, gpu_uuid_provenance=_CONFIRMATION_GPU)
        for row in _confirmation_runs(matrix, selection)
    )
    compute = _confirmation_compute(matrix, selection, runs)
    validated = validate_compute_confirmation_v2(
        runs,
        matrix=matrix,
        selection=selection,
        compute=compute,
        evidence_closure=_confirmation_closure(matrix, runs, compute),
    )
    assert validated.status == "GREEN_NO_REVERSAL"
    assert _base(matrix).gpu_uuid_provenance_by_attempt == (
        ("base-synthetic", _BASE_GPU),
    )
    assert {row.gpu_uuid_provenance for row in runs} == {_CONFIRMATION_GPU}


def test_full_cli_checks_parent_offline_receipt_before_runtime_or_gpu_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        full_campaign_cli,
        "require_resolved_confirmation_semantics_v2",
        lambda: None,
    )

    def reject_offline(_path: Path) -> str:
        calls.append("offline")
        raise RuntimeError("not the parent-launched unshare child")

    monkeypatch.setattr(
        full_campaign_cli,
        "assert_offline_campaign_child_v2",
        reject_offline,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "attest_gtok_training_runtime_v2",
        lambda **kwargs: calls.append("runtime"),
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "require_production_a100_v2",
        lambda _device: calls.append("gpu"),
    )
    with pytest.raises(RuntimeError, match="unshare child"):
        full_campaign_cli.main(_full_campaign_argv(tmp_path))
    assert calls == ["offline"]


def test_full_cli_base_confirmation_v_receipt_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_sha = _hash("training-runtime")
    monkeypatch.setattr(
        full_campaign_cli,
        "require_resolved_confirmation_semantics_v2",
        lambda: None,
    )
    closure_sha = _hash("code-closure")
    cpu_sha = _hash("cpu-runtime")
    runtime = SimpleNamespace(
        receipt_sha256=runtime_sha,
        environment_payload={"python": {"version": "3.11.9"}},
    )
    artifact = SimpleNamespace(
        bytes=1,
        relative_path="training/synthetic.py",
        sha256=_hash("synthetic-code"),
    )
    closure = SimpleNamespace(
        artifacts=(artifact,),
        git_commit="51c67d65",
        receipt_sha256=closure_sha,
        schema="weft1_gtok_v2_code_closure",
        status="GREEN_EXACT_CODE_CLOSURE",
    )
    frozen = SimpleNamespace(receipt_sha256=_hash("frozen-corpus"))
    precalibration = SimpleNamespace(
        receipt_sha256=_hash("precalibration"),
        cpu_runtime_identity_sha256=cpu_sha,
    )
    base = SimpleNamespace(
        compute=SimpleNamespace(receipt_sha256=_hash("base-compute")),
        matrix=SimpleNamespace(receipt_sha256=_hash("base-matrix")),
        gpu_uuid_provenance_by_attempt=(("base-run", _BASE_GPU),),
        offline_network_receipt_sha256_by_attempt=(("base-run", _OFFLINE_A),),
    )
    result = SimpleNamespace(
        compute=SimpleNamespace(receipt_sha256=_hash("confirmation-compute")),
        confirmation=SimpleNamespace(receipt_sha256=_hash("confirmation")),
        selection=SimpleNamespace(receipt_sha256=_hash("selection")),
        vocab_ext_basis=SimpleNamespace(receipt_sha256=_hash("vocab-ext")),
        vocabulary_freeze=SimpleNamespace(receipt_sha256=_hash("v-freeze")),
        gpu_uuid_provenance_by_attempt=(
            ("confirmation-run", _BASE_GPU),
        ),
        offline_network_receipt_sha256_by_attempt=(
            ("confirmation-run", _OFFLINE_A),
        ),
    )
    call_order: list[str] = []
    base_calls: list[dict[str, object]] = []
    confirmation_calls: list[dict[str, object]] = []
    offline_launches: list[str] = []
    gpu_launches: list[str] = []

    def attest_offline(_path: Path) -> str:
        value = (_OFFLINE_A, _OFFLINE_B)[len(offline_launches)]
        offline_launches.append(value)
        call_order.append("offline")
        return value

    def load_offline(_path: Path):
        return SimpleNamespace(policy_sha256=_OFFLINE_POLICY), offline_launches[-1]

    def current_gpu(**kwargs) -> str:
        value = (_BASE_GPU, _CONFIRMATION_GPU)[len(gpu_launches)]
        gpu_launches.append(value)
        return value

    monkeypatch.setattr(
        full_campaign_cli,
        "assert_offline_campaign_child_v2",
        attest_offline,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "load_offline_parent_receipt_v2",
        load_offline,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "attest_gtok_training_runtime_v2",
        lambda **kwargs: (call_order.append("runtime"), runtime)[1],
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "capture_gtok_code_closure_v2",
        lambda _root: (call_order.append("closure"), closure)[1],
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "validate_gtok_code_closure_v2",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(full_campaign_cli.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        full_campaign_cli.torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA A100-SXM4-80GB",
    )
    monkeypatch.setattr(full_campaign_cli, "require_production_a100_v2", lambda _device: None)
    monkeypatch.setattr(
        full_campaign_cli,
        "gpu_uuid_provenance_v2",
        current_gpu,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "load_frozen_screen_corpus_v2",
        lambda **kwargs: frozen,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "load_tokenizer_execution_panel_v2",
        lambda **kwargs: (
            SimpleNamespace(
                offline_network_receipt_sha256=_hash(
                    "tokenizer-offline-network"
                ),
                offline_network_policy_sha256=_hash("tokenizer-offline-policy"),
            ),
        ),
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "load_precalibration_cpu_evidence_v2",
        lambda _path: precalibration,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "cpu_runtime_identity_sha256_from_payload_v2",
        lambda _payload: cpu_sha,
    )
    monkeypatch.setattr(
        full_campaign_cli,
        "load_v4_corpus_source_v2",
        lambda *_args, **_kwargs: "physical-source",
    )

    def run_base(**kwargs):
        call_order.append("base")
        base_calls.append(kwargs)
        return base

    def run_confirmation(**kwargs):
        assert kwargs["base"] is base
        call_order.append("confirmation")
        confirmation_calls.append(kwargs)
        return result

    monkeypatch.setattr(full_campaign_cli, "run_base_campaign_v2", run_base)
    monkeypatch.setattr(
        full_campaign_cli,
        "run_compute_confirmation_and_freeze_v2",
        run_confirmation,
    )
    argv = _full_campaign_argv(tmp_path)
    assert full_campaign_cli.main(argv) == 0
    receipt = tmp_path / "campaign" / "full-campaign-launch-receipt.json"
    first = receipt.read_bytes()
    assert full_campaign_cli.main(argv) == 0
    assert receipt.read_bytes() == first
    assert call_order == [
        "offline",
        "runtime",
        "closure",
        "base",
        "confirmation",
        "offline",
        "runtime",
        "closure",
        "base",
        "confirmation",
    ]
    assert len(base_calls) == len(confirmation_calls) == 2
    assert [row["offline_network_receipt_sha256"] for row in base_calls] == [
        _OFFLINE_A,
        _OFFLINE_B,
    ]
    assert all(
        row["offline_network_policy_sha256"] == _OFFLINE_POLICY
        for row in base_calls
    )
    assert all(row["cpu_runtime_identity_sha256"] == cpu_sha for row in base_calls)
    assert [row["gpu_uuid_provenance"] for row in base_calls] == [
        _BASE_GPU,
        _CONFIRMATION_GPU,
    ]
    assert [row["offline_network_receipt_sha256"] for row in confirmation_calls] == [
        _OFFLINE_A,
        _OFFLINE_B,
    ]
    assert all(
        row["offline_network_policy_sha256"] == _OFFLINE_POLICY
        for row in confirmation_calls
    )
    assert [row["gpu_uuid_provenance"] for row in confirmation_calls] == [
        _BASE_GPU,
        _CONFIRMATION_GPU,
    ]


def test_nonexact_base_step_costs_do_not_restore_exact_reachability_gate() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    receipt = confirmation.build_confirmation_budget_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=_evidence(matrix, reachable=False),
    )
    assert all(row.planned_optimizer_steps >= 100 for row in receipt.rows)


@pytest.mark.parametrize(
    ("realized_numerator", "expected_direction"),
    ((98, "longer"), (102, "shorter")),
)
def test_retry_horizon_recomputes_both_directions_exactly(
    realized_numerator: int,
    expected_direction: str,
) -> None:
    target = 1_000_000_000_000
    original_steps = 1_000
    realized = realized_numerator * target // 100
    retry_steps = confirmation.confirmation_retry_steps_v2(
        target_flops=target,
        realized_flops=realized,
        optimizer_steps=original_steps,
    )
    assert retry_steps == target * original_steps // realized
    if expected_direction == "longer":
        assert retry_steps > original_steps
    else:
        assert retry_steps < original_steps


def test_confirmation_pair_is_winner_and_raw_runner_up() -> None:
    offsets = {
        16_384: 0.90,
        24_576: 0.89,
        32_768: 0.88,
        49_152: 0.87,
    }
    terminal_bpbs = {
        (vocab_size, seed): value
        + (0.20 if seed == SEEDS[1] else 0.0)
        for vocab_size, value in offsets.items()
        for seed in SEEDS
    }
    matrix = _matrix(terminal_bpbs)
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    assert selection.selected_vocab_size == 16_384
    assert selection.compute_confirmation_pair == (16_384, 49_152)


def test_injected_executors_end_in_non_authoritative_result_without_v_mint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    arms = _arms(matrix, tmp_path)
    source = _Source(matrix)
    evidence = _evidence(matrix)
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_materialized_confirmation_order_v4",
        lambda _root, confirmation_run_seed, data_order_seed, **_kwargs: _order(
            matrix,
            run_seed=confirmation_run_seed,
            data_order_seed=data_order_seed,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "plan_confirmation_training_prefix_v2",
        lambda *args, **kwargs: _plan(
            matrix,
            kwargs["confirmation_order_receipt"],
            optimizer_steps=kwargs["optimizer_steps"],
        ),
    )

    calls = {"full": 0}
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}

    def full(**kwargs):
        calls["full"] += 1
        execution_plan = kwargs["execution_plan"]
        bpb = 0.90 + execution_plan.seed_slot * 0.01
        if execution_plan.vocab_size == pair[1]:
            bpb += 0.05
        plan = execution_plan.training_plan
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
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=execution_plan.common_flop_budget,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            observations=_confirmation_observations(matrix, bpb, execution_plan),
            measured_a100_microseconds=100_000,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
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
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )
        ledger = _flop_ledger(execution_plan)
        return confirmation.ConfirmationPhysicalMeasurementV2(
            run=run,
            flop_ledger=ledger,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=execution_plan.training_plan.receipt_sha256,
            heldout_evaluation_steps=execution_plan.heldout_evaluation_steps,
            burst_flop_receipt=_burst_receipt(execution_plan),
            physical_flop_ledger_sha256=(
                confirmation.confirmation_physical_flop_ledger_evidence_sha256_v2(
                    compute_attempt_id=run.compute_attempt_id,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    flop_ledger=ledger,
                )
            ),
            physical_optimizer_steps=execution_plan.training_plan.optimizer_steps,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        )

    dry = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        base_flop_evidence=evidence,
        output_root=tmp_path / "dry",
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_A,
    )
    assert dry.authority_status == "NON_AUTHORITATIVE_INJECTED_CONFIRMATION_EXECUTORS"
    assert len(dry.run_receipt_sha256s) == 2
    assert "runs" not in {field.name for field in fields(dry)}
    assert "compute" not in {field.name for field in fields(dry)}
    assert "selection" not in {field.name for field in fields(dry)}
    assert "confirmation" not in {field.name for field in fields(dry)}
    assert "vocabulary_freeze" not in {field.name for field in fields(dry)}
    assert (tmp_path / "dry" / "non-authoritative-confirmation-dry-run.json").is_file()
    assert not (tmp_path / "dry" / "vocabulary-freeze.json").exists()
    assert "calibration_executor" not in inspect.signature(
        confirmation.run_compute_confirmation_and_freeze_v2
    ).parameters
    assert "full_run_executor" not in inspect.signature(
        confirmation.run_compute_confirmation_and_freeze_v2
    ).parameters
    receipt_bytes_before_gpu_relaunch = {
        path.relative_to(tmp_path / "dry").as_posix(): path.read_bytes()
        for path in (tmp_path / "dry").rglob("*.json")
    }
    repeated = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        base_flop_evidence=evidence,
        output_root=tmp_path / "dry",
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_BASE_GPU,
        offline_network_receipt_sha256=_OFFLINE_B,
    )
    assert repeated == dry
    assert calls == {"full": 2}
    assert {
        path.relative_to(tmp_path / "dry").as_posix(): path.read_bytes()
        for path in (tmp_path / "dry").rglob("*.json")
    } == receipt_bytes_before_gpu_relaunch

    tampered = tmp_path / "tampered"
    shutil.copytree(tmp_path / "dry", tampered)
    authority = tampered / "confirmation-authority.json"
    authority.write_bytes(authority.read_bytes().replace(b"NON_AUTHORITATIVE", b"XON_AUTHORITATIVE", 1))
    with pytest.raises(confirmation.GTokConfirmationV2Error, match="differs on resume"):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=source,  # type: ignore[arg-type]
            tokenizer_arms=arms,
            base_flop_evidence=evidence,
            output_root=tampered,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            full_run_executor=full,
            gpu_uuid_provenance=_CONFIRMATION_GPU,
            offline_network_receipt_sha256=_OFFLINE_B,
        )

    duplicate = tmp_path / "duplicate"
    shutil.copytree(tmp_path / "dry", duplicate)
    lifecycle = campaign.validate_lifecycle_ledger_v2(duplicate)
    original = next(
        row
        for row in lifecycle
        if row.kind == "full_run"
        and row.phase == "TERMINAL"
        and row.terminal_status == "completed"
    )
    duplicate_attempt = original.attempt_id + ".duplicate"
    campaign._append_lifecycle_event_v2(
        duplicate,
        campaign.CampaignLifecycleEventV2(
            logical_attempt_id=original.logical_attempt_id,
            attempt_id=duplicate_attempt,
            scope="confirmation",
            kind="full_run",
            phase="START",
            charged_a100_microseconds=1,
            terminal_status=None,
        ),
    )
    campaign._append_lifecycle_event_v2(
        duplicate,
        campaign.CampaignLifecycleEventV2(
            logical_attempt_id=original.logical_attempt_id,
            attempt_id=duplicate_attempt,
            scope="confirmation",
            kind="full_run",
            phase="TERMINAL",
            charged_a100_microseconds=original.charged_a100_microseconds,
            terminal_status="completed",
            completion_payload=original.completion_payload,
        ),
    )
    with pytest.raises(confirmation.GTokConfirmationV2Error, match="completed twice"):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=source,  # type: ignore[arg-type]
            tokenizer_arms=arms,
            base_flop_evidence=evidence,
            output_root=duplicate,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            full_run_executor=full,
            gpu_uuid_provenance=_CONFIRMATION_GPU,
            offline_network_receipt_sha256=_OFFLINE_B,
        )


def test_invalid_flop_band_retries_fresh_and_resumes_corrected_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    output = tmp_path / "flop-band-retry"
    arms = _arms(matrix, tmp_path)
    source = _Source(matrix)
    evidence = _evidence(matrix)
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_materialized_confirmation_order_v4",
        lambda _root, confirmation_run_seed, data_order_seed, **_kwargs: _order(
            matrix,
            run_seed=confirmation_run_seed,
            data_order_seed=data_order_seed,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "plan_confirmation_training_prefix_v2",
        lambda *args, **kwargs: _plan(
            matrix,
            kwargs["confirmation_order_receipt"],
            optimizer_steps=kwargs["optimizer_steps"],
        ),
    )

    seen_plans: list[confirmation.ConfirmationExecutionPlanV2] = []
    failed_slot_zero = False

    def full(**kwargs):
        nonlocal failed_slot_zero
        execution_plan = kwargs["execution_plan"]
        seen_plans.append(execution_plan)
        if execution_plan.seed_slot == 0 and not failed_slot_zero:
            failed_slot_zero = True
            realized = (98 * execution_plan.target_flops) // 100
            ledger = _flop_ledger(execution_plan, measured_flops=realized)
            raise confirmation.ConfirmationFlopBandViolationV2(
                realized_flops=realized,
                target_flops=execution_plan.target_flops,
                retry_steps=confirmation.confirmation_retry_steps_v2(
                    target_flops=execution_plan.target_flops,
                    realized_flops=realized,
                    optimizer_steps=execution_plan.training_plan.optimizer_steps,
                ),
                flop_ledger=ledger,
                burst_flop_receipt=_burst_receipt(execution_plan),
            )
        bpb = 0.90 + execution_plan.seed_slot * 0.01
        if execution_plan.vocab_size == pair[1]:
            bpb += 0.05
        plan = execution_plan.training_plan
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
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=execution_plan.common_flop_budget,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            observations=_confirmation_observations(matrix, bpb, execution_plan),
            measured_a100_microseconds=100_000,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
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
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )
        ledger = _flop_ledger(execution_plan)
        return confirmation.ConfirmationPhysicalMeasurementV2(
            run=run,
            flop_ledger=ledger,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=plan.receipt_sha256,
            heldout_evaluation_steps=execution_plan.heldout_evaluation_steps,
            burst_flop_receipt=_burst_receipt(execution_plan),
            physical_flop_ledger_sha256=(
                confirmation.confirmation_physical_flop_ledger_evidence_sha256_v2(
                    compute_attempt_id=run.compute_attempt_id,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    flop_ledger=ledger,
                )
            ),
            physical_optimizer_steps=plan.optimizer_steps,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        )

    first = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        base_flop_evidence=evidence,
        output_root=output,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_A,
    )
    assert first.authority_status.startswith("NON_AUTHORITATIVE")
    assert len(seen_plans) == 3
    initial, retry, other_slot = seen_plans
    assert (initial.seed_slot, retry.seed_slot, other_slot.seed_slot) == (0, 0, 1)
    assert retry.training_plan.optimizer_steps == (
        initial.target_flops * initial.training_plan.optimizer_steps
    ) // ((98 * initial.target_flops) // 100)
    assert retry.training_plan.optimizer_steps != initial.training_plan.optimizer_steps
    assert (
        retry.registry_key,
        retry.seed,
        retry.initialization_seed,
        retry.data_order_seed,
        retry.data_order_sha256,
        retry.confirmation_order_receipt_sha256,
    ) == (
        initial.registry_key,
        initial.seed,
        initial.initialization_seed,
        initial.data_order_seed,
        initial.data_order_sha256,
        initial.confirmation_order_receipt_sha256,
    )
    attempts = campaign._load_persisted_attempts_v2(output)
    assert [row.status for row in attempts] == ["failed", "completed", "completed"]
    assert attempts[1].attempt_id == attempts[0].attempt_id + ".retry-1"
    assert sum(row.status == "completed" for row in attempts) == 2
    invalid_artifact = output / f"invalid-flop-band-{attempts[0].attempt_id}.json"
    assert invalid_artifact.is_file()

    calls_before_resume = len(seen_plans)
    resumed = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        base_flop_evidence=evidence,
        output_root=output,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_A,
    )
    assert resumed == first
    assert len(seen_plans) == calls_before_resume

    stored = json.loads(invalid_artifact.read_text(encoding="utf-8"))
    persisted_burst = dict(stored["passed_burst_flop_receipt"])
    persisted_burst["prelaunch_arm_mean_flops"] *= 2
    persisted_burst["byte_matched_optimizer_steps"] *= 2
    burst_receipt = confirmation.ConfirmationBurstFlopReceiptV2(
        ordered_step_flops=tuple(persisted_burst["ordered_step_flops"]),
        prelaunch_arm_mean_flops=persisted_burst["prelaunch_arm_mean_flops"],
        byte_matched_optimizer_steps=persisted_burst[
            "byte_matched_optimizer_steps"
        ],
    )
    stored["passed_burst_flop_receipt"] = persisted_burst
    stored["passed_burst_receipt_sha256"] = burst_receipt.receipt_sha256
    stored["passed_physical_burst_evidence_sha256"] = (
        confirmation.confirmation_physical_burst_evidence_sha256_v2(
            compute_attempt_id=attempts[0].attempt_id,
            execution_plan_sha256=seen_plans[0].receipt_sha256,
            burst=burst_receipt,
        )
    )
    invalid_artifact.write_bytes(confirmation.canonical_json_bytes(stored) + b"\n")
    with pytest.raises(
        confirmation.GTokConfirmationV2Error,
        match="retry chain changed stable evidence",
    ):
        confirmation._load_retry_chain_v2(output, seen_plans[0])


def test_multi_retry_chain_fails_closed_when_later_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    output = tmp_path / "multi-retry-missing-artifact"
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_materialized_confirmation_order_v4",
        lambda _root, confirmation_run_seed, data_order_seed, **_kwargs: _order(
            matrix,
            run_seed=confirmation_run_seed,
            data_order_seed=data_order_seed,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "plan_confirmation_training_prefix_v2",
        lambda *args, **kwargs: _plan(
            matrix,
            kwargs["confirmation_order_receipt"],
            optimizer_steps=kwargs["optimizer_steps"],
        ),
    )
    seen_plans: list[confirmation.ConfirmationExecutionPlanV2] = []
    failed_slot_zero = 0

    def full(**kwargs):
        nonlocal failed_slot_zero
        execution_plan = kwargs["execution_plan"]
        seen_plans.append(execution_plan)
        if execution_plan.seed_slot == 0 and failed_slot_zero < 2:
            failed_slot_zero += 1
            realized = 98 * execution_plan.target_flops // 100
            ledger = _flop_ledger(execution_plan, measured_flops=realized)
            raise confirmation.ConfirmationFlopBandViolationV2(
                realized_flops=realized,
                target_flops=execution_plan.target_flops,
                retry_steps=confirmation.confirmation_retry_steps_v2(
                    target_flops=execution_plan.target_flops,
                    realized_flops=realized,
                    optimizer_steps=execution_plan.training_plan.optimizer_steps,
                ),
                flop_ledger=ledger,
                burst_flop_receipt=_burst_receipt(execution_plan),
            )
        return _synthetic_measurement(matrix, pair, execution_plan, kwargs)

    result = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=_Source(matrix),  # type: ignore[arg-type]
        tokenizer_arms=_arms(matrix, tmp_path),
        base_flop_evidence=_evidence(matrix),
        output_root=output,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_A,
    )
    assert result.authority_status.startswith("NON_AUTHORITATIVE")
    assert [(row.seed_slot, row.training_plan.optimizer_steps) for row in seen_plans] == [
        (0, 100),
        (0, 102),
        (0, 104),
        (1, 100),
    ]
    attempts = campaign._load_persisted_attempts_v2(output)
    failed = [row for row in attempts if row.status == "failed"]
    assert [row.attempt_id for row in failed] == [
        failed[0].attempt_id,
        failed[0].attempt_id + ".retry-1",
    ]
    artifacts = [
        output / f"invalid-flop-band-{attempt.attempt_id}.json"
        for attempt in failed
    ]
    assert all(path.is_file() for path in artifacts)
    preflight = json.loads(
        (output / "confirmation-preflight.json").read_text(encoding="utf-8")
    )
    projection = ArmCalibrationProjectionV2(
        **preflight["payload"]["calibrations"][0]
    )
    chain, recovered_plan = confirmation._load_retry_chain_v2(
        output,
        seen_plans[0],
        projection,
    )
    assert [row.ordinal for row in chain] == [0, 1]
    assert recovered_plan.receipt_sha256 == seen_plans[2].receipt_sha256
    for row, path in zip(chain, artifacts, strict=True):
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["retry_execution_plan_sha256"] == row.retry_plan.receipt_sha256
        assert stored["retry_projected_run_a100_microseconds"] == (
            confirmation._attempt_projection_a100_microseconds_v2(
                projection,
                row.retry_plan,
            )
        )
    artifacts[1].unlink()
    with pytest.raises(
        confirmation.GTokConfirmationV2Error,
        match="failed confirmation terminals and invalid-band artifacts differ",
    ):
        confirmation._load_retry_chain_v2(output, seen_plans[0])


def test_q4_burst_gate_persists_ordered_evidence_and_stop_physical_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    output = tmp_path / "q4-burst-stop"
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_materialized_confirmation_order_v4",
        lambda _root, confirmation_run_seed, data_order_seed, **_kwargs: _order(
            matrix,
            run_seed=confirmation_run_seed,
            data_order_seed=data_order_seed,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "plan_confirmation_training_prefix_v2",
        lambda *args, **kwargs: _plan(
            matrix,
            kwargs["confirmation_order_receipt"],
            optimizer_steps=kwargs["optimizer_steps"],
        ),
    )
    observed: dict[str, object] = {}

    def full(**kwargs):
        execution_plan = kwargs["execution_plan"]
        nominal = (
            execution_plan.arm_mean_flops
            + execution_plan.byte_matched_optimizer_steps // 2
        ) // execution_plan.byte_matched_optimizer_steps
        ordered_step_flops = tuple(
            nominal + index % 3 for index in range(99)
        ) + (nominal + nominal // 50,)
        gate = confirmation.ConfirmationBurstGateEvidenceV2(
            ordered_step_flops=ordered_step_flops,
            prelaunch_arm_mean_flops=execution_plan.arm_mean_flops,
            byte_matched_optimizer_steps=(
                execution_plan.byte_matched_optimizer_steps
            ),
        )
        assert gate.status == "STOP_RANGE_OVER_MEDIAN_ABOVE_ONE_PERCENT"
        observed["gate"] = gate
        observed["attempt_id"] = kwargs["compute_attempt_id"]
        observed["execution_plan_sha256"] = execution_plan.receipt_sha256
        raise confirmation.ConfirmationBurstGateViolationV2(gate)

    with pytest.raises(
        GTokV2Stop,
        match="confirmation Q4 burst gate fired; return to strategy",
    ):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=_Source(matrix),  # type: ignore[arg-type]
            tokenizer_arms=_arms(matrix, tmp_path),
            base_flop_evidence=_evidence(matrix),
            output_root=output,
            training_runtime_receipt_sha256=(
                matrix.training_runtime_receipt_sha256
            ),
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            full_run_executor=full,
            gpu_uuid_provenance=_CONFIRMATION_GPU,
            offline_network_receipt_sha256=_OFFLINE_A,
        )

    gate = observed["gate"]
    assert isinstance(gate, confirmation.ConfirmationBurstGateEvidenceV2)
    evidence_path = next(output.glob("invalid-confirmation-burst-*.json"))
    evidence_raw = evidence_path.read_bytes()
    evidence = json.loads(evidence_raw)
    assert tuple(evidence["burst_gate_evidence"]["ordered_step_flops"]) == (
        gate.ordered_step_flops
    )
    assert evidence["burst_gate_evidence"] == {
        "byte_matched_optimizer_steps": gate.byte_matched_optimizer_steps,
        "ordered_step_flops": list(gate.ordered_step_flops),
        "prelaunch_arm_mean_flops": gate.prelaunch_arm_mean_flops,
    }
    assert evidence["burst_gate_receipt_sha256"] == gate.receipt_sha256
    assert evidence["status"] == gate.status
    assert evidence["failed_execution_plan_sha256"] == observed[
        "execution_plan_sha256"
    ]
    physical_sha256 = hashlib.sha256(evidence_raw).hexdigest()
    stop = json.loads((output / "campaign-stop.json").read_bytes())["payload"]
    assert stop["reason"] == gate.status
    assert stop["decision_evidence_receipt_sha256"] == gate.receipt_sha256
    assert stop["decision_evidence_physical_sha256"] == physical_sha256
    attempts = campaign._load_persisted_attempts_v2(output)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == observed["attempt_id"]
    assert attempts[0].status == "failed"
    terminal = next(
        row
        for row in campaign.validate_lifecycle_ledger_v2(output)
        if row.attempt_id == observed["attempt_id"] and row.phase == "TERMINAL"
    )
    assert terminal.terminal_status == "failed"
    assert evidence["failed_terminal_lifecycle_event_sha256"] == (
        terminal.receipt_sha256
    )


def test_hard_kill_relaunch_charges_orphan_and_retries_fresh_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "hard-kill"
    repository = Path(__file__).resolve().parents[1]
    child = f'''\
import os
from pathlib import Path
import training.weft1_gtok_confirmation_v2 as c
from training.weft1_gtok_campaign_v2 import TokenizerExecutionArmV2
from tests.test_weft1_gtok_confirmation_v2 import _arms, _base, _evidence, _hash, _matrix, _order, _plan, _Source, _BASE_GPU, _OFFLINE_A
m = _matrix()
TokenizerExecutionArmV2.load = lambda self: object()
c.build_materialized_confirmation_order_v4 = lambda root, confirmation_run_seed, data_order_seed, **kwargs: _order(m, run_seed=confirmation_run_seed, data_order_seed=data_order_seed)
c.plan_confirmation_training_prefix_v2 = lambda *args, **kwargs: _plan(m, kwargs["confirmation_order_receipt"], optimizer_steps=kwargs["optimizer_steps"])
def hard_kill(**kwargs):
    os._exit(77)
c.run_compute_confirmation_dry_run_v2(
    base=_base(m), source=_Source(m),
    tokenizer_arms=_arms(m, Path({str(tmp_path)!r})),
    base_flop_evidence=_evidence(m),
    output_root=Path({str(output)!r}),
    training_runtime_receipt_sha256=m.training_runtime_receipt_sha256,
    code_closure_receipt_sha256=m.code_closure_receipt_sha256,
    full_run_executor=hard_kill,
    gpu_uuid_provenance=_BASE_GPU,
    offline_network_receipt_sha256=_OFFLINE_A,
)
'''
    killed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=repository,
        check=False,
    )
    assert killed.returncode == 77
    assert not (output / "campaign-stop.json").exists()
    before = campaign.validate_lifecycle_ledger_v2(output)
    orphan_start = next(
        row
        for row in before
        if row.kind == "full_run" and row.phase == "START"
    )

    matrix = _matrix()
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    arms = _arms(matrix, tmp_path)
    source = _Source(matrix)
    calls = {"full": 0, "attempt_ids": []}
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_materialized_confirmation_order_v4",
        lambda _root, confirmation_run_seed, data_order_seed, **_kwargs: _order(
            matrix,
            run_seed=confirmation_run_seed,
            data_order_seed=data_order_seed,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "plan_confirmation_training_prefix_v2",
        lambda *args, **kwargs: _plan(
            matrix,
            kwargs["confirmation_order_receipt"],
            optimizer_steps=kwargs["optimizer_steps"],
        ),
    )

    def full(**kwargs):
        calls["full"] += 1
        calls["attempt_ids"].append(kwargs["compute_attempt_id"])
        execution_plan = kwargs["execution_plan"]
        if calls["full"] == 1:
            realized = 98 * execution_plan.target_flops // 100
            ledger = _flop_ledger(execution_plan, measured_flops=realized)
            raise confirmation.ConfirmationFlopBandViolationV2(
                realized_flops=realized,
                target_flops=execution_plan.target_flops,
                retry_steps=confirmation.confirmation_retry_steps_v2(
                    target_flops=execution_plan.target_flops,
                    realized_flops=realized,
                    optimizer_steps=execution_plan.training_plan.optimizer_steps,
                ),
                flop_ledger=ledger,
                burst_flop_receipt=_burst_receipt(execution_plan),
            )
        bpb = 0.90 + execution_plan.seed_slot * 0.01
        if execution_plan.vocab_size == pair[1]:
            bpb += 0.05
        plan = execution_plan.training_plan
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
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=execution_plan.common_flop_budget,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            observations=_confirmation_observations(matrix, bpb, execution_plan),
            measured_a100_microseconds=100_000,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
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
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )
        ledger = _flop_ledger(execution_plan)
        return confirmation.ConfirmationPhysicalMeasurementV2(
            run=run,
            flop_ledger=ledger,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=execution_plan.training_plan.receipt_sha256,
            heldout_evaluation_steps=execution_plan.heldout_evaluation_steps,
            burst_flop_receipt=_burst_receipt(execution_plan),
            physical_flop_ledger_sha256=(
                confirmation.confirmation_physical_flop_ledger_evidence_sha256_v2(
                    compute_attempt_id=run.compute_attempt_id,
                    execution_plan_sha256=execution_plan.receipt_sha256,
                    flop_ledger=ledger,
                )
            ),
            physical_optimizer_steps=execution_plan.training_plan.optimizer_steps,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        )

    result = confirmation.run_compute_confirmation_dry_run_v2(
        base=_base(matrix),
        source=source,  # type: ignore[arg-type]
        tokenizer_arms=arms,
        base_flop_evidence=_evidence(matrix),
        output_root=output,
        training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
        code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_B,
    )
    assert result.authority_status.startswith("NON_AUTHORITATIVE")
    assert calls["full"] == 3
    assert calls["attempt_ids"][0] == orphan_start.attempt_id + ".retry-1"
    assert calls["attempt_ids"][1] == orphan_start.attempt_id + ".retry-2"
    attempts = campaign._load_persisted_attempts_v2(output)
    orphan = next(row for row in attempts if row.attempt_id == orphan_start.attempt_id)
    invalid_band = next(
        row for row in attempts if row.attempt_id == orphan_start.attempt_id + ".retry-1"
    )
    retry = next(
        row for row in attempts if row.attempt_id == orphan_start.attempt_id + ".retry-2"
    )
    assert orphan.status == "preempted"
    assert orphan.consumed_a100_microseconds >= orphan_start.charged_a100_microseconds
    assert invalid_band.status == "failed"
    assert retry.status == "completed"
    retry_artifact = output / f"invalid-flop-band-{invalid_band.attempt_id}.json"
    assert retry_artifact.is_file()
    assert '"correction_ordinal":0' in retry_artifact.read_text(encoding="utf-8")
    lifecycle = campaign.validate_lifecycle_ledger_v2(output)
    assert {
        row.gpu_uuid_provenance
        for row in lifecycle
        if row.attempt_id == orphan.attempt_id
    } == {_BASE_GPU}
    assert {
        row.gpu_uuid_provenance
        for row in lifecycle
        if row.attempt_id == retry.attempt_id
    } == {_CONFIRMATION_GPU}
    assert {
        row.offline_network_launch_receipt_sha256
        for row in lifecycle
        if row.attempt_id == orphan.attempt_id
    } == {_OFFLINE_A}
    assert {
        row.offline_network_launch_receipt_sha256
        for row in lifecycle
        if row.attempt_id == retry.attempt_id
    } == {_OFFLINE_B}
    assert sum(row.consumed_a100_microseconds for row in attempts) > sum(
        row.consumed_a100_microseconds
        for row in attempts
        if row.attempt_id != orphan.attempt_id
    )
