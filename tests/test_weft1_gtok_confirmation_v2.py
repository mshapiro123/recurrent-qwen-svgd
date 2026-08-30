from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
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
from training.weft1_gtok_training_v2 import CalibrationMeasurementV2, TrainingPlanV2
from training.weft1_gtok_v2_contract import (
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
        self.physical_d6_evidence_sha256 = matrix.corpus.d6_physical_evidence_sha256
        self.training_raw_bytes = matrix.corpus.training_realized_bytes
        self.heldout_raw_bytes_by_stratum = matrix.corpus.heldout_denominator_signature
        self.training_order_receipts = tuple(
            (seed, 20_000 + seed, _hash(f"data-order-{seed}"))
            for seed in matrix.seeds
        )

    def training_documents(self, seed: int):
        raise AssertionError("prefix planning is replaced by a bound synthetic plan")


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


def _plan() -> TrainingPlanV2:
    return TrainingPlanV2(
        optimizer_steps=100,
        compute_token_slots=100 * 256 * 2_048,
        valid_prediction_count=1,
        realized_raw_bytes=1,
        document_count=1,
        packed_stream_sha256=_hash("prefix-plan"),
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


def test_reachability_uses_minimum_of_selected_four_base_runs_and_exact_prefix() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    )
    evidence = _evidence(matrix)
    receipt = confirmation.build_confirmation_reachability_v2(
        matrix=matrix,
        selection=selection,
        base_flop_evidence=evidence,
    )
    assert receipt.common_flop_budget == min(row.measured_flops for row in evidence)
    assert receipt.all_exact_reachable
    assert {row.reached_optimizer_steps for row in receipt.rows} == {100}
    assert all(row.nearest_lower_flops == receipt.common_flop_budget for row in receipt.rows)


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


def test_unreachable_common_budget_stops_before_any_calibration_and_persists_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    arms = _arms(matrix, tmp_path)
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    called = False

    def calibrate(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreachable budget must stop before calibration")

    with pytest.raises(GTokV2Stop, match="unreachable"):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=_Source(matrix),  # type: ignore[arg-type]
            tokenizer_arms=arms,
            base_flop_evidence=_evidence(matrix, reachable=False),
            output_root=tmp_path / "unreachable",
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            calibration_executor=calibrate,
            full_run_executor=lambda **kwargs: None,  # type: ignore[arg-type]
        )
    assert not called
    assert (tmp_path / "unreachable" / "campaign-stop.json").is_file()
    assert not (tmp_path / "unreachable" / "campaign-events.sqlite3").exists()
    with pytest.raises(GTokV2Stop, match="governed confirmation STOP"):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=_Source(matrix),  # type: ignore[arg-type]
            tokenizer_arms=arms,
            base_flop_evidence=_evidence(matrix, reachable=False),
            output_root=tmp_path / "unreachable",
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            calibration_executor=calibrate,
            full_run_executor=lambda **kwargs: None,  # type: ignore[arg-type]
        )


def test_selected_vocabulary_outside_raw_pair_stops_without_changing_pair(
    tmp_path: Path,
) -> None:
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
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    called = False

    def calibrate(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("out-of-pair selection must stop before calibration")

    root = tmp_path / "out-of-pair"
    with pytest.raises(GTokV2Stop, match="outside the unchanged raw confirmation pair"):
        confirmation.run_compute_confirmation_dry_run_v2(
            base=_base(matrix),
            source=_Source(matrix),  # type: ignore[arg-type]
            tokenizer_arms=_arms(matrix, tmp_path),
            base_flop_evidence=_evidence(matrix),
            output_root=root,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            calibration_executor=calibrate,
            full_run_executor=lambda **kwargs: None,  # type: ignore[arg-type]
        )
    assert not called
    assert b"SELECTED_VOCAB_OUTSIDE_RAW_CONFIRMATION_PAIR" in (
        root / "campaign-stop.json"
    ).read_bytes()


def test_injected_executors_end_in_non_authoritative_result_without_v_mint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    arms = _arms(matrix, tmp_path)
    source = _Source(matrix)
    evidence = _evidence(matrix)
    plan = _plan()
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_confirmation_prefix_plan_v2",
        lambda **kwargs: plan,
    )

    def calibrate(**kwargs):
        calls["calibration"] += 1
        return CalibrationMeasurementV2(
            steps=100,
            warmup_steps=20,
            measured_steps=80,
            measured_tokens=plan.compute_token_slots,
            measured_a100_microseconds=1_000_000,
            charged_a100_microseconds=1_200_000,
            measured_heldout_evaluation_a100_microseconds=100_000,
            heldout_evaluations_per_full_run=3,
            measured_output_surface_a100_microseconds=100_000,
            output_surface_benchmarks_per_full_run=1,
            planned_tokens_per_run=plan.compute_token_slots,
            shared_initial_state_sha256=_hash("shared-calibration"),
        )

    calls = {"calibration": 0, "full": 0}
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}

    def full(**kwargs):
        calls["full"] += 1
        execution_plan = kwargs["execution_plan"]
        bpb = 0.90 + matrix.seeds.index(execution_plan.seed) * 0.01
        if execution_plan.vocab_size == pair[1]:
            bpb += 0.05
        run = ComputeConfirmationRunV2(
            vocab_size=execution_plan.vocab_size,
            seed=execution_plan.seed,
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=execution_plan.common_flop_budget,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            strata=_strata(matrix.corpus, bpb),
            measured_a100_microseconds=100_000,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )
        return confirmation.ConfirmationPhysicalMeasurementV2(
            run=run,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=execution_plan.training_plan.receipt_sha256,
            heldout_evaluation_steps=(34, 67, 100),
            physical_flop_ledger_sha256=_hash(
                f"confirmation-ledger-{run.vocab_size}-{run.seed}"
            ),
            physical_optimizer_steps=100,
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
        calibration_executor=calibrate,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_A,
    )
    assert dry.authority_status == "NON_AUTHORITATIVE_INJECTED_CONFIRMATION_EXECUTORS"
    assert len(dry.run_receipt_sha256s) == 4
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
        calibration_executor=calibrate,
        full_run_executor=full,
        gpu_uuid_provenance=_BASE_GPU,
        offline_network_receipt_sha256=_OFFLINE_B,
    )
    assert repeated == dry
    assert calls == {"calibration": 2, "full": 4}
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
            calibration_executor=calibrate,
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
            calibration_executor=calibrate,
            full_run_executor=full,
            gpu_uuid_provenance=_CONFIRMATION_GPU,
            offline_network_receipt_sha256=_OFFLINE_B,
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
from training.weft1_gtok_training_v2 import CalibrationMeasurementV2
from tests.test_weft1_gtok_confirmation_v2 import _arms, _base, _evidence, _hash, _matrix, _plan, _Source, _BASE_GPU, _OFFLINE_A
m = _matrix()
p = _plan()
TokenizerExecutionArmV2.load = lambda self: object()
c.build_confirmation_prefix_plan_v2 = lambda **kwargs: p
def calibrate(**kwargs):
    return CalibrationMeasurementV2(
        steps=100, warmup_steps=20, measured_steps=80,
        measured_tokens=p.compute_token_slots,
        measured_a100_microseconds=20000000,
        charged_a100_microseconds=20200000,
        measured_heldout_evaluation_a100_microseconds=100000,
        heldout_evaluations_per_full_run=3,
        measured_output_surface_a100_microseconds=100000,
        output_surface_benchmarks_per_full_run=1,
        planned_tokens_per_run=p.compute_token_slots,
        shared_initial_state_sha256=_hash("shared-calibration"),
    )
def hard_kill(**kwargs):
    os._exit(77)
c.run_compute_confirmation_dry_run_v2(
    base=_base(m), source=_Source(m),
    tokenizer_arms=_arms(m, Path({str(tmp_path)!r})),
    base_flop_evidence=_evidence(m),
    output_root=Path({str(output)!r}),
    training_runtime_receipt_sha256=m.training_runtime_receipt_sha256,
    code_closure_receipt_sha256=m.code_closure_receipt_sha256,
    calibration_executor=calibrate,
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
    plan = _plan()
    pair = select_vocabulary_v2(
        matrix,
        admissibility=confirmation.build_rung_b_admissibility_v2(),
    ).compute_confirmation_pair
    arms = _arms(matrix, tmp_path)
    source = _Source(matrix)
    calls = {"calibration": 0, "full": 0, "attempt_ids": []}
    monkeypatch.setattr(TokenizerExecutionArmV2, "load", lambda self: object())
    monkeypatch.setattr(
        confirmation,
        "build_confirmation_prefix_plan_v2",
        lambda **kwargs: plan,
    )

    def calibrate(**kwargs):
        calls["calibration"] += 1
        raise AssertionError("completed calibration must be skipped on relaunch")

    def full(**kwargs):
        calls["full"] += 1
        calls["attempt_ids"].append(kwargs["compute_attempt_id"])
        execution_plan = kwargs["execution_plan"]
        bpb = 0.90 + matrix.seeds.index(execution_plan.seed) * 0.01
        if execution_plan.vocab_size == pair[1]:
            bpb += 0.05
        run = ComputeConfirmationRunV2(
            vocab_size=execution_plan.vocab_size,
            seed=execution_plan.seed,
            base_run_receipt_sha256=kwargs["base_run_receipt_sha256"],
            compute_attempt_id=kwargs["compute_attempt_id"],
            common_flop_budget=execution_plan.common_flop_budget,
            measured_flops=execution_plan.common_flop_budget,
            heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
            strata=_strata(matrix.corpus, bpb),
            measured_a100_microseconds=100_000,
            training_runtime_receipt_sha256=matrix.training_runtime_receipt_sha256,
            code_closure_receipt_sha256=matrix.code_closure_receipt_sha256,
            gpu_uuid_provenance=kwargs["gpu_uuid_provenance"],
        )
        return confirmation.ConfirmationPhysicalMeasurementV2(
            run=run,
            execution_plan_sha256=execution_plan.receipt_sha256,
            base_flop_evidence_sha256=execution_plan.base_flop_evidence_sha256,
            training_plan_sha256=execution_plan.training_plan.receipt_sha256,
            heldout_evaluation_steps=(34, 67, 100),
            physical_flop_ledger_sha256=_hash(
                f"resume-ledger-{{run.vocab_size}}-{{run.seed}}"
            ),
            physical_optimizer_steps=100,
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
        calibration_executor=calibrate,
        full_run_executor=full,
        gpu_uuid_provenance=_CONFIRMATION_GPU,
        offline_network_receipt_sha256=_OFFLINE_B,
    )
    assert result.authority_status.startswith("NON_AUTHORITATIVE")
    assert calls["calibration"] == 0
    assert calls["full"] == 4
    assert calls["attempt_ids"][0] == orphan_start.attempt_id + ".retry-1"
    attempts = campaign._load_persisted_attempts_v2(output)
    orphan = next(row for row in attempts if row.attempt_id == orphan_start.attempt_id)
    retry = next(
        row for row in attempts if row.attempt_id == orphan_start.attempt_id + ".retry-1"
    )
    assert orphan.status == "preempted"
    assert orphan.consumed_a100_microseconds >= orphan_start.charged_a100_microseconds
    assert retry.status == "completed"
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
