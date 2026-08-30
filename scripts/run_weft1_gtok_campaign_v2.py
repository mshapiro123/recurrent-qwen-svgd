#!/usr/bin/env python3
"""Run the governed WEFT-1 G-TOK base campaign, with no seed override surface."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_campaign_v2 import (  # noqa: E402
    GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
    GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
    GTOK_GOVERNED_TRAINING_SEEDS_V2,
    GTOK_MICROBATCH_SEQUENCES_V2,
    load_tokenizer_execution_panel_v2,
    load_precalibration_cpu_evidence_v2,
    require_resolved_confirmation_semantics_v2,
    run_base_campaign_v2,
)
from training.weft1_gtok_contract import canonical_json_bytes  # noqa: E402
from training.weft1_gtok_code_closure_v2 import (  # noqa: E402
    capture_gtok_code_closure_v2,
    validate_gtok_code_closure_v2,
)
from training.weft1_gtok_pb_adapter_v2 import (  # noqa: E402
    load_frozen_screen_corpus_v2,
)
from training.weft1_gtok_offline_v2 import (  # noqa: E402
    assert_offline_campaign_child_v2,
    load_offline_parent_receipt_v2,
)
from training.weft1_gtok_training_v2 import (  # noqa: E402
    load_v4_corpus_source_v2,
    require_production_a100_v2,
)
from training.weft1_gtok_runtime_v2 import (  # noqa: E402
    attest_gtok_training_runtime_v2,
    cpu_runtime_identity_sha256_from_payload_v2,
    gpu_uuid_provenance_v2,
)
from training.weft1_gtok_v2_contract import gtok_v2_bound_sha256  # noqa: E402


def _positive_divisor_of_256(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("microbatch size must be an integer") from error
    if parsed != GTOK_MICROBATCH_SEQUENCES_V2:
        raise argparse.ArgumentTypeError("G-TOK v2 requires microbatch-sequences=8")
    return parsed


def _exclusive_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError("stored production launch receipt differs on relaunch")
        return hashlib.sha256(payload).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--gate-bundle", type=Path, required=True)
    parser.add_argument("--c2-evidence", type=Path, required=True)
    parser.add_argument("--decon-receipt", type=Path, required=True)
    parser.add_argument("--training-requirements-lock", type=Path, required=True)
    parser.add_argument("--runtime-build-receipt", type=Path, required=True)
    parser.add_argument("--pa-runtime-build-receipt", type=Path, required=True)
    parser.add_argument("--training-runtime-binding", type=Path, required=True)
    parser.add_argument("--offline-network-receipt", type=Path, required=True)
    parser.add_argument("--precalibration-cpu-evidence", type=Path, required=True)
    parser.add_argument(
        "--precalibration-offline-network-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--tokenizer-panel-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-offline-network-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--microbatch-sequences", type=_positive_divisor_of_256, required=True)
    parser.add_argument("--cuda-device-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.cuda_device_index < 0:
        raise ValueError("CUDA device index must be non-negative")
    if arguments.output_root.is_symlink():
        raise FileExistsError("campaign output root may not be a symlink")
    require_resolved_confirmation_semantics_v2()

    offline_network_receipt_sha256 = assert_offline_campaign_child_v2(
        arguments.offline_network_receipt
    )
    offline_parent_receipt, observed_offline_sha256 = load_offline_parent_receipt_v2(
        arguments.offline_network_receipt
    )
    if observed_offline_sha256 != offline_network_receipt_sha256:
        raise RuntimeError("offline launch receipt changed after child attestation")
    offline_network_policy_sha256 = offline_parent_receipt.policy_sha256
    runtime = attest_gtok_training_runtime_v2(
        binding_path=arguments.training_runtime_binding,
        requirements_lock=arguments.training_requirements_lock,
        runtime_build_receipt=arguments.runtime_build_receipt,
        pa_runtime_build_receipt=arguments.pa_runtime_build_receipt,
        device_index=arguments.cuda_device_index,
    )
    code_closure = capture_gtok_code_closure_v2(REPOSITORY_ROOT)
    validate_gtok_code_closure_v2(code_closure, repository_root=REPOSITORY_ROOT)
    device = torch.device(f"cuda:{arguments.cuda_device_index}")
    torch.cuda.set_device(device)
    require_production_a100_v2(device)
    gpu_uuid = gpu_uuid_provenance_v2(device_index=arguments.cuda_device_index)

    frozen = load_frozen_screen_corpus_v2(
        materialization_root=arguments.corpus_root,
        freeze_receipt_path=arguments.freeze_receipt,
        gate_bundle_path=arguments.gate_bundle,
        c2_evidence_path=arguments.c2_evidence,
        decon_receipt_path=arguments.decon_receipt,
    )
    tokenizer_arms = load_tokenizer_execution_panel_v2(
        panel_receipt_path=arguments.tokenizer_panel_receipt,
        artifact_root=arguments.tokenizer_artifact_root,
        offline_parent_receipt_path=arguments.tokenizer_offline_network_receipt,
        corpus=frozen,
    )
    precalibration_cpu_evidence = load_precalibration_cpu_evidence_v2(
        arguments.precalibration_cpu_evidence
    )
    cpu_runtime_identity_sha256 = cpu_runtime_identity_sha256_from_payload_v2(
        runtime.environment_payload
    )
    if (
        precalibration_cpu_evidence.cpu_runtime_identity_sha256
        != cpu_runtime_identity_sha256
    ):
        raise RuntimeError(
            "CPU precompute environment differs from the attested A100 training venv"
        )
    with tempfile.TemporaryDirectory(prefix="weft1-gtok-source-") as directory:
        source = load_v4_corpus_source_v2(
            arguments.corpus_root,
            sqlite_path=Path(directory) / "physical-d6.sqlite",
        )
    result = run_base_campaign_v2(
        corpus=frozen,
        source=source,
        tokenizer_arms=tokenizer_arms,
        seeds=GTOK_GOVERNED_TRAINING_SEEDS_V2,
        initialization_seeds=GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
        data_order_seeds=GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
        output_root=arguments.output_root,
        device=device,
        microbatch_sequences=arguments.microbatch_sequences,
        training_runtime_receipt_sha256=runtime.receipt_sha256,
        code_closure_receipt_sha256=code_closure.receipt_sha256,
        code_closure_receipt=code_closure,
        repository_root=REPOSITORY_ROOT,
        offline_network_receipt_sha256=offline_network_receipt_sha256,
        offline_network_policy_sha256=offline_network_policy_sha256,
        gpu_uuid_provenance=gpu_uuid,
        precalibration_cpu_evidence=precalibration_cpu_evidence,
        precalibration_offline_parent_receipt_path=(
            arguments.precalibration_offline_network_receipt
        ),
        cpu_runtime_identity_sha256=cpu_runtime_identity_sha256,
    )
    panel_raw = arguments.tokenizer_panel_receipt.read_bytes()
    launch_payload = {
        "campaign_matrix_receipt_sha256": result.matrix.receipt_sha256,
        "corpus_receipt_sha256": frozen.receipt_sha256,
        "precalibration_cpu_evidence_receipt_sha256": (
            precalibration_cpu_evidence.receipt_sha256
        ),
        "governed_data_order_seeds": GTOK_GOVERNED_DATA_ORDER_SEEDS_V2,
        "governed_initialization_seeds": GTOK_GOVERNED_INITIALIZATION_SEEDS_V2,
        "governed_training_seeds": GTOK_GOVERNED_TRAINING_SEEDS_V2,
        "microbatch_sequences": arguments.microbatch_sequences,
        "offline_network_policy_sha256": offline_network_policy_sha256,
        "offline_network_receipt_sha256_by_attempt": (
            result.offline_network_receipt_sha256_by_attempt
        ),
        "runtime_environment_payload": runtime.environment_payload,
        "training_runtime_receipt_sha256": runtime.receipt_sha256,
        "code_closure": {
            "artifacts": tuple(
                {
                    "bytes": row.bytes,
                    "relative_path": row.relative_path,
                    "sha256": row.sha256,
                }
                for row in code_closure.artifacts
            ),
            "git_commit": code_closure.git_commit,
            "receipt_sha256": code_closure.receipt_sha256,
            "schema": code_closure.schema,
            "status": code_closure.status,
        },
        "code_closure_receipt_sha256": code_closure.receipt_sha256,
        "tokenizer_panel_physical_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "tokenizer_offline_network_receipt_sha256": (
            tokenizer_arms[0].offline_network_receipt_sha256
        ),
        "tokenizer_offline_network_policy_sha256": (
            tokenizer_arms[0].offline_network_policy_sha256
        ),
    }
    launch = {
        "payload": launch_payload,
        "receipt_sha256": gtok_v2_bound_sha256(
            "weft1_gtok_v2_production_campaign_launch", launch_payload
        ),
        "schema": "weft1_gtok_v2_production_campaign_launch",
    }
    _exclusive_json(arguments.output_root / "production-launch-receipt.json", launch)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
