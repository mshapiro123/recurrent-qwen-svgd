#!/usr/bin/env python3
"""Materialize G-TOK stream plans and seed-invariant metrics before A100 allocation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_campaign_v2 import (  # noqa: E402
    build_precalibration_cpu_evidence_v2,
    load_tokenizer_execution_panel_v2,
    write_precalibration_cpu_evidence_v2,
)
from training.weft1_gtok_code_closure_v2 import (  # noqa: E402
    capture_gtok_code_closure_v2,
    validate_gtok_code_closure_v2,
)
from training.weft1_gtok_offline_v2 import (  # noqa: E402
    assert_offline_campaign_child_v2,
    load_offline_parent_receipt_v2,
)
from training.weft1_gtok_pb_adapter_v2 import load_frozen_screen_corpus_v2  # noqa: E402
from training.weft1_gtok_training_v2 import load_v4_corpus_source_v2  # noqa: E402
from training.weft1_gtok_runtime_v2 import (  # noqa: E402
    cpu_runtime_identity_sha256_from_payload_v2,
    observed_training_cpu_runtime_payload_v2,
)


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
    parser.add_argument("--offline-network-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-panel-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-offline-network-receipt", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.output_evidence.exists() or arguments.output_evidence.is_symlink():
        raise FileExistsError("CPU precompute evidence path must be new")
    offline_network_receipt_sha256 = assert_offline_campaign_child_v2(
        arguments.offline_network_receipt
    )
    offline_parent_receipt, observed_offline_sha256 = load_offline_parent_receipt_v2(
        arguments.offline_network_receipt
    )
    if observed_offline_sha256 != offline_network_receipt_sha256:
        raise RuntimeError("precompute offline receipt changed after child attestation")
    cpu_runtime_payload = observed_training_cpu_runtime_payload_v2(
        requirements_lock=arguments.training_requirements_lock,
        runtime_build_receipt=arguments.runtime_build_receipt,
        pa_runtime_build_receipt=arguments.pa_runtime_build_receipt,
    )
    code_closure = capture_gtok_code_closure_v2(REPOSITORY_ROOT)
    validate_gtok_code_closure_v2(code_closure, repository_root=REPOSITORY_ROOT)
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
    with tempfile.TemporaryDirectory(prefix="weft1-gtok-cpu-source-") as directory:
        source = load_v4_corpus_source_v2(
            arguments.corpus_root,
            sqlite_path=Path(directory) / "physical-d6.sqlite",
        )
        evidence = build_precalibration_cpu_evidence_v2(
            corpus=frozen,
            source=source,
            tokenizer_arms=tokenizer_arms,
            code_closure_receipt_sha256=code_closure.receipt_sha256,
            cpu_runtime_identity_sha256=(
                cpu_runtime_identity_sha256_from_payload_v2(cpu_runtime_payload)
            ),
            offline_network_policy_sha256=offline_parent_receipt.policy_sha256,
            offline_network_receipt_sha256=offline_network_receipt_sha256,
            generator_script_sha256=hashlib.sha256(
                Path(__file__).resolve(strict=True).read_bytes()
            ).hexdigest(),
        )
    validate_gtok_code_closure_v2(code_closure, repository_root=REPOSITORY_ROOT)
    arguments.output_evidence.parent.mkdir(parents=True, exist_ok=True)
    write_precalibration_cpu_evidence_v2(arguments.output_evidence, evidence)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
