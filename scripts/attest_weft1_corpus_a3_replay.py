#!/usr/bin/env python3
"""Mint the external clean-HEAD A3 live-replay attestation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_weft1_corpus_a3_observer import run as replay_observer  # noqa: E402
from scripts.run_weft1_corpus_fetch_a3 import _attest_clean_code  # noqa: E402
from training.weft1_corpus_a3 import A3_AUTHORITY_SHA256  # noqa: E402
from training.weft1_corpus_breakdown_a3 import (  # noqa: E402
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
)
from training.weft1_corpus_fetch_a3 import (  # noqa: E402
    A3_REPLAY_ATTESTATION_SCHEMA_V4,
    A3ReplayAttestationV4,
    load_pa_source_execution_context_v4,
    write_a3_replay_attestation_v4,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--breakdown-root",
        required=True,
        type=Path,
        help="Clean repository root containing the resolved A3 artifacts.",
    )
    parser.add_argument(
        "--replay-attestation",
        required=True,
        type=Path,
        help="Output path outside the repository.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    breakdown_root = arguments.breakdown_root.resolve(strict=True)
    if breakdown_root != ROOT.resolve(strict=True):
        raise RuntimeError(
            "A3 replay must attest the repository that contains this script"
        )
    context = load_pa_source_execution_context_v4(
        breakdown_root=breakdown_root
    )
    code_identity = _attest_clean_code(context)
    replay = replay_observer("replay", root=breakdown_root)
    expected_replay = (
        "A3_PATH_BREAKDOWN_REPLAYED",
        context.binding.breakdown_artifact_physical_sha256,
        context.binding.breakdown_artifact_receipt_sha256,
        context.semantic_evidence_artifact_physical_sha256,
        context.semantic_evidence_artifact_receipt_sha256,
        PRODUCTION_OBSERVATION_MODE_A3,
        PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3,
    )
    observed_replay = (
        replay.get("status"),
        replay.get("breakdown_physical_sha256"),
        replay.get("breakdown_receipt_sha256"),
        replay.get("evidence_physical_sha256"),
        replay.get("evidence_receipt_sha256"),
        replay.get("observation_mode"),
        replay.get("observation_client_identity_sha256"),
    )
    if observed_replay != expected_replay:
        raise RuntimeError(
            "live A3 replay differs from the resolved execution context"
        )
    attestation = A3ReplayAttestationV4(
        schema=A3_REPLAY_ATTESTATION_SCHEMA_V4,
        status="ATTESTED_CLEAN_HEAD_LIVE_REPLAY_PASS",
        authorizes_downloads=True,
        git_commit=code_identity.git_commit,
        git_status="CLEAN",
        authority_sha256=A3_AUTHORITY_SHA256,
        execution_binding_sha256=context.binding_sha256,
        source_prep_code_identity_sha256=code_identity.receipt_sha256,
        semantic_evidence_artifact_physical_sha256=str(
            context.semantic_evidence_artifact_physical_sha256
        ),
        semantic_evidence_artifact_receipt_sha256=str(
            context.semantic_evidence_artifact_receipt_sha256
        ),
        semantic_evidence_family_receipt_sha256s=(
            context.semantic_evidence_family_receipt_sha256s
        ),
        breakdown_artifact_physical_sha256=(
            context.binding.breakdown_artifact_physical_sha256
        ),
        breakdown_artifact_receipt_sha256=(
            context.binding.breakdown_artifact_receipt_sha256
        ),
        overlay_artifact_physical_sha256=str(
            context.overlay_physical_sha256
        ),
        overlay_identity_sha256=str(context.overlay_identity_sha256),
        effective_route_identity_sha256=(
            context.binding.effective_route_identity_sha256
        ),
        huggingface_hub_distribution="huggingface-hub",
        huggingface_hub_version="1.24.0",
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity_sha256=(
            PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3
        ),
        live_replay_status="PASS_EXACT_BREAKDOWN_REPLAY",
        live_replay_receipt_sha256=(
            context.path_breakdown.receipt_sha256  # type: ignore[union-attr]
        ),
    )
    if _attest_clean_code(context) != code_identity:
        raise RuntimeError("source-prep code changed during A3 live replay")
    physical_sha256 = write_a3_replay_attestation_v4(
        arguments.replay_attestation,
        attestation,
    )
    print(
        json.dumps(
            {
                "git_commit": code_identity.git_commit,
                "physical_sha256": physical_sha256,
                "receipt_sha256": attestation.receipt_sha256,
                "status": attestation.status,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
