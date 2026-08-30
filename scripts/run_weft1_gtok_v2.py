#!/usr/bin/env python3
"""Fail-closed command surface for WEFT-1 G-TOK tokenizer production.

Every production command first revalidates the stored P-B freeze through the
A2 first-fit adapter.  The base-campaign orchestrator remains intentionally
unexposed; this surface cannot synthesize legacy A1 prefix-floor fields.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_contract import (  # noqa: E402
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
)
from training.weft1_gtok_pb_adapter_v2 import (  # noqa: E402
    load_frozen_screen_corpus_v2,
)
from training.weft1_gtok_offline_v2 import (  # noqa: E402
    assert_offline_campaign_child_v2,
    load_offline_parent_receipt_v2,
)
from training.weft1_gtok_tokenizer_v2 import (  # noqa: E402
    fit_tokenizer_arm_double_v2,
)


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fit_one(
    arguments: argparse.Namespace,
    vocab_size: int,
    output: Path,
    frozen,
) -> dict[str, object]:
    arm, evidence = fit_tokenizer_arm_double_v2(
        corpus_root=arguments.corpus_root,
        output_parent=output,
        vocab_size=vocab_size,
        dependency_lock_path=arguments.dependency_lock,
        offline_network_receipt_path=arguments.offline_network_receipt,
        offline_network_receipt_sha256=arguments.offline_network_receipt_sha256,
        offline_network_policy_sha256=arguments.offline_network_policy_sha256,
        repository_root=arguments.repository_root,
        worker_executable=arguments.worker_executable,
    )
    if (
        arm.full_corpus_manifest_sha256 != frozen.full_corpus_manifest_sha256
        or arm.fit_stream_sha256 != frozen.training_stream_sha256
    ):
        raise RuntimeError("tokenizer fit evidence differs from the revalidated P-B corpus")
    return {
        "arm": asdict(arm),
        "arm_receipt_sha256": arm.receipt_sha256,
        "corpus_receipt_sha256": frozen.receipt_sha256,
        "evidence": evidence,
        "offline_network_receipt_sha256": arguments.offline_network_receipt_sha256,
        "offline_network_policy_sha256": arguments.offline_network_policy_sha256,
        "output_root": str(output.resolve(strict=True)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--gate-bundle", type=Path, required=True)
    parser.add_argument("--c2-evidence", type=Path, required=True)
    parser.add_argument("--decon-receipt", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--worker-executable", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("fit-arm")
    one.add_argument("--vocab-size", type=int, choices=GTOK_VOCABULARY_ARMS, required=True)
    one.add_argument("--output-root", type=Path, required=True)
    one.add_argument("--offline-network-receipt", type=Path, required=True)
    all_arms = subparsers.add_parser("fit-all")
    all_arms.add_argument("--output-root", type=Path, required=True)
    all_arms.add_argument("--offline-network-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # This must precede P-B revalidation and every corpus read.  The generic
    # launcher appends this subcommand argument only after it has successfully
    # parent-probed the exact unshare binary and entered ``unshare --net``.
    arguments.offline_network_receipt_sha256 = assert_offline_campaign_child_v2(
        arguments.offline_network_receipt
    )
    offline_parent_receipt, observed_offline_sha256 = load_offline_parent_receipt_v2(
        arguments.offline_network_receipt
    )
    if observed_offline_sha256 != arguments.offline_network_receipt_sha256:
        raise RuntimeError("tokenizer offline receipt changed after child attestation")
    arguments.offline_network_policy_sha256 = offline_parent_receipt.policy_sha256
    frozen = load_frozen_screen_corpus_v2(
        materialization_root=arguments.corpus_root,
        freeze_receipt_path=arguments.freeze_receipt,
        gate_bundle_path=arguments.gate_bundle,
        c2_evidence_path=arguments.c2_evidence,
        decon_receipt_path=arguments.decon_receipt,
    )
    if arguments.command == "fit-arm":
        _fit_one(arguments, arguments.vocab_size, arguments.output_root, frozen)
        return 0
    parent = arguments.output_root
    parent.mkdir(parents=True, exist_ok=False)
    rows = tuple(
        _fit_one(arguments, vocab_size, parent / f"vocab-{vocab_size}", frozen)
        for vocab_size in GTOK_VOCABULARY_ARMS
    )
    _exclusive_json(
        parent / "tokenizer-panel-receipt.json",
        {
            "arms": rows,
            "offline_network_receipt_sha256": arguments.offline_network_receipt_sha256,
            "offline_network_policy_sha256": arguments.offline_network_policy_sha256,
            "schema": "weft1_gtok_v2_tokenizer_panel",
            "vocabularies": GTOK_VOCABULARY_ARMS,
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
