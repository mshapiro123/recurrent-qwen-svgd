"""Additive A3/V4 execution CLI; the frozen V3 ``full-pa`` remains unchanged."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_replay_a2 import (  # noqa: E402
    ParentReplayError,
    attest_production_storage_v3,
)
from training.weft1_corpus_replay_a3 import (  # noqa: E402
    V4_DEFAULT_WORKER_TIMEOUT_SECONDS,
    verify_production_materialization_replays_v4,
)
from training.weft1_gtok_contract import canonical_json_bytes  # noqa: E402
from training.weft1_strict_io import assert_no_symlink_ancestors  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser("full-pa-v4")
    full.add_argument("--enumeration-receipt", required=True, type=Path)
    full.add_argument("--cache-download-receipt", required=True, type=Path)
    full.add_argument("--source-cache-manifest", required=True, type=Path)
    full.add_argument("--source-cache", required=True, type=Path)
    full.add_argument("--fasttext-model", required=True, type=Path)
    full.add_argument("--runtime-build-receipt", required=True, type=Path)
    full.add_argument("--durable-mount-root", required=True, type=Path)
    full.add_argument("--durable-storage-marker", required=True, type=Path)
    full.add_argument("--durable-output-parent", required=True, type=Path)
    full.add_argument(
        "--durable-parsed-asset-cache-parent", required=True, type=Path
    )
    full.add_argument("--local-work-parent", required=True, type=Path)
    full.add_argument("--receipt-out", required=True, type=Path)
    full.add_argument(
        "--timeout-seconds",
        type=float,
        default=V4_DEFAULT_WORKER_TIMEOUT_SECONDS,
        help=(
            "finite parent-side worker watchdog; defaults to 14 days (2x the "
            "observed per-replay projection), while Colab backend loss is "
            "handled by parsed-asset recovery"
        ),
    )
    return parser


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    output_parent = assert_no_symlink_ancestors(
        arguments.durable_output_parent
    ).resolve(strict=False)
    parsed_asset_cache_parent = assert_no_symlink_ancestors(
        arguments.durable_parsed_asset_cache_parent
    ).resolve(strict=False)
    local_parent = assert_no_symlink_ancestors(
        arguments.local_work_parent
    ).resolve(strict=True)
    receipt_out = assert_no_symlink_ancestors(arguments.receipt_out).resolve(
        strict=False
    )
    if output_parent.exists():
        raise ParentReplayError("V4 durable output parent must be fresh")
    if parsed_asset_cache_parent.exists() and not parsed_asset_cache_parent.is_dir():
        raise ParentReplayError(
            "V4 parsed-asset cache parent must be absent or a directory"
        )
    if not parsed_asset_cache_parent.parent.resolve(strict=True).is_dir():
        raise ParentReplayError(
            "V4 parsed-asset cache parent requires an existing parent"
        )
    if not local_parent.is_dir():
        raise ParentReplayError("V4 local work parent must be a real directory")
    if receipt_out.exists() or receipt_out.parent != output_parent:
        raise ParentReplayError("V4 receipt must be fresh inside the output parent")
    governed_roots = (
        output_parent,
        parsed_asset_cache_parent,
        local_parent,
        assert_no_symlink_ancestors(arguments.source_cache).resolve(strict=True),
    )
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(governed_roots)
        for right in governed_roots[index + 1 :]
    ):
        raise ParentReplayError(
            "V4 output, parsed-asset cache, local work, and source cache must be disjoint"
        )
    output_storage_identity = attest_production_storage_v3(
        durable_mount_root=arguments.durable_mount_root,
        durable_storage_marker_path=arguments.durable_storage_marker,
        durable_output_parent=output_parent.parent,
        local_work_parent=local_parent,
    )
    parsed_cache_attestation_root = (
        parsed_asset_cache_parent
        if parsed_asset_cache_parent.exists()
        else parsed_asset_cache_parent.parent
    )
    if attest_production_storage_v3(
        durable_mount_root=arguments.durable_mount_root,
        durable_storage_marker_path=arguments.durable_storage_marker,
        durable_output_parent=parsed_cache_attestation_root,
        local_work_parent=local_parent,
    ) != output_storage_identity:
        raise ParentReplayError(
            "V4 parsed-asset cache is not on the registered durable storage"
        )
    parsed_asset_cache_parent.mkdir(exist_ok=True)
    output_parent.mkdir(parents=True)
    result = verify_production_materialization_replays_v4(
        python_executable=Path(sys.executable),
        enumeration_receipt_path=arguments.enumeration_receipt,
        cache_download_receipt_path=arguments.cache_download_receipt,
        source_manifest_path=arguments.source_cache_manifest,
        cache_root=arguments.source_cache,
        fasttext_model_path=arguments.fasttext_model,
        runtime_build_receipt_path=arguments.runtime_build_receipt,
        durable_mount_root=arguments.durable_mount_root,
        durable_storage_marker_path=arguments.durable_storage_marker,
        durable_output_parent=output_parent,
        durable_parsed_asset_cache_parent=parsed_asset_cache_parent,
        local_work_parent=local_parent,
        first_output_root=output_parent / "production-v4-replay-a",
        second_output_root=output_parent / "production-v4-replay-b",
        timeout_seconds=arguments.timeout_seconds,
    )
    payload = {**asdict(result), "receipt_sha256": result.receipt_sha256}
    raw = canonical_json_bytes(payload) + b"\n"
    partial = receipt_out.with_name(receipt_out.name + ".partial")
    with partial.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, receipt_out)
    return payload


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command != "full-pa-v4":
        raise AssertionError("unknown V4 command")
    _run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
