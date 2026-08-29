"""Fetch only the authoritative WEFT-1 P-A source-cache prefix and receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_fetch_a2 import (  # noqa: E402
    DOWNLOAD_ARTIFACT_NAME,
    ENUMERATION_ARTIFACT_NAME,
    SELECTION_ARTIFACT_NAME,
    SOURCE_MANIFEST_NAME,
    prepare_pa_sources_online_v3,
)
from training.weft1_gtok_a1_contract import (  # noqa: E402
    SOURCE_ROUTE_MANIFEST_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        required=True,
        type=Path,
        help="Dedicated verified source-asset cache root.",
    )
    parser.add_argument(
        "--transport-cache-root",
        required=True,
        type=Path,
        help="Dedicated Hugging Face and pinned external transport cache root.",
    )
    parser.add_argument(
        "--receipt-root",
        required=True,
        type=Path,
        help="Dedicated canonical enumeration, selection, manifest, and receipt root.",
    )
    parser.add_argument(
        "--route-manifest",
        type=Path,
        default=SOURCE_ROUTE_MANIFEST_PATH,
        help="Exact checked-in A1 source-route ledger.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    result = prepare_pa_sources_online_v3(
        cache_root=arguments.cache_root,
        transport_cache_root=arguments.transport_cache_root,
        receipt_root=arguments.receipt_root,
        route_manifest_path=arguments.route_manifest,
    )
    summary = {
        "download_artifact": str(
            (arguments.receipt_root / DOWNLOAD_ARTIFACT_NAME).resolve()
        ),
        "download_artifact_sha256": result.download_artifact_sha256,
        "download_receipt_sha256": result.download.receipt_sha256,
        "enumeration_artifact": str(
            (arguments.receipt_root / ENUMERATION_ARTIFACT_NAME).resolve()
        ),
        "enumeration_artifact_sha256": result.enumeration_artifact_sha256,
        "enumeration_receipt_sha256": result.enumeration.receipt_sha256,
        "selected_asset_count": len(result.plan.assets),
        "selected_upstream_bytes": sum(
            asset.upstream_bytes for asset in result.plan.assets
        ),
        "selection_artifact": str(
            (arguments.receipt_root / SELECTION_ARTIFACT_NAME).resolve()
        ),
        "selection_artifact_sha256": result.selection_artifact_sha256,
        "selection_receipt_sha256": result.selection.receipt_sha256,
        "source_manifest": str(
            (arguments.receipt_root / SOURCE_MANIFEST_NAME).resolve()
        ),
        "source_manifest_sha256": result.download.source_manifest.manifest_sha256,
        "status": "P_A_SOURCE_CACHE_PREPARED",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
