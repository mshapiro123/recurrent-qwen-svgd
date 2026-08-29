"""Fetch only the authoritative WEFT-1 P-A source-cache prefix and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_fetch_a2 import (  # noqa: E402
    DOWNLOAD_ARTIFACT_NAME,
    ENUMERATION_ARTIFACT_NAME,
    EXTERNAL_TRANSPORT_ARTIFACT_NAME,
    SELECTION_ARTIFACT_NAME,
    SOURCE_MANIFEST_NAME,
    SOURCE_PREP_ATTESTED_HEAD_MODE,
    SOURCE_PREP_CODE_IDENTITY_SCHEMA_V1,
    SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V1,
    SourcePrepCodeIdentityV1,
    SourcePrepImplementationFileV1,
    finalize_authoritative_external_transport_receipt_v1,
    prepare_pa_sources_online_v3,
)
from training.weft1_gtok_a1_contract import (  # noqa: E402
    SOURCE_ROUTE_MANIFEST_PATH,
)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("source preparation requires an exact lowercase Git commit")
    return commit


def _attest_clean_source_prep_code() -> SourcePrepCodeIdentityV1:
    commit = _git_head()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("source preparation requires a completely clean Git tree")
    rows: list[SourcePrepImplementationFileV1] = []
    for repo_path in SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V1:
        local_path = ROOT.joinpath(*repo_path.split("/"))
        local_bytes = local_path.read_bytes()
        committed_blob = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", f"{commit}:{repo_path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        local_blob = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "hash-object",
                f"--path={repo_path}",
                str(local_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if local_blob != committed_blob:
            raise RuntimeError(
                f"source-prep implementation differs from HEAD: {repo_path}"
            )
        rows.append(
            SourcePrepImplementationFileV1(
                repo_path=repo_path,
                bytes=len(local_bytes),
                sha256=hashlib.sha256(local_bytes).hexdigest(),
                git_blob_sha1=committed_blob,
            )
        )
    return SourcePrepCodeIdentityV1(
        schema=SOURCE_PREP_CODE_IDENTITY_SCHEMA_V1,
        mode=SOURCE_PREP_ATTESTED_HEAD_MODE,
        git_commit=commit,
        files=tuple(rows),
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
    source_prep_code_identity = _attest_clean_source_prep_code()
    implementation_by_path = {
        row.repo_path: row for row in source_prep_code_identity.files
    }
    result = prepare_pa_sources_online_v3(
        cache_root=arguments.cache_root,
        transport_cache_root=arguments.transport_cache_root,
        receipt_root=arguments.receipt_root,
        source_prep_code_identity=source_prep_code_identity,
        route_manifest_path=arguments.route_manifest,
    )
    post_execution_code_identity = _attest_clean_source_prep_code()
    external_transport_artifact_sha256 = (
        finalize_authoritative_external_transport_receipt_v1(
            arguments.receipt_root / EXTERNAL_TRANSPORT_ARTIFACT_NAME,
            result.external_transport,
            post_execution_code_identity=post_execution_code_identity,
        )
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
        "external_transport_artifact": str(
            (arguments.receipt_root / EXTERNAL_TRANSPORT_ARTIFACT_NAME).resolve()
        ),
        "external_transport_artifact_sha256": (
            external_transport_artifact_sha256
        ),
        "external_transport_policy_id": (
            result.external_transport.transport_policy_id
        ),
        "external_transport_receipt_sha256": (
            result.external_transport.receipt_sha256
        ),
        "git_commit": source_prep_code_identity.git_commit,
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
        "source_fetch_cli_sha256": implementation_by_path[
            "scripts/run_weft1_corpus_fetch_a2.py"
        ].sha256,
        "source_fetch_module_sha256": implementation_by_path[
            "training/weft1_corpus_fetch_a2.py"
        ].sha256,
        "source_prep_code_identity_sha256": (
            source_prep_code_identity.identity_sha256
        ),
        "status": "P_A_SOURCE_CACHE_PREPARED",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
