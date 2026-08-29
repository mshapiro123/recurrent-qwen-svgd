"""Fetch the authoritative A3/V4 WEFT-1 P-A source-cache prefix."""

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

from training.weft1_corpus_fetch_a3 import (  # noqa: E402
    AUTHORITATIVE_MODE,
    DOWNLOAD_ARTIFACT_NAME_V4,
    ENUMERATION_ARTIFACT_NAME_V4,
    EXTERNAL_TRANSPORT_ARTIFACT_NAME_V4,
    SELECTION_ARTIFACT_NAME_V4,
    SOURCE_MANIFEST_NAME_V4,
    SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4,
    SourcePrepCodeIdentityV4,
    SourcePrepImplementationFileV4,
    load_a3_replay_attestation_v4,
    load_pa_source_execution_context_v4,
    prepare_pa_sources_online_v4,
    write_external_transport_receipt_v4,
)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("A3 source prep requires an exact Git commit")
    return commit


def _attest_clean_code(context: object) -> SourcePrepCodeIdentityV4:
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
        raise RuntimeError("A3 source prep requires a completely clean Git tree")
    binding = getattr(context, "binding", None)
    rows: list[SourcePrepImplementationFileV4] = []
    for repo_path in SOURCE_PREP_IMPLEMENTATION_REPO_PATHS_V4:
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
            raise RuntimeError(f"source-prep file differs from HEAD: {repo_path}")
        rows.append(
            SourcePrepImplementationFileV4(
                repo_path=repo_path,
                bytes=len(local_bytes),
                sha256=hashlib.sha256(local_bytes).hexdigest(),
                git_blob_sha1=committed_blob,
            )
        )
    return SourcePrepCodeIdentityV4(
        mode=AUTHORITATIVE_MODE,
        execution_binding=binding,
        git_commit=commit,
        files=tuple(rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--breakdown-root",
        required=True,
        type=Path,
        help="Governed root containing the A3 combined breakdown artifact.",
    )
    parser.add_argument(
        "--replay-attestation",
        required=True,
        type=Path,
        help="External clean-commit A3 live-replay attestation artifact.",
    )
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--transport-cache-root", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    attestation_path = arguments.replay_attestation.resolve(strict=True)
    repository_root = ROOT.resolve(strict=True)
    if attestation_path == repository_root or repository_root in attestation_path.parents:
        raise RuntimeError("A3 replay attestation must be stored outside the repository")
    context = load_pa_source_execution_context_v4(
        breakdown_root=arguments.breakdown_root
    )
    code_identity = _attest_clean_code(context)
    replay_attestation = load_a3_replay_attestation_v4(
        attestation_path,
        context=context,
        source_prep_code_identity=code_identity,
    )
    result = prepare_pa_sources_online_v4(
        context=context,
        cache_root=arguments.cache_root,
        transport_cache_root=arguments.transport_cache_root,
        receipt_root=arguments.receipt_root,
        source_prep_code_identity=code_identity,
        replay_attestation=replay_attestation,
    )
    post_identity = _attest_clean_code(context)
    if post_identity != code_identity:
        raise RuntimeError("A3 source-prep code identity changed during execution")
    external_sha = write_external_transport_receipt_v4(
        arguments.receipt_root / EXTERNAL_TRANSPORT_ARTIFACT_NAME_V4,
        result.external_transport,
    )
    preparation = result.preparation
    summary = {
        "a3_replay_attestation_receipt_sha256": replay_attestation.receipt_sha256,
        "download_artifact": str(
            (arguments.receipt_root / DOWNLOAD_ARTIFACT_NAME_V4).resolve()
        ),
        "download_artifact_sha256": preparation.download_artifact_sha256,
        "download_receipt_sha256": preparation.download.receipt_sha256,
        "effective_route_identity_sha256": (
            context.binding.effective_route_identity_sha256
        ),
        "enumeration_artifact": str(
            (arguments.receipt_root / ENUMERATION_ARTIFACT_NAME_V4).resolve()
        ),
        "enumeration_artifact_sha256": preparation.enumeration_artifact_sha256,
        "enumeration_receipt_sha256": preparation.enumeration.receipt_sha256,
        "execution_binding_sha256": context.binding_sha256,
        "external_transport_artifact": str(
            (arguments.receipt_root / EXTERNAL_TRANSPORT_ARTIFACT_NAME_V4).resolve()
        ),
        "external_transport_artifact_sha256": external_sha,
        "external_transport_receipt_sha256": result.external_transport.receipt_sha256,
        "git_commit": code_identity.git_commit,
        "selected_asset_count": len(preparation.plan.assets),
        "selected_upstream_bytes": sum(
            asset.upstream_bytes for asset in preparation.plan.assets
        ),
        "selection_artifact": str(
            (arguments.receipt_root / SELECTION_ARTIFACT_NAME_V4).resolve()
        ),
        "selection_artifact_sha256": preparation.selection_artifact_sha256,
        "selection_receipt_sha256": preparation.selection.receipt_sha256,
        "source_manifest": str(
            (arguments.receipt_root / SOURCE_MANIFEST_NAME_V4).resolve()
        ),
        "source_manifest_receipt_sha256": (
            preparation.download.source_manifest.receipt_sha256
        ),
        "source_prep_code_identity_sha256": code_identity.receipt_sha256,
        "status": "P_A_A3_V4_SOURCE_CACHE_PREPARED",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
