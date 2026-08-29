#!/usr/bin/env python3
"""Mint or replay WEFT-1 Amendment-A3 upstream path evidence.

This command performs metadata-only observation.  It never downloads corpus
assets and never authorizes source materialization.  ``mint`` writes the
canonical semantic-evidence ledger and combined path-breakdown receipt once;
``replay`` re-fetches the same pinned evidence and repository trees and
requires both checked-in artifacts to reproduce exactly.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence
import urllib.request

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_a3 import (
    A3_AUTHORITY_SHA256,
    REPOSITORY_ROOT,
    verify_a3_authority_artifact,
)
from training.weft1_corpus_breakdown_a3 import (
    DOLMA_SOURCE_FAMILY,
    DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
    FIXTURE_OBSERVATION_MODE_A3,
    FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
    FINEWEB_SOURCE_FAMILY,
    PathBreakdownError,
    PinnedSemanticEvidenceA3,
    PriorRouteDeclarationA3,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
    build_dolma_path_breakdown_a3,
    build_fineweb_path_breakdown_a3,
    build_upstream_path_breakdown_a3,
    load_upstream_path_breakdown_snapshot_a3,
    observe_hf_tree_files_a3,
    replay_upstream_path_breakdown_a3,
    write_upstream_path_breakdown_a3,
)
from training.weft1_gtok_a1_contract import load_source_route_manifest
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_corpus_semantic_evidence_a3 import (
    SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1,
    SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1,
    SEMANTIC_EVIDENCE_RELATIVE_PATH_A3,
    SEMANTIC_EVIDENCE_SCHEMA_A3_V1,
    load_semantic_evidence_snapshot_a3,
    validate_semantic_evidence_payload_a3,
)
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
)


HF_HUB_VERSION = "1.24.0"
EVIDENCE_SCHEMA = SEMANTIC_EVIDENCE_SCHEMA_A3_V1
EVIDENCE_RECEIPT_SCHEMA = SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1
EVIDENCE_FAMILY_SCHEMA = SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1
EVIDENCE_RELATIVE_PATH = Path(SEMANTIC_EVIDENCE_RELATIVE_PATH_A3)
BREAKDOWN_RELATIVE_PATH = Path(
    "training/weft1_corpus_path_breakdown_a3_20260829.json"
)

DOLMA_POOL_REPOSITORY = "allenai/dolma3_pool"
DOLMA_POOL_REVISION = "6462556697df1a8f5c953727e9c686629ad98b68"
FINEWEB_REPOSITORY = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DOLMA_CONSTRUCTION_REVISION = "1a9daced81670e0fa768e47fbed32af6694a1865"

_FINEWEB_DUMP = re.compile(r"data/(CC-MAIN-[0-9]{4}-[0-9]{2})/")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "weft1-a3-observer/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


_EVIDENCE_DOCUMENTS: tuple[Mapping[str, object], ...] = (
    {
        "source_family": DOLMA_SOURCE_FAMILY,
        "repository": DOLMA_POOL_REPOSITORY,
        "revision": DOLMA_POOL_REVISION,
        "path": "README.md",
        "url": (
            "https://huggingface.co/datasets/allenai/dolma3_pool/resolve/"
            f"{DOLMA_POOL_REVISION}/README.md"
        ),
        "bytes": 4_909,
        "content_sha256": (
            "7a43fb286ca2d57f6e44fbc109b7208a778d59a7f6e3c8a52ccd4f172d4e0ab1"
        ),
        "supports": "pool path grammar identifies Common Crawl topic-vigintile groups",
    },
    {
        "source_family": DOLMA_SOURCE_FAMILY,
        "repository": "allenai/dolma3",
        "revision": DOLMA_CONSTRUCTION_REVISION,
        "path": "datasets/dolma3_mix/pools/9T/README.md",
        "url": (
            "https://raw.githubusercontent.com/allenai/dolma3/"
            f"{DOLMA_CONSTRUCTION_REVISION}/datasets/dolma3_mix/pools/9T/README.md"
        ),
        "bytes": 25_097,
        "content_sha256": (
            "30e6ffa107eb00c5eecaabb276bc49b9b3cea9025a5293f6a09ce45c366eb18d"
        ),
        "supports": "quality buckets are approximately five-percent vigintiles",
    },
    {
        "source_family": DOLMA_SOURCE_FAMILY,
        "repository": "allenai/dolma3",
        "revision": DOLMA_CONSTRUCTION_REVISION,
        "path": "datasets/dolma3_mix/mixes/6T-1025/README.md",
        "url": (
            "https://raw.githubusercontent.com/allenai/dolma3/"
            f"{DOLMA_CONSTRUCTION_REVISION}/datasets/dolma3_mix/mixes/6T-1025/README.md"
        ),
        "bytes": 7_212,
        "content_sha256": (
            "4ac79cc424249cb384cd5e6476b2a65b0caf59f83a72125dad1793d596c99396"
        ),
        "supports": "higher vigintiles receive monotonically higher quality weight",
    },
    {
        "source_family": DOLMA_SOURCE_FAMILY,
        "repository": "allenai/dolma3",
        "revision": DOLMA_CONSTRUCTION_REVISION,
        "path": "datasets/dolma3_mix/mixes/simple_mixing.py",
        "url": (
            "https://raw.githubusercontent.com/allenai/dolma3/"
            f"{DOLMA_CONSTRUCTION_REVISION}/datasets/dolma3_mix/mixes/simple_mixing.py"
        ),
        "bytes": 24_423,
        "content_sha256": (
            "1320d26ae39b2b62a292d131a6c57639430cbe0a466fe668f36239a76e01a8b3"
        ),
        "supports": "sorted bucket labels map to monotonically increasing percentile intervals",
    },
    {
        "source_family": FINEWEB_SOURCE_FAMILY,
        "repository": FINEWEB_REPOSITORY,
        "revision": FINEWEB_REVISION,
        "path": "README.md",
        "url": (
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
            f"{FINEWEB_REVISION}/README.md"
        ),
        "bytes": 26_354,
        "content_sha256": (
            "a0cc8998a20499432b28b6575f3046b714938eb8e11b8d59a1d25ddf3716061e"
        ),
        "supports": "default main data enumerates the CC-MAIN configurations; samples are separate",
    },
)


def build_semantic_evidence_payload(
    fetch: Callable[[str], bytes] = _fetch,
) -> Mapping[str, object]:
    """Fetch exact pinned documents and derive the two family claims."""

    verified: list[Mapping[str, object]] = []
    fineweb_card: bytes | None = None
    for expected in _EVIDENCE_DOCUMENTS:
        raw = fetch(str(expected["url"]))
        if len(raw) != expected["bytes"] or _sha256(raw) != expected["content_sha256"]:
            raise PathBreakdownError(
                f"semantic evidence drifted: {expected['repository']}:{expected['path']}"
            )
        verified.append(dict(expected))
        if expected["source_family"] == FINEWEB_SOURCE_FAMILY:
            fineweb_card = raw
    if fineweb_card is None:
        raise PathBreakdownError("FineWeb semantic card was not fetched")
    text = fineweb_card.decode("utf-8")
    dump_ids = tuple(
        sorted(
            set(_FINEWEB_DUMP.findall(text)),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(dump_ids) != 110:
        raise PathBreakdownError(
            f"FineWeb pinned card exposes {len(dump_ids)} rather than 110 main dumps"
        )
    families: list[Mapping[str, object]] = []
    for source_family, assertion, pin, facts in (
        (
            DOLMA_SOURCE_FAMILY,
            DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
            DOLMA_CONSTRUCTION_REVISION,
            (
                "the pinned pool exposes numeric vigintile path groups",
                "construction sources order those groups from lower to higher quality",
                "0019 is the highest released selectable Common Crawl vigintile",
            ),
        ),
        (
            FINEWEB_SOURCE_FAMILY,
            FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
            FINEWEB_REVISION,
            (
                "the pinned card enumerates exactly 110 CC-MAIN main-data dumps",
                "sample configurations are outside the main-data family",
                "all Parquet shards within those 110 dump directories are in-family",
            ),
        ),
    ):
        documents = [
            dict(row)
            for row in verified
            if row["source_family"] == source_family
        ]
        family = {
            "assertion": assertion,
            "configured_group_ids": (
                list(dump_ids) if source_family == FINEWEB_SOURCE_FAMILY else []
            ),
            "derived_facts": list(facts),
            "pin": pin,
            "source_family": source_family,
            "upstream_documents": documents,
        }
        families.append(
            {
                **family,
                "family_evidence_sha256": canonical_sha256(
                    {"payload": family, "schema": EVIDENCE_FAMILY_SCHEMA}
                ),
            }
        )
    core = {
        "authority_sha256": A3_AUTHORITY_SHA256,
        "families": families,
        "schema": EVIDENCE_SCHEMA,
    }
    return {
        **core,
        "receipt_sha256": canonical_sha256(
            {"payload": core, "schema": EVIDENCE_RECEIPT_SCHEMA}
        ),
    }


def _validate_semantic_evidence_payload(payload: Mapping[str, object]) -> None:
    validate_semantic_evidence_payload_a3(payload)  # type: ignore[arg-type]


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    assert_no_symlink_ancestors(path)
    if path.exists():
        raise PathBreakdownError(f"refusing to overwrite governed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload) + b"\n"
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise PathBreakdownError(f"stale governed partial exists: {partial}")
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _load_semantic_evidence(path: Path) -> tuple[bytes, Mapping[str, object]]:
    raw, payload, unused_identity = load_semantic_evidence_snapshot_a3(path)
    del unused_identity
    return raw, payload


def _family_evidence(
    payload: Mapping[str, object], source_family: str
) -> Mapping[str, object]:
    families = payload["families"]
    assert isinstance(families, list)
    return next(
        row
        for row in families
        if isinstance(row, Mapping) and row["source_family"] == source_family
    )


def _observe_repository(api: HfApi, repository: str, revision: str):
    raw_tree = api.list_repo_tree(
        repository,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=False,
    )
    return observe_hf_tree_files_a3(raw_tree)


def _build_breakdown(
    evidence: Mapping[str, object],
    *,
    api: HfApi,
    observation_mode: str,
    observation_client_identity: object,
):
    route_manifest = load_source_route_manifest()
    routes = {route.source_family: route for route in route_manifest.routes}
    dolma_route = routes[DOLMA_SOURCE_FAMILY]
    fineweb_route = routes[FINEWEB_SOURCE_FAMILY]
    if (
        dolma_route.repository != DOLMA_POOL_REPOSITORY
        or dolma_route.revision != DOLMA_POOL_REVISION
        or fineweb_route.repository != FINEWEB_REPOSITORY
        or fineweb_route.revision != FINEWEB_REVISION
    ):
        raise PathBreakdownError("A1 route pins drifted from the A3 observer")
    dolma_evidence = _family_evidence(evidence, DOLMA_SOURCE_FAMILY)
    fineweb_evidence = _family_evidence(evidence, FINEWEB_SOURCE_FAMILY)
    dolma_members = _observe_repository(api, dolma_route.repository, dolma_route.revision)
    fineweb_members = _observe_repository(api, fineweb_route.repository, fineweb_route.revision)
    dolma = build_dolma_path_breakdown_a3(
        dolma_members,
        repository=dolma_route.repository,
        revision=dolma_route.revision,
        prior_declaration=PriorRouteDeclarationA3(
            source_family=dolma_route.source_family,
            asset_selector=dolma_route.asset_selector,
            asset_count=dolma_route.asset_count,
            available_bytes=dolma_route.available_bytes,
            declaration_receipt_sha256=dolma_route.receipt_sha256,
        ),
        semantic_evidence=PinnedSemanticEvidenceA3(
            evidence_id="dolma3-top-quality-bucket-ordering",
            locator=EVIDENCE_RELATIVE_PATH.as_posix() + "#dolma_web",
            pin=str(dolma_evidence["pin"]),
            content_sha256=str(dolma_evidence["family_evidence_sha256"]),
            assertion=str(dolma_evidence["assertion"]),
        ),
    )
    configured_ids = fineweb_evidence["configured_group_ids"]
    if not isinstance(configured_ids, list):
        raise PathBreakdownError("FineWeb configured dump ids are not a list")
    fineweb = build_fineweb_path_breakdown_a3(
        fineweb_members,
        repository=fineweb_route.repository,
        revision=fineweb_route.revision,
        prior_declaration=PriorRouteDeclarationA3(
            source_family=fineweb_route.source_family,
            asset_selector=fineweb_route.asset_selector,
            asset_count=fineweb_route.asset_count,
            available_bytes=fineweb_route.available_bytes,
            declaration_receipt_sha256=fineweb_route.receipt_sha256,
        ),
        semantic_evidence=PinnedSemanticEvidenceA3(
            evidence_id="fineweb-edu-main-data-all-cc-dumps",
            locator=EVIDENCE_RELATIVE_PATH.as_posix() + "#fineweb_edu",
            pin=str(fineweb_evidence["pin"]),
            content_sha256=str(fineweb_evidence["family_evidence_sha256"]),
            assertion=str(fineweb_evidence["assertion"]),
        ),
        configured_main_dump_ids=tuple(configured_ids),
    )
    return (
        build_upstream_path_breakdown_a3(
            authority_sha256=A3_AUTHORITY_SHA256,
            observation_mode=observation_mode,
            observation_client_identity=observation_client_identity,
            dolma=dolma,
            fineweb=fineweb,
        ),
        dolma_members,
        fineweb_members,
    )


def _run(
    mode: str,
    *,
    root: Path,
    fetch: Callable[[str], bytes],
    api: object,
    observation_mode: str,
    observation_client_identity: object,
) -> Mapping[str, object]:
    if mode not in {"mint", "replay"}:
        raise ValueError("A3 observer mode must be mint or replay")
    verify_a3_authority_artifact()
    evidence_path = root / EVIDENCE_RELATIVE_PATH
    breakdown_path = root / BREAKDOWN_RELATIVE_PATH
    if mode == "mint" and (evidence_path.exists() or breakdown_path.exists()):
        raise PathBreakdownError(
            "mint requires both governed A3 observer outputs to be absent"
        )
    live_evidence = build_semantic_evidence_payload(fetch)
    if mode == "mint":
        _write_once(evidence_path, live_evidence)
        evidence_raw = evidence_path.read_bytes()
        evidence = live_evidence
    else:
        evidence_raw, evidence = _load_semantic_evidence(evidence_path)
        if evidence != live_evidence:
            raise PathBreakdownError("live semantic evidence replay differs")
    receipt, dolma_members, fineweb_members = _build_breakdown(
        evidence,
        api=api,  # type: ignore[arg-type]
        observation_mode=observation_mode,
        observation_client_identity=observation_client_identity,
    )
    if mode == "mint":
        breakdown_physical_sha256 = write_upstream_path_breakdown_a3(
            receipt, breakdown_path
        )
        breakdown_raw = breakdown_path.read_bytes()
    else:
        breakdown_raw, expected = load_upstream_path_breakdown_snapshot_a3(
            breakdown_path
        )
        replay_upstream_path_breakdown_a3(
            expected,
            dolma_members=dolma_members,
            fineweb_members=fineweb_members,
        )
        if receipt != expected:
            raise PathBreakdownError("live combined breakdown replay differs")
        breakdown_physical_sha256 = _sha256(breakdown_raw)
    return {
        "authorizes_downloads": False,
        "breakdown_bytes": len(breakdown_raw),
        "breakdown_path": BREAKDOWN_RELATIVE_PATH.as_posix(),
        "breakdown_physical_sha256": breakdown_physical_sha256,
        "breakdown_receipt_sha256": receipt.receipt_sha256,
        "dolma_selected_asset_count": receipt.families[0].selected_asset_count,
        "dolma_selected_upstream_bytes": receipt.families[0].selected_upstream_bytes,
        "evidence_bytes": len(evidence_raw),
        "evidence_path": EVIDENCE_RELATIVE_PATH.as_posix(),
        "evidence_physical_sha256": _sha256(evidence_raw),
        "evidence_receipt_sha256": str(evidence["receipt_sha256"]),
        "fineweb_selected_asset_count": receipt.families[1].selected_asset_count,
        "fineweb_selected_upstream_bytes": receipt.families[1].selected_upstream_bytes,
        "observation_client_identity_sha256": (
            receipt.observation_client_identity_sha256
        ),
        "observation_mode": receipt.observation_mode,
        "mode": mode,
        "status": "A3_PATH_BREAKDOWN_REPLAYED" if mode == "replay" else "A3_PATH_BREAKDOWN_MINTED",
    }


def _production_hf_api() -> HfApi:
    if version("huggingface_hub") != HF_HUB_VERSION:
        raise PathBreakdownError(
            f"huggingface_hub must be exactly {HF_HUB_VERSION}"
        )
    if HfApi.__module__ != "huggingface_hub.hf_api" or HfApi.__qualname__ != "HfApi":
        raise PathBreakdownError("production observation client class drifted")
    api = HfApi()
    if type(api) is not HfApi:
        raise PathBreakdownError("production observation client instance drifted")
    return api


def run(
    mode: str,
    *,
    root: Path,
    fetch: Callable[[str], bytes] = _fetch,
) -> Mapping[str, object]:
    """Run the only production observation path; client injection is forbidden."""

    return _run(
        mode,
        root=root,
        fetch=fetch,
        api=_production_hf_api(),
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity=PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    )


def run_nonproduction_fixture(
    mode: str,
    *,
    root: Path,
    fetch: Callable[[str], bytes],
    api: object,
) -> Mapping[str, object]:
    """Exercise deterministic mechanics with an explicitly unusable receipt."""

    return _run(
        mode,
        root=root,
        fetch=fetch,
        api=api,
        observation_mode=FIXTURE_OBSERVATION_MODE_A3,
        observation_client_identity=FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mint", "replay"))
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    payload = run(args.mode, root=args.root.resolve())
    sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
