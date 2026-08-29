from __future__ import annotations

import hashlib
import io
from pathlib import Path, PurePosixPath
from typing import Iterable

import pytest

import training.weft1_corpus_fetch_a2 as fetch_module
from training.weft1_corpus_enumeration_a2 import (
    FIXTURE_MODE,
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
    enumerate_upstream_assets_v3,
    load_upstream_enumeration_receipt_v3,
)
from training.weft1_corpus_fetch_a2 import (
    DOWNLOAD_ARTIFACT_NAME,
    ENUMERATION_ARTIFACT_NAME,
    SELECTION_ARTIFACT_NAME,
    SOURCE_MANIFEST_NAME,
    ExternalResourceCacheV3,
    SourceFetchError,
    prepare_selected_source_cache_v3,
    select_required_asset_prefixes_v3,
)
from training.weft1_corpus_source_io_a2 import (
    SourceTransportError,
    load_source_cache_download_receipt_v3,
)
from training.weft1_corpus_sources_a2 import load_exact_source_routes_v3
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    load_source_route_manifest,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _locator(family: str, index: int) -> str:
    return {
        "dolma_web": f"data/common_crawl-fetch-0019/{index:05d}.jsonl.zst",
        "wikipedia_wikibooks": (
            f"https://olmo-data.org/dolma-v1_7/wiki/wiki-{index:04d}.json.gz"
        ),
        "stackedu": f"data/stack_edu-fetch/{index:05d}.jsonl.zst",
        "finemath_3plus": f"finemath-3plus/train-{index:05d}.parquet",
        "arxiv": f"data/rpj-proofpile-arxiv/{index:05d}.jsonl.zst",
        "olmocr": f"data/olmocr_science_pdfs-fetch/{index:05d}.jsonl.zst",
        "fineweb_edu": f"data/CC-MAIN-fetch/train-{index:05d}.parquet",
    }[family]


def _fixture_enumeration():
    routes = load_exact_source_routes_v3()
    declared = {
        row.source_family: row for row in load_source_route_manifest().routes
    }
    trees: dict[tuple[str, str], list[dict[str, object]]] = {}
    payloads_by_locator: dict[str, bytes] = {}
    for route in routes:
        count = 2 if route.source_family == "wikipedia_wikibooks" else 3
        for index in range(count):
            locator = _locator(route.source_family, index)
            payload = (
                f"fetch-fixture:{route.source_family}:{index}:" + "x" * (index + 1)
            ).encode("ascii")
            payloads_by_locator[locator] = payload
            if route.source_family != "wikipedia_wikibooks":
                trees.setdefault((route.repository, route.revision), []).append(
                    {
                        "blob_id": hashlib.sha1(locator.encode("utf-8")).hexdigest(),
                        "lfs": {"sha256": _sha(payload), "size": len(payload)},
                        "path": locator,
                        "size": len(payload),
                        "type": "file",
                    }
                )

    def tree(**kwargs: object) -> Iterable[object]:
        return tuple(trees[(str(kwargs["repo_id"]), str(kwargs["revision"]))])

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        family = "wikipedia_wikibooks"
        assets = tuple(
            ExternalLocatorAssetV3(
                locator=_locator(family, index),
                upstream_bytes=len(payloads_by_locator[_locator(family, index)]),
                content_sha256=_sha(payloads_by_locator[_locator(family, index)]),
            )
            for index in range(2)
        )
        row = declared[family]
        return ExternalLocatorListingV3.fixture(
            source_family=family,
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=sum(asset.upstream_bytes for asset in assets),
            available_bytes_basis=row.available_bytes_basis,
            assets=assets,
        )

    enumeration = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )
    payloads = {
        asset.asset_identity_sha256: payloads_by_locator[asset.asset_locator]
        for family in enumeration.families
        for asset in family.assets
    }
    targets = {
        family.source_family: (
            1
            if family.source_family == "wikipedia_wikibooks"
            else sum(asset.upstream_bytes for asset in family.assets[:2])
        )
        for family in enumeration.families
    }
    return enumeration, payloads, targets


def test_prefix_selection_is_minimal_and_keeps_both_science_reserves() -> None:
    enumeration, unused, targets = _fixture_enumeration()
    del unused
    plan, receipt = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )

    rows = {row.source_family: row for row in receipt.families}
    assert rows["wikipedia_wikibooks"].selected_asset_count == 2
    assert rows["wikipedia_wikibooks"].selection_rule == (
        "complete_pinned_wikipedia_asset_set"
    )
    assert rows["arxiv"].selected_asset_count == 2
    assert rows["olmocr"].selected_asset_count == 2
    assert all(
        row.selected_upstream_bytes >= row.required_bytes
        for row in receipt.families
    )
    assert len(plan.assets) == 14
    again_plan, again_receipt = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )
    assert again_plan == plan
    assert again_receipt == receipt


def test_fetch_is_resumable_and_persists_replayable_receipts(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    opened: list[str] = []

    def opener(asset):
        opened.append(asset.asset_identity_sha256)
        return io.BytesIO(payloads[asset.asset_identity_sha256])

    cache_root = tmp_path / "cache"
    receipt_root = tmp_path / "receipts"
    plan, unused_selection = select_required_asset_prefixes_v3(
        enumeration,
        required_bytes_by_family=targets,
    )
    del unused_selection
    first_upstream = plan.assets[0]
    suffix = next(
        value
        for value in (".jsonl.zst", ".json.gz", ".parquet")
        if first_upstream.asset_locator.endswith(value)
    )
    interrupted = (
        cache_root
        / "assets"
        / first_upstream.source_family
        / f"{first_upstream.asset_identity_sha256}{suffix}.partial"
    )
    interrupted.parent.mkdir(parents=True)
    interrupted.write_bytes(b"interrupted")
    first = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    assert not interrupted.exists()
    assert len(opened) == len(first.plan.assets)
    governed_paths = (
        receipt_root / ENUMERATION_ARTIFACT_NAME,
        receipt_root / SELECTION_ARTIFACT_NAME,
        receipt_root / SOURCE_MANIFEST_NAME,
        receipt_root / DOWNLOAD_ARTIFACT_NAME,
        *tuple(
            cache_root.joinpath(*PurePosixPath(asset.relative_path).parts)
            for asset in first.download.source_manifest.assets
        ),
    )
    before = {path: (path.stat().st_mtime_ns, _sha(path.read_bytes())) for path in governed_paths}

    second = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=cache_root,
        receipt_root=receipt_root,
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    assert second == first
    assert len(opened) == len(first.plan.assets)
    assert before == {
        path: (path.stat().st_mtime_ns, _sha(path.read_bytes())) for path in governed_paths
    }
    assert all(path.read_bytes().endswith(b"\n") for path in governed_paths[:4])
    assert all(not path.read_bytes().endswith(b"\n\n") for path in governed_paths[:4])

    replay_enumeration = load_upstream_enumeration_receipt_v3(
        receipt_root / ENUMERATION_ARTIFACT_NAME
    )
    replay_download, verified = load_source_cache_download_receipt_v3(
        receipt_root / DOWNLOAD_ARTIFACT_NAME,
        enumeration=replay_enumeration,
        source_manifest_path=receipt_root / SOURCE_MANIFEST_NAME,
        cache_root=cache_root,
        route_manifest_path=SOURCE_ROUTE_MANIFEST_PATH,
    )
    assert replay_download == first.download
    assert verified.source_manifest == first.download.source_manifest


def test_completed_cache_tamper_fails_without_redownload(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    calls = 0

    def opener(asset):
        nonlocal calls
        calls += 1
        return io.BytesIO(payloads[asset.asset_identity_sha256])

    result = prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=tmp_path / "cache",
        receipt_root=tmp_path / "receipts",
        open_upstream=opener,
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    first_asset = result.download.source_manifest.assets[0]
    path = (tmp_path / "cache").joinpath(
        *PurePosixPath(first_asset.relative_path).parts
    )
    tampered = b"!" + path.read_bytes()[1:]
    path.write_bytes(tampered)
    before_calls = calls
    with pytest.raises(SourceTransportError, match="differs from upstream"):
        prepare_selected_source_cache_v3(
            enumeration=enumeration,
            cache_root=tmp_path / "cache",
            receipt_root=tmp_path / "receipts",
            open_upstream=opener,
            required_bytes_by_family=targets,
            allow_nonauthoritative_fixture=True,
        )
    assert calls == before_calls
    assert path.read_bytes() == tampered


def test_external_transport_cache_uses_completed_bytes_without_overwrite(
    tmp_path: Path,
) -> None:
    resources = {
        "https://huggingface.co/datasets/allenai/dolma/resolve/rev/urls/v1_7.txt": b"parent\n",
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz": b"wiki-0",
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz": b"wiki-1",
    }
    calls: list[str] = []

    def open_url(locator: str):
        calls.append(locator)
        return io.BytesIO(resources[locator])

    cache = ExternalResourceCacheV3(tmp_path / "external", open_url=open_url)
    for locator, expected in resources.items():
        with cache.open(locator) as handle:
            assert handle.read() == expected
    assert calls == list(resources)
    paths = {
        row.locator: (tmp_path / "external").joinpath(
            *PurePosixPath(row.relative_path).parts
        )
        for row in cache.observations
    }
    before = {locator: path.stat().st_mtime_ns for locator, path in paths.items()}
    for locator, expected in resources.items():
        with cache.open(locator) as handle:
            assert handle.read() == expected
    assert calls == list(resources)
    assert before == {locator: path.stat().st_mtime_ns for locator, path in paths.items()}
    assert all(row.cache_hit for row in cache.observations)

    first_locator = next(iter(resources))
    tampered = b"!" + paths[first_locator].read_bytes()[1:]
    paths[first_locator].write_bytes(tampered)
    with pytest.raises(SourceFetchError, match="immutable receipt"):
        cache.open(first_locator)
    assert paths[first_locator].read_bytes() == tampered
    assert calls == list(resources)


def test_default_external_transport_rejects_redirect_before_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedirectedResponse(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"untrusted")
            self.was_closed = False

        def geturl(self) -> str:
            return "https://unapproved.example.invalid/redirected"

        def close(self) -> None:
            self.was_closed = True
            super().close()

    response = RedirectedResponse()
    monkeypatch.setattr(
        fetch_module,
        "urlopen",
        lambda request, timeout: response,
    )
    cache = ExternalResourceCacheV3(tmp_path / "redirect-cache")
    locator = "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz"
    with pytest.raises(SourceFetchError, match="redirect is forbidden"):
        cache.open(locator)
    assert response.was_closed
    assert not tuple((tmp_path / "redirect-cache").rglob("*.bin"))


def test_existing_receipt_artifact_is_never_overwritten(tmp_path: Path) -> None:
    enumeration, payloads, targets = _fixture_enumeration()
    receipt_root = tmp_path / "receipts"
    prepare_selected_source_cache_v3(
        enumeration=enumeration,
        cache_root=tmp_path / "cache",
        receipt_root=receipt_root,
        open_upstream=lambda asset: io.BytesIO(payloads[asset.asset_identity_sha256]),
        required_bytes_by_family=targets,
        allow_nonauthoritative_fixture=True,
    )
    path = receipt_root / SELECTION_ARTIFACT_NAME
    tampered = path.read_bytes().replace(b"minimal_seeded", b"tampered_seeded", 1)
    path.write_bytes(tampered)
    with pytest.raises(SourceFetchError, match="refusing overwrite"):
        prepare_selected_source_cache_v3(
            enumeration=enumeration,
            cache_root=tmp_path / "cache",
            receipt_root=receipt_root,
            open_upstream=lambda asset: io.BytesIO(
                payloads[asset.asset_identity_sha256]
            ),
            required_bytes_by_family=targets,
            allow_nonauthoritative_fixture=True,
        )
    assert path.read_bytes() == tampered
