from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace
from typing import Iterable

import pytest

from training.weft1_corpus_enumeration_a2 import (
    AUTHORITATIVE_MODE,
    FIXTURE_MODE,
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
    UpstreamEnumerationReceiptV3,
    enumerate_authoritative_upstream_assets_v3,
    enumerate_upstream_assets_v3,
    load_upstream_enumeration_receipt_v3,
    locator_matches_route_v3,
    read_pinned_external_locator_listing_v3,
    write_upstream_enumeration_receipt_v3,
)
from training.weft1_corpus_sources_a2 import load_exact_source_routes_v3
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    load_source_route_manifest,
)


def _locator(family: str, index: int) -> str:
    return {
        "dolma_web": f"data/common_crawl-unit-0019/{index:05d}.jsonl.zst",
        "wikipedia_wikibooks": (
            f"https://olmo-data.org/dolma-v1_7/wiki/wiki-{index:05d}.json.gz"
        ),
        "stackedu": f"data/stack_edu-unit/{index:05d}.jsonl.zst",
        "finemath_3plus": f"finemath-3plus/train-{index:05d}.parquet",
        "arxiv": f"data/rpj-proofpile-arxiv/{index:05d}.jsonl.zst",
        "olmocr": f"data/olmocr_science_pdfs-unit/{index:05d}.jsonl.zst",
        "fineweb_edu": f"data/CC-MAIN-unit/train-{index:05d}.parquet",
    }[family]


def _entry(family: str, index: int, size: int) -> dict[str, object]:
    locator = _locator(family, index)
    return {
        "blob_id": hashlib.sha1(locator.encode("utf-8")).hexdigest(),
        "path": locator,
        "size": size,
        "type": "file",
    }


class OfflineTree:
    def __init__(self, trees: dict[tuple[str, str], list[dict[str, object]]]):
        self.trees = trees
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Iterable[object]:
        self.calls.append(dict(kwargs))
        return tuple(self.trees[(str(kwargs["repo_id"]), str(kwargs["revision"]))])


def _fixture_inputs(
    *, reverse: bool = False,
) -> tuple[OfflineTree, object]:
    routes = load_exact_source_routes_v3()
    trees: dict[tuple[str, str], list[dict[str, object]]] = {}
    for route in routes:
        if route.source_family == "wikipedia_wikibooks":
            continue
        key = (route.repository, route.revision)
        trees.setdefault(key, []).append(
            _entry(route.source_family, 1, 100 + SOURCE_FAMILIES.index(route.source_family))
        )
    for rows in trees.values():
        rows.append(
            {
                "blob_id": "f" * 40,
                "path": "README.md",
                "size": 10,
                "type": "file",
            }
        )
        rows.append({"path": "data", "type": "directory"})
        if reverse:
            rows.reverse()
    external_rows = (
        ExternalLocatorAssetV3(
            locator=_locator("wikipedia_wikibooks", 1),
            upstream_bytes=20,
            content_sha256="1" * 64,
        ),
        ExternalLocatorAssetV3(
            locator=_locator("wikipedia_wikibooks", 0),
            upstream_bytes=10,
            content_sha256="2" * 64,
        ),
    )

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        assert kwargs["route"].source_family == "wikipedia_wikibooks"  # type: ignore[union-attr]
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=30,
            available_bytes_basis="pinned repository card reported UTF-8 bytes",
            assets=tuple(reversed(external_rows)) if reverse else external_rows,
        )

    return OfflineTree(trees), external


def _authoritative_inputs() -> tuple[OfflineTree, object]:
    routes = {route.source_family: route for route in load_exact_source_routes_v3()}
    declared = {
        route.source_family: route for route in load_source_route_manifest().routes
    }
    trees: dict[tuple[str, str], list[dict[str, object]]] = {}
    for family in SOURCE_FAMILIES:
        if family == "wikipedia_wikibooks":
            continue
        route = routes[family]
        row = declared[family]
        sizes = [1] * row.asset_count
        sizes[-1] = row.available_bytes - row.asset_count + 1
        trees.setdefault((route.repository, route.revision), []).extend(
            _entry(family, index, size) for index, size in enumerate(sizes)
        )

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        row = declared["wikipedia_wikibooks"]
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=row.available_bytes,
            available_bytes_basis=row.available_bytes_basis,
            assets=tuple(
                ExternalLocatorAssetV3(
                    locator=_locator("wikipedia_wikibooks", index),
                    upstream_bytes=index + 1,
                    content_sha256=hashlib.sha256(
                        f"wiki-{index}".encode("ascii")
                    ).hexdigest(),
                )
                for index in range(row.asset_count)
            ),
        )

    return OfflineTree(trees), external


def test_fixture_receipt_is_complete_canonical_and_explicitly_nonauthoritative() -> None:
    tree, external = _fixture_inputs()
    receipt = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )

    assert receipt.mode == "NONAUTHORITATIVE_FIXTURE"
    assert receipt.authoritative is False
    assert tuple(family.source_family for family in receipt.families) == SOURCE_FAMILIES
    assert receipt.total_asset_count == 8
    assert receipt.receipt_sha256 == (
        "a746ad5831f125e435f91e9ae2b8917045f48ee9c80b67482f4bb7e6d469286d"
    )
    assert len(tree.calls) == 4  # the three Dolma-mix routes share one tree observation
    assert all(
        set(call) == {"repo_id", "repo_type", "revision", "recursive", "expand"}
        and call["repo_type"] == "dataset"
        and call["recursive"] is True
        and call["expand"] is False
        for call in tree.calls
    )


def test_input_iteration_order_cannot_change_seeded_receipt() -> None:
    forward_tree, forward_external = _fixture_inputs(reverse=False)
    reverse_tree, reverse_external = _fixture_inputs(reverse=True)
    forward = enumerate_upstream_assets_v3(
        list_repo_tree=forward_tree,
        enumerate_external_locators=forward_external,
        mode=FIXTURE_MODE,
    )
    reverse = enumerate_upstream_assets_v3(
        list_repo_tree=reverse_tree,
        enumerate_external_locators=reverse_external,
        mode=FIXTURE_MODE,
    )
    assert reverse.receipt_sha256 == forward.receipt_sha256
    assert reverse.families == forward.families


def test_every_fixture_locator_is_checked_against_its_exact_route() -> None:
    routes = load_exact_source_routes_v3()
    for route in routes:
        assert locator_matches_route_v3(route, _locator(route.source_family, 0))
        assert not locator_matches_route_v3(route, "data/outside-selector.bin")


def test_external_locator_or_manifest_drift_fails_closed() -> None:
    tree, _ = _fixture_inputs()

    def bad_locator(**kwargs: object) -> ExternalLocatorListingV3:
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=30,
            available_bytes_basis="pinned repository card reported UTF-8 bytes",
            assets=(
                ExternalLocatorAssetV3(
                    locator="https://example.com/wiki-00000.json.gz",
                    upstream_bytes=30,
                    content_sha256="3" * 64,
                ),
            ),
        )

    with pytest.raises(ValueError, match="outside the pinned selector"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=bad_locator,
            mode=FIXTURE_MODE,
        )

    def bad_manifest(**kwargs: object) -> ExternalLocatorListingV3:
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256="4" * 64,
            available_bytes=30,
            available_bytes_basis="pinned repository card reported UTF-8 bytes",
            assets=(
                ExternalLocatorAssetV3(
                    locator=_locator("wikipedia_wikibooks", 0),
                    upstream_bytes=30,
                    content_sha256="3" * 64,
                ),
            ),
        )

    with pytest.raises(ValueError, match="manifest identity drifted"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=bad_manifest,
            mode=FIXTURE_MODE,
        )


def test_selected_hf_asset_without_exact_blob_identity_fails_closed() -> None:
    tree, external = _fixture_inputs()
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == "finemath_3plus"
    )
    selected = next(
        row
        for row in tree.trees[(route.repository, route.revision)]
        if str(row.get("path", "")).startswith("finemath-3plus/")
    )
    selected["blob_id"] = "not-a-blob-id"
    with pytest.raises(ValueError, match="blob_id is not a lowercase hash"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=external,
            mode=FIXTURE_MODE,
        )


def test_missing_family_fails_even_in_fixture_mode() -> None:
    tree, external = _fixture_inputs()
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == "arxiv"
    )
    key = (route.repository, route.revision)
    tree.trees[key] = [
        row
        for row in tree.trees[key]
        if not str(row.get("path", "")).startswith("data/rpj-proofpile-arxiv/")
    ]
    with pytest.raises(ValueError, match="enumerated no assets for arxiv"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=external,
            mode=FIXTURE_MODE,
        )


def test_injected_enumerator_cannot_mint_authoritative_receipt() -> None:
    fixture_tree, fixture_external = _fixture_inputs()
    with pytest.raises(ValueError, match="injected list_repo_tree"):
        enumerate_upstream_assets_v3(
            list_repo_tree=fixture_tree,
            enumerate_external_locators=fixture_external,
            mode=AUTHORITATIVE_MODE,
        )


def test_authoritative_factory_requires_exact_hub_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "training.weft1_corpus_enumeration_a2.metadata.version",
        lambda unused: "1.23.9",
    )
    with pytest.raises(RuntimeError, match="1.24.0 exactly"):
        enumerate_authoritative_upstream_assets_v3(
            open_resource=lambda unused: io.BytesIO(b"unreachable")
        )



def test_concrete_external_reader_hashes_parent_and_selected_assets() -> None:
    route = next(
        item
        for item in load_exact_source_routes_v3()
        if item.source_family == "wikipedia_wikibooks"
    )
    declared = next(
        item
        for item in load_source_route_manifest().routes
        if item.source_family == "wikipedia_wikibooks"
    )
    selected = (
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0000.json.gz",
        "https://olmo-data.org/dolma-v1_7/wiki/wiki-0001.json.gz",
    )
    parent = (
        "https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz\n"
        + "\n".join(selected)
        + "\n"
    ).encode("utf-8")
    parent_sha = hashlib.sha256(parent).hexdigest()
    assets = {selected[0]: b"wiki-zero", selected[1]: b"wiki-one"}

    def opener(locator: str) -> io.BytesIO:
        if locator.endswith("urls/v1_7.txt"):
            return io.BytesIO(parent)
        return io.BytesIO(assets[locator])

    listing = read_pinned_external_locator_listing_v3(
        route=route,
        declared=declared,
        open_resource=opener,
        expected_manifest_sha256=parent_sha,
        allow_nonauthoritative_fixture_hash=True,
    )
    assert listing.parent_manifest_verified is False
    assert listing.external_locator_manifest_bytes == len(parent)
    assert tuple(item.locator for item in listing.assets) == selected
    assert tuple(item.content_sha256 for item in listing.assets) == tuple(
        hashlib.sha256(assets[locator]).hexdigest() for locator in selected
    )

    with pytest.raises(ValueError, match="parent SHA-256 drifted"):
        read_pinned_external_locator_listing_v3(
            route=route,
            declared=declared,
            open_resource=lambda locator: io.BytesIO(
                parent + b"tamper" if locator.endswith("urls/v1_7.txt") else assets[locator]
            ),
            expected_manifest_sha256=parent_sha,
            allow_nonauthoritative_fixture_hash=True,
        )


def test_external_listing_and_xet_only_assets_fail_closed() -> None:
    with pytest.raises(TypeError, match="factory-minted"):
        ExternalLocatorListingV3()  # type: ignore[call-arg]
    tree, external = _fixture_inputs()
    route = next(
        item
        for item in load_exact_source_routes_v3()
        if item.source_family == "finemath_3plus"
    )
    row = next(
        item
        for item in tree.trees[(route.repository, route.revision)]
        if str(item.get("path", "")).startswith("finemath-3plus/")
    )
    row["xet_hash"] = "b" * 64
    with pytest.raises(ValueError, match="Xet-only asset"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=external,
            mode=FIXTURE_MODE,
        )


def test_enumeration_receipt_canonical_round_trip_and_tamper(tmp_path) -> None:
    tree, external = _fixture_inputs()
    receipt = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )
    path = tmp_path / "enumeration.json"
    artifact_sha = write_upstream_enumeration_receipt_v3(receipt, path)
    assert artifact_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    replayed = load_upstream_enumeration_receipt_v3(path)
    assert replayed == receipt
    assert replayed.receipt_sha256 == receipt.receipt_sha256

    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(path.read_bytes().replace(b'"upstream_bytes":100', b'"upstream_bytes":101', 1))
    with pytest.raises(ValueError):
        load_upstream_enumeration_receipt_v3(tampered)


def test_lfs_identity_is_captured_and_size_drift_rejected() -> None:
    tree, external = _fixture_inputs()
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == "finemath_3plus"
    )
    row = next(
        row
        for row in tree.trees[(route.repository, route.revision)]
        if str(row.get("path", "")).startswith("finemath-3plus/")
    )
    row["lfs"] = {"sha256": "a" * 64, "size": row["size"]}
    receipt = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )
    family = next(
        family
        for family in receipt.families
        if family.source_family == "finemath_3plus"
    )
    assert family.assets[0].content_sha256 == "a" * 64

    row["lfs"] = {"sha256": "a" * 64, "size": int(row["size"]) + 1}
    with pytest.raises(ValueError, match="file and LFS sizes disagree"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=external,
            mode=FIXTURE_MODE,
        )


def test_pinned_huggingface_object_shape_needs_no_synthetic_type_field() -> None:
    tree, external = _fixture_inputs()
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == "finemath_3plus"
    )
    key = (route.repository, route.revision)
    raw = next(
        row
        for row in tree.trees[key]
        if str(row.get("path", "")).startswith("finemath-3plus/")
    )
    tree.trees[key] = [
        SimpleNamespace(
            path=raw["path"],
            size=raw["size"],
            blob_id=raw["blob_id"],
            lfs=None,
            xet_hash=None,
        )
    ]
    receipt = enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )
    family = next(
        family
        for family in receipt.families
        if family.source_family == "finemath_3plus"
    )
    assert family.assets[0].blob_identity == raw["blob_id"]


def test_receipt_is_factory_only_and_mode_is_not_inferred() -> None:
    with pytest.raises(TypeError, match="factory-minted"):
        UpstreamEnumerationReceiptV3()  # type: ignore[call-arg]
    tree, external = _fixture_inputs()
    with pytest.raises(ValueError, match="mode must explicitly"):
        enumerate_upstream_assets_v3(
            list_repo_tree=tree,
            enumerate_external_locators=external,
            mode="fixture",
        )
