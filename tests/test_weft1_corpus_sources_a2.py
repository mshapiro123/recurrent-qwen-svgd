from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import training.weft1_strict_io as strict_io

from training.weft1_corpus_sources_a2 import (
    CanonicalSourceRecordV3,
    ExactSourceRouteV3,
    SCIENCE_SOURCE_PRECEDENCE,
    SOURCE_CACHE_SCHEMA_V3,
    SOURCE_ROUTE_MANIFEST_SHA256,
    SourceCacheAssetV3,
    SourceCacheManifestV3,
    VerifiedLocalCacheManifestV3,
    asset_order_digest_v3,
    canonical_asset_order_v3,
    fineweb_ranked_remainder_v3,
    load_exact_source_routes_v3,
    order_family_records_v3,
    order_science_records_v3,
    verify_local_source_cache_v3,
    verify_source_cache_manifest,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    load_source_route_manifest,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _asset(
    source_family: str,
    locator: str,
    *,
    relative_path: str | None = None,
    payload: bytes = b"payload",
) -> SourceCacheAssetV3:
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == source_family
    )
    return SourceCacheAssetV3(
        source_family=source_family,
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=locator,
        relative_path=relative_path or f"{source_family}/{Path(locator).name}",
        bytes=len(payload),
        sha256=_digest(payload),
    )


def _record(
    asset: SourceCacheAssetV3,
    ordinal: int,
    *,
    native_id: str | None = None,
    int_score: int | None = None,
    byte_count: int = 10,
) -> CanonicalSourceRecordV3:
    return CanonicalSourceRecordV3(
        asset=asset,
        source_record_ordinal=ordinal,
        retained_byte_count=byte_count,
        native_record_id=native_id,
        int_score=int_score,
    )


def _write_route_mutation(tmp_path: Path, field: str, value: object) -> Path:
    payload = json.loads(SOURCE_ROUTE_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["routes"][0][field] = value
    if field == "revision":
        repository = payload["routes"][0]["repository"]
        payload["routes"][0]["card_url"] = (
            f"https://huggingface.co/datasets/{repository}/blob/{value}/README.md"
        )
    path = tmp_path / f"drift-{field}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_exact_routes_bind_every_repository_config_revision_and_license() -> None:
    routes = load_exact_source_routes_v3()
    assert tuple(route.source_family for route in routes) == SOURCE_FAMILIES
    assert SCIENCE_SOURCE_PRECEDENCE == ("arxiv", "olmocr")
    assert all(route.declared_license == "odc-by" for route in routes)
    assert all(len(route.revision) == 40 for route in routes)
    assert load_source_route_manifest().manifest_sha256 == (
        SOURCE_ROUTE_MANIFEST_SHA256
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("config", "alternate", "identity drifted|route drifted"),
        ("revision", "1" * 40, "identity drifted|route drifted"),
        ("declared_license", "apache-2.0", "ODC-By|identity drifted"),
    ),
)
def test_exact_route_loader_fails_closed_on_literal_drift(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = _write_route_mutation(tmp_path, field, value)
    with pytest.raises(ValueError, match=match):
        load_exact_source_routes_v3(path)


def test_exact_route_value_cannot_be_replaced_by_valid_looking_config() -> None:
    route = ExactSourceRouteV3.from_a1(load_source_route_manifest().routes[0])
    with pytest.raises(ValueError, match="route drifted"):
        replace(route, config="other-valid-config")


def test_assets_use_seeded_sha256_locator_order() -> None:
    later_family = _asset(
        "fineweb_edu",
        "data/2024/train-z.parquet",
        relative_path="fineweb/file.parquet",
    )
    second = _asset(
        "dolma_web",
        "data/common_crawl-z-0019/file.jsonl.zst",
        relative_path="dolma/z.jsonl.zst",
    )
    first = _asset(
        "dolma_web",
        "data/common_crawl-a-0019/file.jsonl.zst",
        relative_path="dolma/a.jsonl.zst",
    )
    assert asset_order_digest_v3(
        "data/common_crawl-z-0019/file.jsonl.zst"
    ).hex() == (
        "a4bdf75566ba4067d7df1fd3228b2168f0a55e394ca48fc7b1acfd63b3273ebb"
    )
    assert canonical_asset_order_v3((later_family, second, first)) == (
        later_family,
        second,
        first,
    )
    with pytest.raises(ValueError, match="repeats a source asset locator"):
        canonical_asset_order_v3((first, replace(first, relative_path="other/file")))


def test_cache_asset_rejects_repository_config_and_revision_drift() -> None:
    asset = _asset("finemath_3plus", "finemath-3plus/train-000.parquet")
    for field, value in (
        ("repository", "other/repository"),
        ("config", "default"),
        ("revision", "1" * 40),
    ):
        with pytest.raises(ValueError, match="route drifted"):
            replace(asset, **{field: value})


def test_cache_asset_rejects_nonportable_relative_paths() -> None:
    asset = _asset(
        "dolma_web", "data/common_crawl-test-0019/part.jsonl.zst"
    )
    for value in (
        "../escape.bin",
        "nested\\escape.bin",
        "cache/CON.bin",
        "cache/trailing ",
        "cache/file.txt:stream",
    ):
        with pytest.raises(ValueError, match="canonical relative POSIX path"):
            replace(asset, relative_path=value)


def test_native_source_identity_survives_upsampled_asset_occurrences() -> None:
    first_asset = _asset("stackedu", "data/stack_edu-a/one.jsonl.zst")
    second_asset = _asset("stackedu", "data/stack_edu-b/two.jsonl.zst")
    first = _record(first_asset, 1, native_id="stable-upstream-id", int_score=3)
    duplicate = _record(
        second_asset, 999, native_id="stable-upstream-id", int_score=3
    )
    physical = _record(second_asset, 999, int_score=3)
    assert first.canonical_source_record_id == duplicate.canonical_source_record_id
    assert first.canonical_source_record_id != physical.canonical_source_record_id


def test_scored_families_sort_descending_with_canonical_id_tie() -> None:
    asset = _asset("finemath_3plus", "finemath-3plus/train-000.parquet")
    tied_b = _record(asset, 1, native_id="b", int_score=4)
    low = _record(asset, 2, native_id="low", int_score=3)
    tied_a = _record(asset, 3, native_id="a", int_score=4)
    ordered = order_family_records_v3((tied_b, low, tied_a))
    expected_tied = tuple(
        sorted((tied_a, tied_b), key=lambda row: row.canonical_source_record_id)
    )
    assert ordered == (*expected_tied, low)
    with pytest.raises(ValueError, match="repeats a canonical"):
        order_family_records_v3((tied_a, tied_a))


@pytest.mark.parametrize("score", (2, 2.5, True, None))
@pytest.mark.parametrize(
    "family", ("stackedu", "finemath_3plus", "fineweb_edu")
)
def test_quality_gated_sources_require_integer_score_at_least_three(
    family: str, score: object
) -> None:
    locator = {
        "stackedu": "data/stack_edu-test/part.jsonl.zst",
        "finemath_3plus": "finemath-3plus/train-test.parquet",
        "fineweb_edu": "data/test/train-test.parquet",
    }[family]
    with pytest.raises(ValueError, match="integer int_score >= 3"):
        _record(
            _asset(family, locator),
            0,
            int_score=score,  # type: ignore[arg-type]
        )


def test_no_score_family_uses_physical_asset_then_record_order() -> None:
    later_asset = _asset(
        "dolma_web", "data/common_crawl-b-0019/part.jsonl.zst"
    )
    first_asset = _asset(
        "dolma_web", "data/common_crawl-a-0019/part.jsonl.zst"
    )
    rows = (
        _record(later_asset, 0, native_id="0"),
        _record(first_asset, 2, native_id="2"),
        _record(first_asset, 1, native_id="1"),
    )
    assert order_family_records_v3(rows) == (rows[0], rows[2], rows[1])
    with pytest.raises(ValueError, match="may not carry int_score"):
        _record(first_asset, 0, int_score=3)


def test_science_is_arxiv_then_olmocr_regardless_of_input_order() -> None:
    arxiv = _record(
        _asset("arxiv", "data/rpj-proofpile-arxiv/arxiv.jsonl.zst"), 0
    )
    olmocr = _record(
        _asset("olmocr", "data/olmocr_science_pdfs-a/olmocr.jsonl.zst"), 0
    )
    assert order_science_records_v3((olmocr, arxiv)) == (arxiv, olmocr)
    with pytest.raises(ValueError, match="only arXiv and olmOCR"):
        order_science_records_v3(
            (
                arxiv,
                _record(
                    _asset("stackedu", "data/stack_edu-a/x.jsonl.zst"),
                    0,
                    int_score=3,
                ),
            )
        )


def test_fineweb_topup_resumes_ranked_remainder_without_restarting() -> None:
    asset = _asset("fineweb_edu", "data/test/train-part.parquet")
    rows = tuple(
        _record(asset, index, native_id=str(index), int_score=10 - index)
        for index in range(5)
    )
    excluded = frozenset((rows[3].canonical_source_record_id,))
    assert fineweb_ranked_remainder_v3(
        rows, consumed_prefix_count=2, excluded_record_ids=excluded
    ) == (rows[2], rows[4])
    with pytest.raises(ValueError, match="already be canonically ranked"):
        fineweb_ranked_remainder_v3(
            tuple(reversed(rows)), consumed_prefix_count=2
        )


def test_local_cache_verification_is_byte_exact_and_root_independent(
    tmp_path: Path,
) -> None:
    payloads = (("a/first.bin", b"first"), ("b/second.bin", b"second"))
    assets = tuple(
        _asset(
            "dolma_web",
            f"data/common_crawl-test-0019/{index}.jsonl.zst",
            relative_path=relative_path,
            payload=payload,
        )
        for index, (relative_path, payload) in enumerate(payloads)
    )
    manifest = SourceCacheManifestV3(
        schema=SOURCE_CACHE_SCHEMA_V3,
        source_route_manifest_sha256=SOURCE_ROUTE_MANIFEST_SHA256,
        assets=canonical_asset_order_v3(assets),
    )
    manifest_payload = {
        "schema": manifest.schema,
        "source_route_manifest_sha256": manifest.source_route_manifest_sha256,
        "assets": [asset.__dict__ for asset in manifest.assets],
    }
    manifest_path = tmp_path / "source-cache.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    identities = []
    for root_name in ("cache-one", "cache-two"):
        root = tmp_path / root_name
        for relative_path, payload in payloads:
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        verified = verify_local_source_cache_v3(manifest_path, root)
        identities.append(verified.offline_replay_identity_sha256)
        assert verified.total_bytes == len(b"firstsecond")
        summary = verify_source_cache_manifest(manifest_path, root)
        assert summary["asset_count"] == 2
        assert summary["total_bytes"] == len(b"firstsecond")
    assert identities[0] == identities[1] == manifest.offline_replay_identity_sha256


def test_verified_cache_receipt_is_factory_only() -> None:
    with pytest.raises(TypeError, match="factory-minted"):
        VerifiedLocalCacheManifestV3()  # type: ignore[call-arg]


def test_local_cache_verification_rejects_byte_tampering(tmp_path: Path) -> None:
    asset = _asset(
        "dolma_web",
        "data/common_crawl-test-0019/part.jsonl.zst",
        relative_path="dolma/part.bin",
        payload=b"expected",
    )
    manifest_path = tmp_path / "source-cache.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": SOURCE_CACHE_SCHEMA_V3,
                "source_route_manifest_sha256": SOURCE_ROUTE_MANIFEST_SHA256,
                "assets": [asset.__dict__],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "cache"
    destination = root / asset.relative_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count drifted|SHA-256 drifted"):
        verify_local_source_cache_v3(manifest_path, root)


def test_local_cache_verification_rejects_symlinked_asset_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"expected"
    asset = _asset(
        "dolma_web",
        "data/common_crawl-test-0019/part.jsonl.zst",
        relative_path="alias/part.bin",
        payload=payload,
    )
    manifest_path = tmp_path / "source-cache.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": SOURCE_CACHE_SCHEMA_V3,
                "source_route_manifest_sha256": SOURCE_ROUTE_MANIFEST_SHA256,
                "assets": [asset.__dict__],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "cache"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "part.bin").write_bytes(payload)
    alias = root / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        # Windows may deny symlink creation without Developer Mode.  Preserve
        # the source-API adversarial test by simulating the exact lstat result
        # on a real directory containing the otherwise-readable asset.
        alias.mkdir()
        (alias / "part.bin").write_bytes(payload)
        original = strict_io._is_link_or_reparse
        monkeypatch.setattr(
            strict_io,
            "_is_link_or_reparse",
            lambda path: path == alias or original(path),
        )

    with pytest.raises(ValueError, match="symlink/reparse"):
        verify_local_source_cache_v3(manifest_path, root)
