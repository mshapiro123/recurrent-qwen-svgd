from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from training import weft1_corpus_source_io_a2 as source_io
from training.weft1_corpus_enumeration_a2 import (
    FIXTURE_MODE,
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
    UpstreamAssetV3,
    enumerate_upstream_assets_v3,
)
from training.weft1_corpus_fetch_a3 import SourceCacheAssetV4
from training.weft1_corpus_source_io_a2 import (
    DROP_INVALID_UTF8,
    DROP_QUALITY_LT3,
    FINEWEB_SELECTED_SCHEMA_CENSUS_PATH_V1,
    FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_BYTES_V1,
    FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_SHA256_V1,
    FINEWEB_SELECTED_SCHEMA_CENSUS_RECEIPT_SHA256_V1,
    PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3,
    PRODUCTION_PARSER_BINDINGS_V3,
    RETAIN,
    STACKEDU_NORMALIZED_PARSER_BINDING_V3,
    STACKEDU_PYTHON_PARSER_BINDING_V3,
    SourceContainerError,
    SourceSchemaError,
    SourceTransportError,
    finalize_source_cache_v3,
    materialize_complete_fixture_source_cache_v3,
    fixture_source_parser_binding_v3,
    iter_source_asset_events_v3,
    load_source_cache_download_receipt_v3,
    materialize_parsed_source_spool_v3,
    materialize_source_cache_v3,
    plan_source_cache_assets_v3,
    resolve_production_parser_binding_v3,
    verify_parsed_source_spool_v3,
    write_source_cache_download_receipt_v3,
)
from training.weft1_corpus_sources_a2 import (
    SourceCacheAssetV3,
    VerifiedLocalCacheAssetV3,
    load_exact_source_routes_v3,
    verify_local_source_cache_v3,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _locator(family: str, index: int = 0) -> str:
    return {
        "dolma_web": f"data/common_crawl-unit-0019/{index:05d}.jsonl.zst",
        "wikipedia_wikibooks": (
            f"https://olmo-data.org/dolma-v1_7/wiki/wiki-{index:04d}.json.gz"
        ),
        "stackedu": f"data/stack_edu-Java/shard_{index:08d}.jsonl.zst",
        "finemath_3plus": f"finemath-3plus/train-{index:05d}.parquet",
        "arxiv": f"data/rpj-proofpile-arxiv/{index:05d}.jsonl.zst",
        "olmocr": f"data/olmocr_science_pdfs-unit/{index:05d}.jsonl.zst",
        "fineweb_edu": f"data/CC-MAIN-unit/train-{index:05d}.parquet",
    }[family]


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - exact upstream Git object identity
        f"blob {len(value)}\0".encode("ascii") + value
    ).hexdigest()


def _fixture_enumeration(
    payloads: dict[str, bytes],
    *,
    git_blob_family: str | None = None,
    assets_per_family: int = 1,
):
    routes = load_exact_source_routes_v3()
    trees: dict[tuple[str, str], list[dict[str, object]]] = {}
    for route in routes:
        if route.source_family == "wikipedia_wikibooks":
            continue
        payload = payloads[route.source_family]
        for index in range(assets_per_family):
            entry: dict[str, object] = {
                "blob_id": (
                    _git_blob_sha1(payload)
                    if route.source_family == git_blob_family
                    else "a" * 40
                ),
                "path": _locator(route.source_family, index),
                "size": len(payload),
                "type": "file",
            }
            if route.source_family != git_blob_family:
                entry["lfs"] = {"sha256": _sha(payload), "size": len(payload)}
            trees.setdefault((route.repository, route.revision), []).append(entry)

    def tree(**kwargs: object) -> Iterable[object]:
        return tuple(trees[(str(kwargs["repo_id"]), str(kwargs["revision"]))])

    def external(**kwargs: object) -> ExternalLocatorListingV3:
        payload = payloads["wikipedia_wikibooks"]
        return ExternalLocatorListingV3.fixture(
            source_family="wikipedia_wikibooks",
            external_locator_manifest_sha256=str(
                kwargs["expected_manifest_sha256"]
            ),
            available_bytes=len(payload) * assets_per_family,
            available_bytes_basis="pinned repository card reported UTF-8 bytes",
            assets=tuple(
                ExternalLocatorAssetV3(
                    locator=_locator("wikipedia_wikibooks", index),
                    upstream_bytes=len(payload),
                    content_sha256=_sha(payload),
                )
                for index in range(assets_per_family)
            ),
        )

    return enumerate_upstream_assets_v3(
        list_repo_tree=tree,
        enumerate_external_locators=external,
        mode=FIXTURE_MODE,
    )


def _asset_for_bytes(
    tmp_path: Path,
    family: str,
    payload: bytes,
    *,
    asset_locator: str | None = None,
) -> tuple[VerifiedLocalCacheAssetV3, Path]:
    route = next(
        item for item in load_exact_source_routes_v3() if item.source_family == family
    )
    suffix = {
        "dolma_web": ".jsonl.zst",
        "wikipedia_wikibooks": ".json.gz",
        "stackedu": ".jsonl.zst",
        "finemath_3plus": ".parquet",
        "arxiv": ".jsonl.zst",
        "olmocr": ".jsonl.zst",
        "fineweb_edu": ".parquet",
    }[family]
    relative = f"fixture/{family}{suffix}"
    root = tmp_path / f"cache-{family}"
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    expected = SourceCacheAssetV3(
        source_family=family,
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=asset_locator or _locator(family),
        relative_path=relative,
        bytes=len(payload),
        sha256=_sha(payload),
    )
    return (
        VerifiedLocalCacheAssetV3(
            expected=expected,
            observed_bytes=len(payload),
            observed_sha256=_sha(payload),
        ),
        root,
    )


_FINEWEB_CENSUS_V4_ASSETS = (
    (
        "data/CC-MAIN-2018-30/train-00013-of-00017.parquet",
        2_289_354_131,
        "47ef8acbe973f15fe58ee2fabe8de8c10172378e5f6a0c668a2e8e1491056419",
    ),
    (
        "data/CC-MAIN-2023-40/train-00003-of-00031.parquet",
        2_295_347_141,
        "d1429ae4cca67f8e8d629da9b69726e1ad55076c773a7725ba3d4c7217d20e16",
    ),
    (
        "data/CC-MAIN-2017-13/train-00012-of-00022.parquet",
        2_279_677_242,
        "220c8ad2ba1418c507f0a6459cdd0d0c35b898561bf6a117b18cdbace7bf9b8a",
    ),
)


def _fineweb_census_v4_asset(index: int) -> SourceCacheAssetV4:
    route = next(
        item
        for item in load_exact_source_routes_v3()
        if item.source_family == "fineweb_edu"
    )
    locator, byte_count, sha256 = _FINEWEB_CENSUS_V4_ASSETS[index]
    return SourceCacheAssetV4(
        source_family=route.source_family,
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator=locator,
        relative_path=f"assets/fineweb_edu/{sha256}.parquet",
        bytes=byte_count,
        sha256=sha256,
        effective_route_receipt_sha256=route.a1_route_receipt_sha256,
        execution_binding_sha256="e" * 64,
    )


def _json_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _compress_json(family: str, logical: bytes) -> bytes:
    if family == "wikipedia_wikibooks":
        return gzip.compress(logical, compresslevel=6, mtime=0)
    return zstandard.ZstdCompressor(level=3).compress(logical)


def _production_json_row(family: str, *, score: int = 3) -> dict[str, object]:
    common = {
        "added": "2025-01-01",
        "created": "2024-01-01",
        "id": f"{family}-id",
        "metadata": {},
        "source": family,
        "text": f"retained {family} text",
    }
    if family == "dolma_web":
        return {"id": common["id"], "metadata": {}, "text": common["text"]}
    if family == "wikipedia_wikibooks":
        common["metadata"] = {
            "length": 25,
            "provenance": "wikipedia",
            "revid": "123",
            "url": "https://example.test/wiki",
        }
        common["version"] = "1.0"
        return common
    if family == "stackedu":
        common["metadata"] = {
            "int_score": score,
            "path": "src/example.py",
            "score": 3.5,
            "uri": "https://example.test/repo",
        }
        return common
    if family == "arxiv":
        common.pop("source")
        common["doc"] = {"fixture": True}
        return common
    if family == "olmocr":
        return common
    raise AssertionError(family)


def _stackedu_python_row(*, score: int = 3) -> dict[str, object]:
    return {
        "blob_id": "python-blob-id",
        "detected_licenses": [],
        "download_success": True,
        "int_score": score,
        "language": "Python",
        "length_bytes": 28,
        "license_type": "permissive",
        "path": "src/example.py",
        "repo_name": "example/repository",
        "score": 3.5,
        "src_encoding": "UTF-8",
        "text": "def retained_python(): pass",
    }


def _arrow_type(name: str) -> pa.DataType:
    return {
        "string": pa.string(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float64": pa.float64(),
    }[name]


def _parquet_payload(path: Path, family: str, *, score: int = 3) -> bytes:
    binding = PRODUCTION_PARSER_BINDINGS_V3[family]
    values: dict[str, list[object]] = {}
    fields: list[pa.Field] = []
    for name, kind in binding.declared_parquet_columns:
        fields.append(pa.field(name, _arrow_type(kind), nullable=True))
        if name == "text":
            value: object = f"retained {family} text"
        elif name == "id":
            value = f"{family}-id"
        elif name == "int_score":
            value = score
        elif kind == "string":
            value = f"fixture-{name}"
        elif kind in {"int32", "int64"}:
            value = 1
        else:
            value = 1.0
        values[name] = [value]
    table = pa.Table.from_pydict(values, schema=pa.schema(fields))
    pq.write_table(table, path, compression="NONE")
    return path.read_bytes()


def test_cache_factory_hashes_actual_bytes_for_all_seven_families(
    tmp_path: Path,
) -> None:
    payloads = {
        family: f"payload:{family}".encode("ascii") for family in SOURCE_FAMILIES
    }
    enumeration = _fixture_enumeration(payloads)
    by_identity = {
        item.asset_identity_sha256: payloads[item.source_family]
        for family in enumeration.families
        for item in family.assets
    }

    with pytest.raises(SourceTransportError, match="authoritative enumeration"):
        materialize_complete_fixture_source_cache_v3(
            enumeration,
            tmp_path / "blocked-cache",
            tmp_path / "blocked-manifest.json",
            open_upstream=lambda item: io.BytesIO(by_identity[item.asset_identity_sha256]),
        )

    receipt = materialize_complete_fixture_source_cache_v3(
        enumeration,
        tmp_path / "cache",
        tmp_path / "manifest.json",
        open_upstream=lambda item: io.BytesIO(by_identity[item.asset_identity_sha256]),
        allow_nonauthoritative_fixture=True,
    )
    assert len(receipt.source_manifest.assets) == 7
    assert len(receipt.evidence) == 7
    assert {item.upstream_identity_check for item in receipt.evidence} == {
        "content_sha256"
    }
    for asset in receipt.source_manifest.assets:
        path = (tmp_path / "cache").joinpath(
            *PurePosixPath(asset.relative_path).parts
        )
        assert path.read_bytes() == payloads[asset.source_family]
        assert asset.sha256 == _sha(path.read_bytes())
    assert receipt.receipt_sha256 == receipt.receipt_sha256
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    assert manifest_bytes.endswith(b"\n")
    assert not manifest_bytes.endswith(b"\n\n")
    assert manifest_bytes.count(b"\n") == 1
    assert not tuple(tmp_path.rglob("*.partial"))

    artifact_path = tmp_path / "download-receipt.json"
    artifact_sha = write_source_cache_download_receipt_v3(
        receipt, artifact_path
    )
    assert artifact_sha == _sha(artifact_path.read_bytes())
    replayed, verified = load_source_cache_download_receipt_v3(
        artifact_path,
        enumeration=enumeration,
        source_manifest_path=tmp_path / "manifest.json",
        cache_root=tmp_path / "cache",
    )
    assert replayed == receipt
    assert verified.source_manifest == receipt.source_manifest

    tampered_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered_payload["receipt"]["evidence"][0]["observed_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered-download-receipt.json"
    tampered_path.write_text(
        json.dumps(
            tampered_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises((ValueError, SourceTransportError)):
        load_source_cache_download_receipt_v3(
            tampered_path,
            enumeration=enumeration,
            source_manifest_path=tmp_path / "manifest.json",
            cache_root=tmp_path / "cache",
        )


def test_large_enumeration_fetches_only_typed_selected_subset(
    tmp_path: Path,
) -> None:
    payloads = {
        family: f"payload:{family}".encode("ascii") for family in SOURCE_FAMILIES
    }
    enumeration = _fixture_enumeration(payloads, assets_per_family=256)
    complete = tuple(
        asset for family in enumeration.families for asset in family.assets
    )
    selected = complete[:3]
    combined_plan = plan_source_cache_assets_v3(enumeration, selected)
    incremental_plans = tuple(
        plan_source_cache_assets_v3(enumeration, (asset,)) for asset in selected
    )
    calls: list[str] = []

    def opener(asset: UpstreamAssetV3) -> io.BytesIO:
        calls.append(asset.asset_identity_sha256)
        return io.BytesIO(payloads[asset.source_family])

    materializations = tuple(
        materialize_source_cache_v3(
            enumeration,
            plan,
            tmp_path / "subset-cache",
            open_upstream=opener,
            allow_nonauthoritative_fixture=True,
        )
        for plan in incremental_plans
    )
    assert len(complete) == 7 * 256
    assert calls == [asset.asset_identity_sha256 for asset in selected]
    assert [len(item.cache_assets) for item in materializations] == [1, 1, 1]
    assert len(tuple((tmp_path / "subset-cache").rglob("*.zst"))) == 3
    receipt = finalize_source_cache_v3(
        enumeration,
        materializations,
        tmp_path / "subset-cache",
        tmp_path / "subset-manifest.json",
        allow_nonauthoritative_fixture=True,
    )
    assert len(receipt.source_manifest.assets) == 3
    assert receipt.selection_plan_sha256 == combined_plan.plan_sha256

    with pytest.raises(ValueError, match="duplicated or noncanonical"):
        plan_source_cache_assets_v3(enumeration, tuple(reversed(selected)))
    with pytest.raises(ValueError, match="duplicated or noncanonical"):
        plan_source_cache_assets_v3(enumeration, (selected[0], selected[0]))
    outsider = replace(selected[0], upstream_bytes=selected[0].upstream_bytes + 1)
    with pytest.raises(ValueError, match="outside the enumeration"):
        plan_source_cache_assets_v3(enumeration, (outsider,))


def test_cache_factory_rejects_transport_tamper_and_git_blob_mismatch(
    tmp_path: Path,
) -> None:
    payloads = {
        family: f"payload:{family}".encode("ascii") for family in SOURCE_FAMILIES
    }
    enumeration = _fixture_enumeration(payloads)
    by_identity = {
        item.asset_identity_sha256: payloads[item.source_family]
        for family in enumeration.families
        for item in family.assets
    }
    target = next(
        item
        for family in enumeration.families
        for item in family.assets
        if item.source_family == "arxiv"
    )

    def tampered(item: UpstreamAssetV3) -> io.BytesIO:
        value = by_identity[item.asset_identity_sha256]
        if item == target:
            value = bytes([value[0] ^ 1]) + value[1:]
        return io.BytesIO(value)

    with pytest.raises(SourceTransportError, match="content SHA-256"):
        materialize_complete_fixture_source_cache_v3(
            enumeration,
            tmp_path / "tampered-cache",
            tmp_path / "tampered-manifest.json",
            open_upstream=tampered,
            allow_nonauthoritative_fixture=True,
        )
    assert not tuple(tmp_path.rglob("*.partial"))

    raw = b"git-object"
    git_payloads = dict(payloads)
    git_payloads["arxiv"] = raw
    git_enumeration = _fixture_enumeration(
        git_payloads,
        git_blob_family="arxiv",
    )
    by_git_identity = {
        item.asset_identity_sha256: git_payloads[item.source_family]
        for family in git_enumeration.families
        for item in family.assets
    }
    git_receipt = materialize_complete_fixture_source_cache_v3(
        git_enumeration,
        tmp_path / "git-cache",
        tmp_path / "git-manifest.json",
        open_upstream=lambda item: io.BytesIO(
            by_git_identity[item.asset_identity_sha256]
        ),
        allow_nonauthoritative_fixture=True,
    )
    arxiv_evidence = next(
        item
        for item, asset in zip(
            git_receipt.evidence,
            git_receipt.source_manifest.assets,
            strict=True,
        )
        if asset.source_family == "arxiv"
    )
    assert arxiv_evidence.upstream_identity_check == "git_blob_sha1"


@pytest.mark.parametrize(
    "family",
    ("dolma_web", "wikipedia_wikibooks", "stackedu", "arxiv", "olmocr"),
)
def test_exact_production_json_schemas_emit_typed_records(
    tmp_path: Path,
    family: str,
) -> None:
    row = _production_json_row(family)
    logical = _json_bytes([row])
    payload = _compress_json(family, logical)
    asset, root = _asset_for_bytes(tmp_path, family, payload)
    events = tuple(iter_source_asset_events_v3(asset, root))
    assert len(events) == 1
    assert events[0].disposition == RETAIN
    record = events[0].record
    assert record is not None
    assert record.raw_document.source == family
    assert record.raw_document.stable_source_record_id == (
        record.canonical_record.canonical_source_record_id
    )
    assert record.canonical_record.native_record_id == f"{family}-id"
    if family == "wikipedia_wikibooks":
        assert record.canonical_record.native_record_namespace == "wikipedia"
    else:
        assert record.canonical_record.native_record_namespace is None
    assert record.parser_binding_sha256 == (
        PRODUCTION_PARSER_BINDINGS_V3[family].binding_sha256
    )
    observation = events[0].observation
    assert observation is not None
    assert observation.record_ordinal == 0
    assert observation.source_cache_asset_identity_sha256 == (
        asset.expected.asset_identity_sha256
    )
    assert observation.raw_row_sha256 == _sha(logical[:-1])
    assert observation.canonical_row_sha256 == _sha(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert '"kind":"object"' in observation.observed_schema_canonical_json


def test_wikipedia_same_page_id_uses_full_provenance_namespace(
    tmp_path: Path,
) -> None:
    first = _production_json_row("wikipedia_wikibooks")
    second = _production_json_row("wikipedia_wikibooks")
    first["id"] = second["id"] = "12"
    first["metadata"]["provenance"] = (  # type: ignore[index]
        "en_simple_wiki_v0-0001.json.gz:2755174"
    )
    second["metadata"]["provenance"] = (  # type: ignore[index]
        "en_simple_wiki_v0-0000.json.gz:4"
    )
    first["text"] = "first project page"
    second["text"] = "second project page"
    payload = _compress_json(
        "wikipedia_wikibooks",
        _json_bytes([first, second]),
    )
    asset, root = _asset_for_bytes(
        tmp_path,
        "wikipedia_wikibooks",
        payload,
    )
    records = tuple(
        event.record
        for event in iter_source_asset_events_v3(asset, root)
        if event.record is not None
    )
    assert tuple(
        record.canonical_record.native_record_namespace for record in records
    ) == (
        "en_simple_wiki_v0-0001.json.gz:2755174",
        "en_simple_wiki_v0-0000.json.gz:4",
    )
    assert records[0].raw_document.stable_source_record_id != (
        records[1].raw_document.stable_source_record_id
    )

    second["metadata"]["provenance"] = first["metadata"][  # type: ignore[index]
        "provenance"
    ]
    same_namespace_payload = _compress_json(
        "wikipedia_wikibooks",
        _json_bytes([first, second]),
    )
    same_asset, same_root = _asset_for_bytes(
        tmp_path / "same-namespace",
        "wikipedia_wikibooks",
        same_namespace_payload,
    )
    same_records = tuple(
        event.record
        for event in iter_source_asset_events_v3(same_asset, same_root)
        if event.record is not None
    )
    assert same_records[0].raw_document.stable_source_record_id == (
        same_records[1].raw_document.stable_source_record_id
    )


def test_stackedu_direct_python_variant_is_exact_and_path_resolved(
    tmp_path: Path,
) -> None:
    payload = _compress_json(
        "stackedu",
        _json_bytes([_stackedu_python_row(score=4)]),
    )
    asset, root = _asset_for_bytes(
        tmp_path,
        "stackedu",
        payload,
        asset_locator="data/stack_edu-Python/part-000000054.jsonl.zst",
    )
    assert resolve_production_parser_binding_v3(asset) == (
        STACKEDU_PYTHON_PARSER_BINDING_V3
    )
    events = tuple(iter_source_asset_events_v3(asset, root))
    assert [event.disposition for event in events] == [RETAIN]
    record = events[0].record
    assert record is not None
    assert record.canonical_record.native_record_id == "python-blob-id"
    assert record.canonical_record.int_score == 4
    assert record.parser_binding_sha256 == (
        STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256
    )
    with pytest.raises(SourceSchemaError, match="exact asset variant"):
        tuple(
            iter_source_asset_events_v3(
                asset,
                root,
                binding=STACKEDU_NORMALIZED_PARSER_BINDING_V3,
            )
        )


def test_stackedu_variant_path_and_direct_field_types_fail_closed(
    tmp_path: Path,
) -> None:
    direct_row = _stackedu_python_row()
    payload = _compress_json("stackedu", _json_bytes([direct_row]))
    mismatched, mismatch_root = _asset_for_bytes(
        tmp_path / "path-mismatch",
        "stackedu",
        payload,
        asset_locator="data/stack_edu-Python/shard_00000054.jsonl.zst",
    )
    with pytest.raises(SourceSchemaError, match="no exact governed parser variant"):
        tuple(iter_source_asset_events_v3(mismatched, mismatch_root))

    for index, (field, value, match) in enumerate(
        (
            ("download_success", 1, "download_success.*boolean"),
            ("score", 3, "score.*float"),
        )
    ):
        typed_row = _stackedu_python_row()
        typed_row[field] = value
        typed_payload = _compress_json("stackedu", _json_bytes([typed_row]))
        typed, typed_root = _asset_for_bytes(
            tmp_path / f"type-mismatch-{index}",
            "stackedu",
            typed_payload,
            asset_locator="data/stack_edu-Python/part-000000054.jsonl.zst",
        )
        with pytest.raises(SourceSchemaError, match=match):
            tuple(iter_source_asset_events_v3(typed, typed_root))


def test_stackedu_composite_parser_identity_binds_both_exact_variants() -> None:
    identity = PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3["stackedu"]
    assert identity == (
        "30236658d243ef29c06fbac12fdb999db036661fdd602740dc09a8f9665346f7"
    )
    assert identity not in {
        STACKEDU_NORMALIZED_PARSER_BINDING_V3.binding_sha256,
        STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256,
    }
    assert PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3["dolma_web"] == (
        PRODUCTION_PARSER_BINDINGS_V3["dolma_web"].binding_sha256
    )


@pytest.mark.parametrize("family", ("finemath_3plus", "fineweb_edu"))
def test_exact_production_parquet_schemas_emit_typed_records(
    tmp_path: Path,
    family: str,
) -> None:
    scratch = tmp_path / f"make-{family}.parquet"
    payload = _parquet_payload(scratch, family)
    asset, root = _asset_for_bytes(tmp_path, family, payload)
    binding = PRODUCTION_PARSER_BINDINGS_V3[family]
    allow_fixture = False
    if family == "fineweb_edu":
        # A synthetic same-projection file is not one of the three censused
        # production assets.  Bind its complete Arrow schema explicitly as a
        # non-production fixture instead of weakening the production census.
        binding = replace(
            binding,
            authority="FIXTURE_ONLY",
            authority_sha256=_sha(b"fixture:fineweb-full-schema"),
            declared_parquet_schema_ipc_sha256=_sha(
                pq.ParquetFile(scratch).schema_arrow.serialize().to_pybytes()
            ),
        )
        allow_fixture = True
    events = tuple(
        iter_source_asset_events_v3(
            asset,
            root,
            binding=binding,
            allow_fixture_binding=allow_fixture,
        )
    )
    assert [event.disposition for event in events] == [RETAIN]
    record = events[0].record
    assert record is not None
    if family == "finemath_3plus":
        assert record.canonical_record.native_record_id is None
    else:
        assert record.canonical_record.native_record_id == "fineweb_edu-id"
    assert record.canonical_record.int_score == 3
    observation = events[0].observation
    assert observation is not None
    assert observation.row_representation == "arrow_ipc_single_row"
    assert observation.observed_arrow_schema_ipc_hex
    observed_schema = pa.ipc.read_schema(
        pa.BufferReader(bytes.fromhex(observation.observed_arrow_schema_ipc_hex))
    )
    assert observed_schema == pq.ParquetFile(scratch).schema_arrow


def test_fineweb_production_binding_rejects_same_projection_noncensus_asset(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "same-projection.parquet"
    payload = _parquet_payload(scratch, "fineweb_edu")
    asset, root = _asset_for_bytes(tmp_path, "fineweb_edu", payload)
    with pytest.raises(
        SourceSchemaError,
        match="full Arrow schema identity|absent from the selected-schema census",
    ):
        tuple(iter_source_asset_events_v3(asset, root))


@pytest.mark.parametrize("index", range(3))
def test_fineweb_census_projects_real_v4_asset_shape_to_frozen_v3_identity(
    index: int,
) -> None:
    asset = _fineweb_census_v4_asset(index)
    v3_asset = SourceCacheAssetV3(
        source_family=asset.source_family,
        repository=asset.repository,
        config=asset.config,
        revision=asset.revision,
        split=asset.split,
        asset_locator=asset.asset_locator,
        relative_path=asset.relative_path,
        bytes=asset.bytes,
        sha256=asset.sha256,
    )
    census_row = source_io._fineweb_selected_census_rows_v1()[index]
    census_identity = str(census_row["source_asset_identity_sha256"])

    assert asset.asset_identity_sha256 != census_identity
    assert v3_asset.asset_identity_sha256 == census_identity
    assert (
        source_io._fineweb_selected_census_asset_identity_v3(asset)
        == census_identity
    )
    verified = VerifiedLocalCacheAssetV3(
        expected=asset,
        observed_bytes=asset.bytes,
        observed_sha256=asset.sha256,
    )
    assert source_io._fineweb_selected_census_row_for_asset_v1(verified) == (
        census_row
    )


def test_fineweb_census_v3_projection_rejects_changed_transport() -> None:
    original = _fineweb_census_v4_asset(0)
    changed = replace(original, bytes=original.bytes + 1, sha256="0" * 64)
    assert source_io._fineweb_selected_census_asset_identity_v3(changed) != (
        source_io._fineweb_selected_census_asset_identity_v3(original)
    )
    verified = VerifiedLocalCacheAssetV3(
        expected=changed,
        observed_bytes=changed.bytes,
        observed_sha256=changed.sha256,
    )
    with pytest.raises(SourceSchemaError, match="absent from the selected-schema census"):
        source_io._fineweb_selected_census_row_for_asset_v1(verified)


def test_fineweb_census_v3_projection_rejects_changed_asset_order(
    tmp_path: Path,
) -> None:
    verified = tuple(
        VerifiedLocalCacheAssetV3(
            expected=asset,
            observed_bytes=asset.bytes,
            observed_sha256=asset.sha256,
        )
        for asset in (_fineweb_census_v4_asset(index) for index in range(3))
    )
    with pytest.raises(
        SourceSchemaError,
        match="selected asset identity or order differs from census",
    ):
        source_io.validate_fineweb_selected_schema_census_assets_v1(
            (verified[1], verified[0], verified[2]),
            tmp_path,
        )


def test_fineweb_census_lookup_preserves_v4_parsed_observation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "fineweb-v4-observation.parquet"
    payload = _parquet_payload(scratch, "fineweb_edu")
    v3_verified, root = _asset_for_bytes(tmp_path, "fineweb_edu", payload)
    base = v3_verified.expected
    asset = SourceCacheAssetV4(
        source_family=base.source_family,
        repository=base.repository,
        config=base.config,
        revision=base.revision,
        split=base.split,
        asset_locator=base.asset_locator,
        relative_path=base.relative_path,
        bytes=base.bytes,
        sha256=base.sha256,
        effective_route_receipt_sha256="d" * 64,
        execution_binding_sha256="e" * 64,
    )
    census_identity = source_io._fineweb_selected_census_asset_identity_v3(asset)
    assert census_identity != asset.asset_identity_sha256
    monkeypatch.setattr(
        source_io,
        "_fineweb_selected_census_rows_v1",
        lambda: (
            {
                "row_count": 1,
                "source_asset_bytes": asset.bytes,
                "source_asset_identity_sha256": census_identity,
                "source_asset_sha256": asset.sha256,
            },
        ),
    )
    binding = replace(
        PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"],
        declared_parquet_schema_ipc_sha256=_sha(
            pq.ParquetFile(scratch).schema_arrow.serialize().to_pybytes()
        ),
    )
    monkeypatch.setattr(
        source_io,
        "resolve_production_parser_binding_v3",
        lambda _asset: binding,
    )
    verified = VerifiedLocalCacheAssetV3(
        expected=asset,
        observed_bytes=asset.bytes,
        observed_sha256=asset.sha256,
    )

    events = tuple(iter_source_asset_events_v3(verified, root))
    assert len(events) == 1
    assert events[0].record is not None
    assert type(events[0].record.canonical_record.asset) is SourceCacheAssetV4
    assert events[0].record.canonical_record.asset.asset_identity_sha256 == (
        asset.asset_identity_sha256
    )
    assert events[0].observation is not None
    assert events[0].observation.source_cache_asset_identity_sha256 == (
        asset.asset_identity_sha256
    )


def test_parquet_full_schema_binding_detects_metadata_only_drift(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.parquet"
    clean_payload = _parquet_payload(clean_path, "fineweb_edu")
    clean_schema = pq.ParquetFile(clean_path).schema_arrow
    fixture_binding = replace(
        PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"],
        authority="FIXTURE_ONLY",
        authority_sha256=_sha(b"fixture:fineweb-metadata"),
        declared_parquet_schema_ipc_sha256=_sha(
            clean_schema.serialize().to_pybytes()
        ),
    )
    clean_asset, clean_root = _asset_for_bytes(
        tmp_path / "clean", "fineweb_edu", clean_payload
    )
    assert tuple(
        iter_source_asset_events_v3(
            clean_asset,
            clean_root,
            binding=fixture_binding,
            allow_fixture_binding=True,
        )
    )

    changed_table = pq.read_table(clean_path).replace_schema_metadata(
        {b"schema-version": b"changed"}
    )
    changed_path = tmp_path / "changed.parquet"
    pq.write_table(changed_table, changed_path, compression="NONE")
    changed_asset, changed_root = _asset_for_bytes(
        tmp_path / "changed", "fineweb_edu", changed_path.read_bytes()
    )
    with pytest.raises(SourceSchemaError, match="full Arrow schema identity drifted"):
        tuple(
            iter_source_asset_events_v3(
                changed_asset,
                changed_root,
                binding=fixture_binding,
                allow_fixture_binding=True,
            )
        )


def test_factory_parsed_spool_binds_receipts_events_and_text_bytes(
    tmp_path: Path,
) -> None:
    logical = _json_bytes([_production_json_row("dolma_web")])
    dolma_payload = _compress_json("dolma_web", logical)
    payloads = {
        family: (
            dolma_payload
            if family == "dolma_web"
            else f"unused:{family}".encode("ascii")
        )
        for family in SOURCE_FAMILIES
    }
    enumeration = _fixture_enumeration(payloads)
    selected = next(
        asset
        for family in enumeration.families
        for asset in family.assets
        if asset.source_family == "dolma_web"
    )
    plan = plan_source_cache_assets_v3(enumeration, (selected,))
    materialized = materialize_source_cache_v3(
        enumeration,
        plan,
        tmp_path / "spool-cache",
        open_upstream=lambda unused: io.BytesIO(dolma_payload),
        allow_nonauthoritative_fixture=True,
    )
    download = finalize_source_cache_v3(
        enumeration,
        (materialized,),
        tmp_path / "spool-cache",
        tmp_path / "spool-manifest.json",
        allow_nonauthoritative_fixture=True,
    )
    verified = verify_local_source_cache_v3(
        tmp_path / "spool-manifest.json",
        tmp_path / "spool-cache",
    )
    spool_path = tmp_path / "parsed-source.jsonl"
    receipt = materialize_parsed_source_spool_v3(
        enumeration,
        download,
        verified,
        tmp_path / "spool-cache",
        spool_path,
        allow_nonauthoritative_fixture=True,
    )
    assert receipt.event_count == receipt.retained_record_count == 1
    assert len(receipt.observations) == 1
    assert _production_json_row("dolma_web")["text"].encode("utf-8") in (
        spool_path.read_bytes()
    )
    verify_parsed_source_spool_v3(receipt, spool_path)

    spool_path.write_bytes(spool_path.read_bytes().replace(b"retained", b"altered!", 1))
    with pytest.raises((SourceTransportError, SourceSchemaError)):
        verify_parsed_source_spool_v3(receipt, spool_path)


def test_generic_spool_binds_stackedu_composite_variant_identity(
    tmp_path: Path,
) -> None:
    stack_payload = _compress_json(
        "stackedu",
        _json_bytes([_production_json_row("stackedu")]),
    )
    payloads = {
        family: (
            stack_payload
            if family == "stackedu"
            else f"unused:{family}".encode("ascii")
        )
        for family in SOURCE_FAMILIES
    }
    enumeration = _fixture_enumeration(payloads)
    selected = next(
        asset
        for family in enumeration.families
        for asset in family.assets
        if asset.source_family == "stackedu"
    )
    plan = plan_source_cache_assets_v3(enumeration, (selected,))
    materialized = materialize_source_cache_v3(
        enumeration,
        plan,
        tmp_path / "stack-cache",
        open_upstream=lambda unused: io.BytesIO(stack_payload),
        allow_nonauthoritative_fixture=True,
    )
    download = finalize_source_cache_v3(
        enumeration,
        (materialized,),
        tmp_path / "stack-cache",
        tmp_path / "stack-manifest.json",
        allow_nonauthoritative_fixture=True,
    )
    verified = verify_local_source_cache_v3(
        tmp_path / "stack-manifest.json",
        tmp_path / "stack-cache",
    )
    receipt = materialize_parsed_source_spool_v3(
        enumeration,
        download,
        verified,
        tmp_path / "stack-cache",
        tmp_path / "stack-spool.jsonl",
        allow_nonauthoritative_fixture=True,
    )
    assert receipt.parser_bindings == (
        (
            "stackedu",
            PRODUCTION_PARSER_COMPOSITE_IDENTITIES_V3["stackedu"],
        ),
    )
    assert receipt.observations[0].source_family == "stackedu"


def test_invalid_utf8_is_whole_document_drop_and_low_score_is_explicit(
    tmp_path: Path,
) -> None:
    good = _json_bytes([_production_json_row("stackedu", score=2)])
    payload = _compress_json("stackedu", b"\xff\n" + good)
    asset, root = _asset_for_bytes(tmp_path, "stackedu", payload)
    events = tuple(iter_source_asset_events_v3(asset, root))
    assert [event.disposition for event in events] == [
        DROP_INVALID_UTF8,
        DROP_QUALITY_LT3,
    ]
    assert all(event.record is None and event.reason for event in events)


def test_stackedu_low_score_occurrence_does_not_block_later_retained_repeat(
    tmp_path: Path,
) -> None:
    low = _production_json_row("stackedu", score=2)
    retained = _production_json_row("stackedu", score=4)
    assert low["id"] == retained["id"]
    assert low["text"] == retained["text"]
    payload = _compress_json("stackedu", _json_bytes([low, retained]))
    asset, root = _asset_for_bytes(tmp_path, "stackedu", payload)
    events = tuple(iter_source_asset_events_v3(asset, root))
    assert [event.disposition for event in events] == [
        DROP_QUALITY_LT3,
        RETAIN,
    ]
    assert events[0].record is None
    assert events[1].record is not None
    assert events[1].record.canonical_record.int_score == 4


def test_json_schema_drift_duplicate_keys_and_malformed_container_fail_closed(
    tmp_path: Path,
) -> None:
    row = _production_json_row("dolma_web")
    row["unknown"] = True
    payload = _compress_json("dolma_web", _json_bytes([row]))
    asset, root = _asset_for_bytes(tmp_path, "dolma_web", payload)
    with pytest.raises(SourceSchemaError, match="top-level schema drifted"):
        tuple(iter_source_asset_events_v3(asset, root))

    duplicate = b'{"id":"a","id":"b","metadata":{},"text":"x"}\n'
    payload = _compress_json("dolma_web", duplicate)
    asset, root = _asset_for_bytes(tmp_path / "duplicate", "dolma_web", payload)
    with pytest.raises(SourceSchemaError, match="repeats a key"):
        tuple(iter_source_asset_events_v3(asset, root))

    valid = _compress_json(
        "dolma_web", _json_bytes([_production_json_row("dolma_web")])
    )
    payload = valid[:-3]
    asset, root = _asset_for_bytes(tmp_path / "truncated", "dolma_web", payload)
    with pytest.raises(SourceContainerError, match="malformed or truncated"):
        tuple(iter_source_asset_events_v3(asset, root))


def test_parquet_schema_drift_and_post_verification_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "text": pa.array(["text"], type=pa.string()),
            "id": pa.array(["id"], type=pa.string()),
            "int_score": pa.array([3], type=pa.int32()),
        }
    )
    scratch = tmp_path / "wrong.parquet"
    pq.write_table(table, scratch)
    asset, root = _asset_for_bytes(
        tmp_path / "wrong-schema", "fineweb_edu", scratch.read_bytes()
    )
    with pytest.raises(SourceSchemaError, match="column projection drifted"):
        tuple(iter_source_asset_events_v3(asset, root))

    payload = _compress_json(
        "dolma_web", _json_bytes([_production_json_row("dolma_web")])
    )
    asset, root = _asset_for_bytes(tmp_path / "tampered", "dolma_web", payload)
    path = root.joinpath(*PurePosixPath(asset.expected.relative_path).parts)
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(SourceTransportError, match="changed hash"):
        tuple(iter_source_asset_events_v3(asset, root))


def test_parser_consumes_one_verified_snapshot_without_reopening_cache_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _compress_json(
        "dolma_web", _json_bytes([_production_json_row("dolma_web")])
    )
    replacement = _compress_json(
        "dolma_web",
        _json_bytes(
            [
                {
                    "id": "replacement",
                    "metadata": {},
                    "text": "replacement text",
                }
            ]
        ),
    )
    asset, root = _asset_for_bytes(tmp_path / "snapshot-race", "dolma_web", payload)
    path = root.joinpath(*PurePosixPath(asset.expected.relative_path).parts)
    original_open = Path.open
    source_opens = 0

    def racing_open(self: Path, mode: str = "r", *args: object, **kwargs: object):
        nonlocal source_opens
        if self == path and mode == "rb":
            source_opens += 1
            if source_opens == 2:
                original_open(path, "wb").write(replacement)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    events = tuple(iter_source_asset_events_v3(asset, root))
    assert source_opens == 1
    assert events[0].record is not None
    assert events[0].record.raw_document.text == "retained dolma_web text"


def test_fixture_binding_is_explicitly_nonproduction(tmp_path: Path) -> None:
    binding = fixture_source_parser_binding_v3("dolma_web")
    payload = _compress_json(
        "dolma_web", _json_bytes([{"id": "fixture", "text": "text"}])
    )
    asset, root = _asset_for_bytes(tmp_path, "dolma_web", payload)
    with pytest.raises(SourceSchemaError, match="fixture-only"):
        tuple(iter_source_asset_events_v3(asset, root, binding=binding))
    events = tuple(
        iter_source_asset_events_v3(
            asset,
            root,
            binding=binding,
            allow_fixture_binding=True,
        )
    )
    assert [event.disposition for event in events] == [RETAIN]


def test_production_bindings_cover_all_families_and_pin_quality_paths() -> None:
    assert tuple(PRODUCTION_PARSER_BINDINGS_V3) == SOURCE_FAMILIES
    assert PRODUCTION_PARSER_BINDINGS_V3[
        "wikipedia_wikibooks"
    ].native_record_namespace_path == ("metadata", "provenance")
    assert PRODUCTION_PARSER_BINDINGS_V3["stackedu"].int_score_path == (
        "metadata",
        "int_score",
    )
    assert STACKEDU_PYTHON_PARSER_BINDING_V3.native_id_path == ("blob_id",)
    assert STACKEDU_PYTHON_PARSER_BINDING_V3.int_score_path == ("int_score",)
    assert PRODUCTION_PARSER_BINDINGS_V3["finemath_3plus"].native_id_path is None
    assert PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"].native_id_path == ("id",)
    census_bytes = FINEWEB_SELECTED_SCHEMA_CENSUS_PATH_V1.read_bytes()
    assert len(census_bytes) == FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_BYTES_V1
    assert _sha(census_bytes) == FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_SHA256_V1
    census = json.loads(census_bytes)
    assert census["receipt_sha256"] == (
        FINEWEB_SELECTED_SCHEMA_CENSUS_RECEIPT_SHA256_V1
    )
    assert census["selected_asset_count"] == 3
    assert census["claim_scope"] == (
        "THREE_SELECTED_FINEWEB_EDU_ASSETS_ONLY_NOT_UPSTREAM_FAMILY"
    )
    assert PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"].authority == (
        "PINNED_ASSET_DECLARATION"
    )
    assert PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"].authority_sha256 == (
        FINEWEB_SELECTED_SCHEMA_CENSUS_PHYSICAL_SHA256_V1
    )
    assert all(
        binding.authority != "FIXTURE_ONLY"
        for binding in PRODUCTION_PARSER_BINDINGS_V3.values()
    )
