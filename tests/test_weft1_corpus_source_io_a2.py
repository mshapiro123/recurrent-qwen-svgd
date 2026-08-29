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

from training.weft1_corpus_enumeration_a2 import (
    FIXTURE_MODE,
    ExternalLocatorAssetV3,
    ExternalLocatorListingV3,
    UpstreamAssetV3,
    enumerate_upstream_assets_v3,
)
from training.weft1_corpus_source_io_a2 import (
    DROP_INVALID_UTF8,
    DROP_QUALITY_LT3,
    PRODUCTION_PARSER_BINDINGS_V3,
    RETAIN,
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
        "stackedu": f"data/stack_edu-unit/{index:05d}.jsonl.zst",
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
        asset_locator=_locator(family),
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


@pytest.mark.parametrize("family", ("finemath_3plus", "fineweb_edu"))
def test_exact_production_parquet_schemas_emit_typed_records(
    tmp_path: Path,
    family: str,
) -> None:
    scratch = tmp_path / f"make-{family}.parquet"
    payload = _parquet_payload(scratch, family)
    asset, root = _asset_for_bytes(tmp_path, family, payload)
    events = tuple(iter_source_asset_events_v3(asset, root))
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
    assert PRODUCTION_PARSER_BINDINGS_V3["stackedu"].int_score_path == (
        "metadata",
        "int_score",
    )
    assert PRODUCTION_PARSER_BINDINGS_V3["finemath_3plus"].native_id_path is None
    assert PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"].native_id_path == ("id",)
    assert all(
        binding.authority != "FIXTURE_ONLY"
        for binding in PRODUCTION_PARSER_BINDINGS_V3.values()
    )
