from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest
import zstandard

from training.weft1_corpus_fetch_a3 import SourceCacheAssetV4
from training.weft1_corpus_parsed_asset_cache_v1 import (
    ParsedAssetRecoveryContextV1,
    ParsedAssetRecoveryError,
    iter_parsed_asset_segment_v1,
    parsed_asset_runtime_identity_v1,
    probe_parsed_asset_segment_v1,
    write_parsed_asset_segment_v1,
)
from training.weft1_corpus_source_io_a2 import (
    DROP_QUALITY_LT3,
    RETAIN,
    iter_source_asset_events_v3,
    resolve_production_parser_binding_v3,
)
from training.weft1_corpus_sources_a2 import (
    SourceCacheAssetV3,
    VerifiedLocalCacheAssetV3,
    load_exact_source_routes_v3,
)
from training.weft1_gtok_contract import canonical_json_bytes


def _runtime_payload() -> dict[str, object]:
    linkage_row = {"bytes": 1, "path": "/runtime/lib", "sha256": "a" * 64}
    return {
        "byteorder": "little",
        "cache_tag": "cpython-311",
        "dependency_lock_sha256": "b" * 64,
        "distributions": (("zstandard", "0.25.0"),),
        "environment": (("TOKENIZERS_PARALLELISM", "false"),),
        "filesystem_encoding": "utf-8",
        "implementation": "CPython",
        "installed_distribution_inventory": {
            "bootstrap_distributions": [],
            "distributions": [],
            "files": [
                {
                    "bytes": 1,
                    "owners": ["zstandard"],
                    "relative_path": "site-packages/zstandard.py",
                    "sha256": "c" * 64,
                }
            ],
            "installation_prefix": "/runtime",
            "inventory_identity_sha256": "d" * 64,
            "schema": "fixture_inventory",
            "site_roots": ["site-packages"],
        },
        "locale": "C.UTF-8",
        "machine": "x86_64",
        "maxunicode": 1114111,
        "platform_release": "host-kernel-one",
        "platform_system": "Linux",
        "preferred_encoding": "UTF-8",
        "python_executable_sha256": "e" * 64,
        "runtime_linkage": {
            "executable": dict(linkage_row),
            "libpython_library": dict(linkage_row),
            "linkage_identity_sha256": "f" * 64,
            "schema": "fixture_linkage",
            "sqlite_extension": dict(linkage_row),
            "sqlite_library": dict(linkage_row),
        },
        "runtime_versions": {"python_version": "3.11.9"},
    }


def test_parser_runtime_identity_survives_host_and_prefix_replacement() -> None:
    first = _runtime_payload()
    replacement = copy.deepcopy(first)
    replacement["platform_release"] = "host-kernel-two"
    inventory = replacement["installed_distribution_inventory"]
    assert isinstance(inventory, dict)
    inventory["installation_prefix"] = "/replacement/runtime"
    inventory["inventory_identity_sha256"] = "1" * 64
    linkage = replacement["runtime_linkage"]
    assert isinstance(linkage, dict)
    linkage["linkage_identity_sha256"] = "2" * 64
    for name in (
        "executable",
        "libpython_library",
        "sqlite_extension",
        "sqlite_library",
    ):
        row = linkage[name]
        assert isinstance(row, dict)
        row["path"] = f"/replacement/runtime/{name}"
    assert parsed_asset_runtime_identity_v1(first) == (
        parsed_asset_runtime_identity_v1(replacement)
    )


def test_parser_runtime_identity_changes_with_linked_runtime_bytes() -> None:
    first = _runtime_payload()
    changed = copy.deepcopy(first)
    linkage = changed["runtime_linkage"]
    assert isinstance(linkage, dict)
    executable = linkage["executable"]
    assert isinstance(executable, dict)
    executable["sha256"] = "9" * 64
    assert parsed_asset_runtime_identity_v1(first) != (
        parsed_asset_runtime_identity_v1(changed)
    )


def test_parser_runtime_identity_rejects_unruled_schema_growth() -> None:
    changed = _runtime_payload()
    changed["future_parser_relevant_field"] = "unruled"
    with pytest.raises(
        ParsedAssetRecoveryError,
        match="explicit projection",
    ):
        parsed_asset_runtime_identity_v1(changed)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _context(
    *,
    runtime: str = "1",
    run_id: str = "replay-a",
    storage: str = "0",
) -> ParsedAssetRecoveryContextV1:
    return ParsedAssetRecoveryContextV1(
        run_id=run_id,
        durable_marker_physical_sha256=storage * 64,
        runtime_identity_sha256=runtime * 64,
        code_identity_sha256="2" * 64,
        input_identity_sha256="3" * 64,
    )


def _stackedu_row(*, score: int) -> dict[str, object]:
    return {
        "added": "2025-01-01",
        "created": "2024-01-01",
        "id": "same-native-id",
        "metadata": {
            "int_score": score,
            "path": "src/example.py",
            "score": 3.5,
            "uri": "https://example.test/repo",
        },
        "source": "stackedu",
        "text": "retained StackEdu text",
    }


def _verified_asset(
    tmp_path: Path,
    *,
    source_family: str,
    payload: bytes,
    v4_authority: bool = False,
) -> tuple[VerifiedLocalCacheAssetV3, Path]:
    route = next(
        route
        for route in load_exact_source_routes_v3()
        if route.source_family == source_family
    )
    locator = {
        "dolma_web": "data/common_crawl-unit-0019/00000.jsonl.zst",
        "stackedu": "data/stack_edu-Java/shard_00000000.jsonl.zst",
    }[source_family]
    relative = f"fixture/{source_family}.jsonl.zst"
    cache_root = tmp_path / f"cache-{source_family}"
    asset_path = cache_root.joinpath(*PurePosixPath(relative).parts)
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(payload)
    asset_fields = {
        "source_family": source_family,
        "repository": route.repository,
        "config": route.config,
        "revision": route.revision,
        "split": route.split,
        "asset_locator": locator,
        "relative_path": relative,
        "bytes": len(payload),
        "sha256": _sha(payload),
    }
    if v4_authority:
        expected = SourceCacheAssetV4(
            **asset_fields,
            effective_route_receipt_sha256="8" * 64,
            execution_binding_sha256="9" * 64,
        )
    else:
        expected = SourceCacheAssetV3(**asset_fields)
    return (
        VerifiedLocalCacheAssetV3(
            expected=expected,
            observed_bytes=len(payload),
            observed_sha256=_sha(payload),
        ),
        cache_root,
    )


def _stackedu_events(tmp_path: Path, *, v4_authority: bool = False):
    logical = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in (_stackedu_row(score=2), _stackedu_row(score=4))
    )
    payload = zstandard.ZstdCompressor(level=3).compress(logical)
    asset, cache_root = _verified_asset(
        tmp_path,
        source_family="stackedu",
        payload=payload,
        v4_authority=v4_authority,
    )
    binding = resolve_production_parser_binding_v3(asset)
    events = tuple(
        iter_source_asset_events_v3(asset, cache_root, binding=binding)
    )
    return asset, binding, events


def test_parsed_asset_cache_round_trip_reconstructs_events_and_projections(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    assert [event.disposition for event in events] == [DROP_QUALITY_LT3, RETAIN]
    published = write_parsed_asset_segment_v1(
        tmp_path / "parsed-cache",
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=3,
        first_event_ordinal=17,
        events=events,
    )

    recovered = tuple(
        iter_parsed_asset_segment_v1(
            tmp_path / "parsed-cache",
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=3,
            expected_first_event_ordinal=17,
        )
    )
    assert published.receipt.event_count == 2
    assert published.receipt.retained_record_count == 1
    assert published.receipt.observation_count == 1
    assert [row.event for row in recovered] == list(events)
    assert [row.event_ordinal for row in recovered] == [17, 18]
    assert recovered[0].ledger_payload == {
        "asset_order_ordinal": 3,
        "disposition": DROP_QUALITY_LT3,
        "event_ordinal": 17,
        "event_sha256": events[0].event_sha256,
        "source_asset_identity_sha256": asset.expected.asset_identity_sha256,
        "source_family": "stackedu",
        "source_record_ordinal": 0,
    }
    assert recovered[0].sqlite_insert_fields is None
    insert = recovered[1].sqlite_insert_fields
    assert insert is not None
    assert insert["source"] == "stackedu"
    assert insert["asset_order_ordinal"] == 3
    assert insert["asset_record_ordinal"] == 1
    assert insert["text_bytes"] == b"retained StackEdu text"
    assert insert["retained_bytes"] == len(b"retained StackEdu text")
    assert insert["int_score"] == 4


def test_parsed_asset_cache_round_trip_preserves_v4_asset_authority(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path, v4_authority=True)
    assert type(asset.expected) is SourceCacheAssetV4
    root = tmp_path / "parsed-cache"

    published = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=0,
        first_event_ordinal=0,
        events=events,
    )
    recovered = tuple(
        iter_parsed_asset_segment_v1(
            root,
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=0,
            expected_first_event_ordinal=0,
        )
    )

    retained = next(row.event.record for row in recovered if row.event.record is not None)
    recovered_asset = retained.canonical_record.asset
    assert published.receipt.retained_record_count == 1
    assert type(recovered_asset) is SourceCacheAssetV4
    assert recovered_asset == asset.expected
    assert recovered_asset.effective_route_receipt_sha256 == "8" * 64
    assert recovered_asset.execution_binding_sha256 == "9" * 64


def test_parsed_asset_cache_reopen_hash_and_context_fail_closed(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    published = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=0,
        first_event_ordinal=0,
        events=events,
    )
    with pytest.raises(ParsedAssetRecoveryError, match="unavailable"):
        tuple(
            iter_parsed_asset_segment_v1(
                root,
                context=_context(run_id="replay-b"),
                verified_asset=asset,
                parser_binding=binding,
                asset_order_ordinal=0,
                expected_first_event_ordinal=0,
            )
        )
    with pytest.raises(ParsedAssetRecoveryError, match="unavailable"):
        tuple(
            iter_parsed_asset_segment_v1(
                root,
                context=_context(storage="4"),
                verified_asset=asset,
                parser_binding=binding,
                asset_order_ordinal=0,
                expected_first_event_ordinal=0,
            )
        )

    with published.segment_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ParsedAssetRecoveryError, match="physical bytes"):
        tuple(
            iter_parsed_asset_segment_v1(
                root,
                context=_context(),
                verified_asset=asset,
                parser_binding=binding,
                asset_order_ordinal=0,
                expected_first_event_ordinal=0,
            )
        )


def test_parsed_asset_cache_never_overwrites_committed_or_partial_state(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    published = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=1,
        first_event_ordinal=5,
        events=events,
    )
    before = (
        published.segment_path.read_bytes(),
        published.receipt_path.read_bytes(),
    )
    with pytest.raises(ParsedAssetRecoveryError, match="refusing to overwrite"):
        write_parsed_asset_segment_v1(
            root,
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=1,
            first_event_ordinal=5,
            events=events,
        )
    assert before == (
        published.segment_path.read_bytes(),
        published.receipt_path.read_bytes(),
    )


def test_parsed_asset_cache_adopts_only_an_exact_receiptless_orphan(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    first = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=2,
        first_event_ordinal=9,
        events=events,
    )
    segment_before = first.segment_path.read_bytes()
    first.receipt_path.unlink()
    with pytest.raises(ParsedAssetRecoveryError, match="orphan requires opt-in"):
        probe_parsed_asset_segment_v1(
            root,
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=2,
            expected_first_event_ordinal=9,
        )
    assert probe_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=2,
        expected_first_event_ordinal=9,
        allow_receiptless_orphan=True,
    ) == "MISS"
    stale_receipt_partial = first.receipt_path.with_name(
        first.receipt_path.name + ".partial"
    )
    stale_receipt_partial.write_bytes(b"interrupted receipt")

    adopted = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=2,
        first_event_ordinal=9,
        events=events,
    )
    assert adopted.segment_path.read_bytes() == segment_before
    assert adopted.receipt == first.receipt
    assert adopted.receipt_path.exists()
    assert stale_receipt_partial.exists()


def test_parsed_asset_cache_ignores_uncommitted_unique_partial_on_retry(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    context = _context()
    parent = root / context.identity_sha256[:24] / "stackedu"
    parent.mkdir(parents=True)
    stale_partial = parent / f".data-{'f' * 32}.partial"
    stale_partial.write_bytes(b"uncommitted partial bytes")
    assert probe_parsed_asset_segment_v1(
        root,
        context=context,
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=4,
        expected_first_event_ordinal=11,
    ) == "MISS"

    published = write_parsed_asset_segment_v1(
        root,
        context=context,
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=4,
        first_event_ordinal=11,
        events=events,
    )
    assert stale_partial.exists()
    assert published.receipt_path.exists()
    assert probe_parsed_asset_segment_v1(
        root,
        context=context,
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=4,
        expected_first_event_ordinal=11,
    ) == "HIT"


def test_parsed_asset_cache_probe_rejects_receipt_without_data(
    tmp_path: Path,
) -> None:
    asset, binding, events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    published = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=5,
        first_event_ordinal=20,
        events=events,
    )
    published.segment_path.unlink()
    with pytest.raises(ParsedAssetRecoveryError, match="receipt exists without"):
        probe_parsed_asset_segment_v1(
            root,
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=5,
            expected_first_event_ordinal=20,
        )


def test_parsed_asset_cache_allows_completed_zero_event_asset(
    tmp_path: Path,
) -> None:
    payload = zstandard.ZstdCompressor(level=3).compress(b"")
    asset, cache_root = _verified_asset(
        tmp_path,
        source_family="dolma_web",
        payload=payload,
    )
    binding = resolve_production_parser_binding_v3(asset)
    events = tuple(
        iter_source_asset_events_v3(asset, cache_root, binding=binding)
    )
    assert events == ()
    root = tmp_path / "parsed-cache"
    published = write_parsed_asset_segment_v1(
        root,
        context=_context(),
        verified_asset=asset,
        parser_binding=binding,
        asset_order_ordinal=0,
        first_event_ordinal=0,
        events=events,
    )
    assert published.receipt.event_count == 0
    assert published.receipt.logical_jsonl_bytes == 0
    assert published.receipt.observation_count == 0
    assert tuple(
        iter_parsed_asset_segment_v1(
            root,
            context=_context(),
            verified_asset=asset,
            parser_binding=binding,
            asset_order_ordinal=0,
            expected_first_event_ordinal=0,
        )
    ) == ()


def test_legacy_progress_receipt_is_not_a_parsed_asset_cache(
    tmp_path: Path,
) -> None:
    asset, binding, unused_events = _stackedu_events(tmp_path)
    root = tmp_path / "parsed-cache"
    context = _context()
    name = (
        f"000000-{asset.expected.asset_identity_sha256}.parsed.jsonl.zst"
    )
    receipt_path = (
        root
        / context.identity_sha256[:24]
        / "stackedu"
        / f"{name}.receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "weft1_source_parse_checkpoint_v3",
                "progress_semantics": "PARSE_PROGRESS_ONLY_NO_RESUME",
                "resume_authorized": False,
            }
        )
        + b"\n"
    )
    with pytest.raises(ParsedAssetRecoveryError, match="fields are not exact"):
        tuple(
            iter_parsed_asset_segment_v1(
                root,
                context=context,
                verified_asset=asset,
                parser_binding=binding,
                asset_order_ordinal=0,
                expected_first_event_ordinal=0,
            )
        )
