from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

import training.weft1_corpus_materialize_a2 as materializer
import training.weft1_corpus_parsed_asset_cache_v1 as parsed_cache
import training.weft1_corpus_replay_a3 as replay_v4
from training.weft1_corpus_parsed_asset_cache_v1 import (
    CURRENT_CONTEXT_RESOLUTION_V1,
    PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1,
    PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1,
    PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1,
    READ_ONLY_PREDECESSOR_RESOLUTION_V1,
    ParsedAssetCompatibilityPolicyV1,
    ParsedAssetRecoveryContextV1,
    load_parsed_asset_compatibility_policy_v1,
    load_parsed_asset_composite_bridge_v1,
    parsed_asset_composite_bridge_path_v1,
    validate_parsed_asset_composite_bridge_policy_v1,
)
from training.weft1_corpus_sources_a2 import (
    SourceCacheAssetV3,
    VerifiedLocalCacheAssetV3,
    load_exact_source_routes_v3,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import canonical_json_bytes
from training.weft1_corpus_source_io_a2 import (
    PRODUCTION_PARSER_BINDINGS_V3,
    STACKEDU_PYTHON_PARSER_BINDING_V3,
)


def _sha(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _context(
    *,
    run_id: str = "production-v4-replay-a",
    code_identity_sha256: str = "2" * 64,
) -> ParsedAssetRecoveryContextV1:
    return ParsedAssetRecoveryContextV1(
        run_id=run_id,
        durable_marker_physical_sha256="0" * 64,
        runtime_identity_sha256="1" * 64,
        code_identity_sha256=code_identity_sha256,
        input_identity_sha256="3" * 64,
    )


def _dolma_assets(root: Path) -> tuple[VerifiedLocalCacheAssetV3, ...]:
    route = next(
        item
        for item in load_exact_source_routes_v3()
        if item.source_family == "dolma_web"
    )
    root.mkdir()
    assets = []
    rows_by_asset = (
        ({"id": "dolma-0", "metadata": {}, "text": "first retained text"},),
        (
            {"id": "dolma-empty", "metadata": {}, "text": ""},
            {"id": "dolma-1", "metadata": {}, "text": "second retained text"},
        ),
    )
    for ordinal, rows in enumerate(rows_by_asset):
        logical = b"".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
        payload = zstandard.ZstdCompressor(level=3).compress(logical)
        relative = f"fixture/dolma-{ordinal:05d}.jsonl.zst"
        path = root.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected = SourceCacheAssetV3(
            source_family="dolma_web",
            repository=route.repository,
            config=route.config,
            revision=route.revision,
            split=route.split,
            asset_locator=(
                f"data/common_crawl-unit-0019/{ordinal:05d}.jsonl.zst"
            ),
            relative_path=relative,
            bytes=len(payload),
            sha256=_sha(payload),
        )
        assets.append(
            VerifiedLocalCacheAssetV3(
                expected=expected,
                observed_bytes=len(payload),
                observed_sha256=_sha(payload),
            )
        )
    return tuple(assets)


def _fineweb_asset(root: Path) -> VerifiedLocalCacheAssetV3:
    route = next(
        item
        for item in load_exact_source_routes_v3()
        if item.source_family == "fineweb_edu"
    )
    scratch = root.parent / "fineweb.parquet"
    table = pa.table(
        {
            "text": pa.array(["fresh FineWeb text"], type=pa.string()),
            "id": pa.array(["fineweb-0"], type=pa.string()),
            "dump": pa.array(["CC-MAIN-2018-30"], type=pa.string()),
            "url": pa.array(["https://example.test"], type=pa.string()),
            "file_path": pa.array(["crawl/example"], type=pa.string()),
            "language": pa.array(["en"], type=pa.string()),
            "language_score": pa.array([0.99], type=pa.float64()),
            "token_count": pa.array([4], type=pa.int64()),
            "score": pa.array([3.7], type=pa.float64()),
            "int_score": pa.array([3], type=pa.int64()),
        }
    )
    pq.write_table(table, scratch, compression="NONE")
    payload = scratch.read_bytes()
    relative = "fixture/fineweb.parquet"
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    expected = SourceCacheAssetV3(
        source_family="fineweb_edu",
        repository=route.repository,
        config=route.config,
        revision=route.revision,
        split=route.split,
        asset_locator="data/CC-MAIN-2018-30/train-00000-of-00001.parquet",
        relative_path=relative,
        bytes=len(payload),
        sha256=_sha(payload),
    )
    return VerifiedLocalCacheAssetV3(
        expected=expected,
        observed_bytes=len(payload),
        observed_sha256=_sha(payload),
    )


def _fixture_fineweb_binding(
    source_cache: Path,
    asset: VerifiedLocalCacheAssetV3,
) -> object:
    path = source_cache.joinpath(
        *PurePosixPath(asset.expected.relative_path).parts
    )
    return replace(
        PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"],
        authority="FIXTURE_ONLY",
        authority_sha256=_sha("fixture:composite-fineweb"),
        declared_parquet_schema_ipc_sha256=_sha(
            pq.ParquetFile(path).schema_arrow.serialize().to_pybytes()
        ),
    )


def _compatibility_policy(
    *,
    current: ParsedAssetRecoveryContextV1,
    predecessor: ParsedAssetRecoveryContextV1,
    predecessor_count: int,
    current_count: int,
) -> ParsedAssetCompatibilityPolicyV1:
    bindings = {
        (source, binding.binding_sha256)
        for source, binding in PRODUCTION_PARSER_BINDINGS_V3.items()
        if source != "fineweb_edu"
    }
    bindings.add(("stackedu", STACKEDU_PYTHON_PARSER_BINDING_V3.binding_sha256))
    # Small fixtures exercise the mechanics without pretending to be the
    # governed 394/3 incident.  Temporarily substitute an explicit fixture
    # scope only while constructing the typed value; artifact loading under
    # production constants still rejects it.
    production_scope = (
        parsed_cache.PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1,
    )
    (
        parsed_cache.PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1,
        parsed_cache.PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1,
    ) = (
        current.run_id,
        predecessor.code_identity_sha256,
        predecessor_count,
        current_count,
    )
    try:
        return ParsedAssetCompatibilityPolicyV1(
            schema=PARSED_ASSET_COMPATIBILITY_POLICY_SCHEMA_V1,
            authority_sha256=PARSED_ASSET_INCIDENT_AUTHORITY_PHYSICAL_SHA256_V1,
            eligible_run_id=current.run_id,
            predecessor_code_identity_sha256=predecessor.code_identity_sha256,
            successor_code_identity_sha256=current.code_identity_sha256,
            compatible_parser_bindings=tuple(sorted(bindings)),
            excluded_source_families=("fineweb_edu",),
            expected_predecessor_asset_count=predecessor_count,
            expected_current_asset_count=current_count,
        )
    finally:
        (
            parsed_cache.PARSED_ASSET_INCIDENT_ELIGIBLE_RUN_ID_V1,
            parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_CODE_IDENTITY_SHA256_V1,
            parsed_cache.PARSED_ASSET_INCIDENT_PREDECESSOR_ASSET_COUNT_V1,
            parsed_cache.PARSED_ASSET_INCIDENT_CURRENT_ASSET_COUNT_V1,
        ) = production_scope


def _instance(
    root: Path,
    *,
    assets: tuple[VerifiedLocalCacheAssetV3, ...],
    source_cache_root: Path,
    parsed_cache_root: Path,
) -> object:
    output_root = root / "output"
    work_root = root / "work"
    output_root.mkdir(parents=True)
    work_root.mkdir()
    instance = object.__new__(materializer._Materializer)
    instance.inputs = SimpleNamespace(
        mode=materializer.PRODUCTION_MODE,
        verified_cache=SimpleNamespace(assets=assets),
        cache_root=source_cache_root,
        source_cache_download_receipt=object(),
        source_identity_sha256=_sha("transport"),
    )
    instance.output_root = output_root
    instance.work_root = work_root
    instance.parsed_asset_cache_root = parsed_cache_root
    instance.parsed_asset_recovery_context = _context()
    instance.source_parse_drop_counts = {
        source: {"empty_text": 0, "invalid_utf8": 0, "quality_lt3": 0}
        for source in SOURCE_FAMILIES
    }
    instance.invalid_utf8_by_source = Counter(
        {source: 0 for source in SOURCE_FAMILIES}
    )
    instance.source_parse_receipts = {}
    instance._production_source_db = None
    return instance


def _snapshot(instance: object) -> dict[str, object]:
    connection = instance._production_source_db
    assert connection is not None
    rows = tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT source, stable_source_record_id, "
            "source_asset_identity_sha256, asset_order_ordinal, "
            "asset_record_ordinal, text, retained_bytes, int_score "
            "FROM parsed_records ORDER BY source, stable_source_record_id"
        )
    )
    parse_root = instance.output_root / "source-parse"
    return {
        "invalid_utf8": tuple(instance.invalid_utf8_by_source.items()),
        "materialized_source_identity_sha256": (
            instance.materialized_source_identity_sha256
        ),
        "parse_tree": tuple(
            (path.relative_to(parse_root).as_posix(), path.read_bytes())
            for path in sorted(parse_root.rglob("*"))
            if path.is_file()
        ),
        "receipts": tuple(
            (source, tuple(sorted(instance.source_parse_receipts[source].items())))
            for source in SOURCE_FAMILIES
        ),
        "rows": rows,
        "source_parse_drop_counts": tuple(
            (
                source,
                tuple(sorted(instance.source_parse_drop_counts[source].items())),
            )
            for source in SOURCE_FAMILIES
        ),
    }


def _run(instance: object) -> dict[str, object]:
    instance._prepare_production_sources()
    try:
        return _snapshot(instance)
    finally:
        assert instance._production_source_db is not None
        instance._production_source_db.close()


def test_second_fresh_materialization_uses_only_validated_cache_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    assets = _dolma_assets(source_cache)
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()

    first = _run(
        _instance(
            tmp_path / "first",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=parsed_cache,
        )
    )
    assert len(tuple(parsed_cache.rglob("*.receipt.json"))) == 2

    def forbidden_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cache HIT opened the upstream parser")

    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        forbidden_parser,
    )
    second = _run(
        _instance(
            tmp_path / "second",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=parsed_cache,
        )
    )
    assert second == first


def test_mixed_cache_hit_and_miss_parses_only_the_missing_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    assets = _dolma_assets(source_cache)
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    baseline = _run(
        _instance(
            tmp_path / "baseline",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=parsed_cache,
        )
    )

    missing_segment = next(parsed_cache.rglob("000001-*.parsed.jsonl.zst"))
    missing_receipt = missing_segment.with_name(
        missing_segment.name + ".receipt.json"
    )
    missing_receipt.unlink()
    missing_segment.unlink()

    original_parser = materializer.iter_source_asset_events_v3
    parser_calls: list[str] = []

    def counting_parser(
        asset: VerifiedLocalCacheAssetV3,
        root: Path,
        *,
        binding: object,
    ) -> object:
        parser_calls.append(asset.expected.asset_identity_sha256)
        return original_parser(asset, root, binding=binding)

    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        counting_parser,
    )
    recovered = _run(
        _instance(
            tmp_path / "mixed",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=parsed_cache,
        )
    )
    assert parser_calls == [assets[1].expected.asset_identity_sha256]
    assert recovered == baseline


def test_prefill_skips_hit_payloads_before_parsing_only_the_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    assets = _dolma_assets(source_cache)
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    _run(
        _instance(
            tmp_path / "baseline",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=parsed_cache,
        )
    )
    missing_segment = next(parsed_cache.rglob("000001-*.parsed.jsonl.zst"))
    missing_segment.with_name(missing_segment.name + ".receipt.json").unlink()
    missing_segment.unlink()

    def forbidden_cache_payload(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prefill read a completed cache payload")

    original_parser = materializer.iter_source_asset_events_v3
    parser_calls: list[str] = []

    def counting_parser(
        asset: VerifiedLocalCacheAssetV3,
        root: Path,
        *,
        binding: object,
    ) -> object:
        parser_calls.append(asset.expected.asset_identity_sha256)
        return original_parser(asset, root, binding=binding)

    monkeypatch.setattr(
        materializer,
        "iter_parsed_asset_segment_v1",
        forbidden_cache_payload,
    )
    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        counting_parser,
    )
    instance = _instance(
        tmp_path / "prefill",
        assets=assets,
        source_cache_root=source_cache,
        parsed_cache_root=parsed_cache,
    )
    instance._prefill_production_parsed_asset_cache()
    assert parser_calls == [assets[1].expected.asset_identity_sha256]
    assert len(tuple(parsed_cache.rglob("*.receipt.json"))) == 2


def test_composite_bridge_reuses_only_donor_and_parses_fineweb_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    dolma = _dolma_assets(source_cache)[:1]
    fineweb = _fineweb_asset(source_cache)
    predecessor_cache = tmp_path / "predecessor-cache"
    current_cache = tmp_path / "current-cache"
    baseline_cache = tmp_path / "baseline-cache"
    predecessor_cache.mkdir()
    current_cache.mkdir()
    baseline_cache.mkdir()
    predecessor_context = _context(code_identity_sha256="4" * 64)
    current_context = _context()
    donor_inputs = _instance(
        tmp_path / "donor-inputs",
        assets=dolma,
        source_cache_root=source_cache,
        parsed_cache_root=predecessor_cache,
    ).inputs
    materializer.prefill_production_parsed_asset_cache_v1(
        inputs=donor_inputs,
        parsed_asset_cache_root=predecessor_cache,
        parsed_asset_recovery_context=predecessor_context,
        allow_writes=True,
    )
    assets = (*dolma, fineweb)
    inputs = _instance(
        tmp_path / "bridge-inputs",
        assets=assets,
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    ).inputs
    policy = _compatibility_policy(
        current=current_context,
        predecessor=predecessor_context,
        predecessor_count=1,
        current_count=1,
    )
    parser_calls: list[str] = []
    original_parser = materializer.iter_source_asset_events_v3
    original_resolver = materializer.resolve_production_parser_binding_v3
    fixture_fineweb_binding = _fixture_fineweb_binding(source_cache, fineweb)

    def fixture_resolver(asset: VerifiedLocalCacheAssetV3) -> object:
        if asset.expected.source_family == "fineweb_edu":
            return fixture_fineweb_binding
        return original_resolver(asset)

    def counting_parser(
        asset: VerifiedLocalCacheAssetV3,
        root: Path,
        *,
        binding: object,
    ) -> object:
        parser_calls.append(asset.expected.source_family)
        return original_parser(
            asset,
            root,
            binding=binding,
            allow_fixture_binding=(
                getattr(binding, "authority", None) == "FIXTURE_ONLY"
            ),
        )

    monkeypatch.setattr(
        materializer,
        "resolve_production_parser_binding_v3",
        fixture_resolver,
    )
    monkeypatch.setattr(materializer, "iter_source_asset_events_v3", counting_parser)
    bridge = materializer.prefill_production_parsed_asset_cache_v1(
        inputs=inputs,
        parsed_asset_cache_root=current_cache,
        parsed_asset_recovery_context=current_context,
        allow_writes=True,
        predecessor_cache_root=predecessor_cache,
        predecessor_recovery_context=predecessor_context,
        compatibility_policy=policy,
    )
    assert parser_calls == ["fineweb_edu"]
    assert bridge.predecessor_asset_count == 1
    assert bridge.current_asset_count == 1
    assert tuple(row.resolution for row in bridge.rows) == (
        READ_ONLY_PREDECESSOR_RESOLUTION_V1,
        CURRENT_CONTEXT_RESOLUTION_V1,
    )
    loaded, bridge_bytes, bridge_physical_sha256 = (
        load_parsed_asset_composite_bridge_v1(current_cache)
    )
    assert loaded == bridge
    assert bridge_bytes > 0
    assert len(bridge_physical_sha256) == 64
    assert len(tuple(current_cache.rglob("*.parsed.jsonl.zst"))) == 1
    assert len(tuple(predecessor_cache.rglob("*.parsed.jsonl.zst"))) == 1

    baseline = _run(
        _instance(
            tmp_path / "baseline",
            assets=assets,
            source_cache_root=source_cache,
            parsed_cache_root=baseline_cache,
        )
    )
    bridged_instance = _instance(
        tmp_path / "bridged",
        assets=assets,
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    )
    bridged_instance.parsed_asset_cache_read_only = True
    bridged_instance.predecessor_cache_root = predecessor_cache
    bridged_instance.predecessor_recovery_context = predecessor_context
    bridged_instance.compatibility_policy = policy

    def forbidden_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validated composite materialization reopened a parser")

    monkeypatch.setattr(materializer, "iter_source_asset_events_v3", forbidden_parser)
    assert _run(bridged_instance) == baseline

    donor_segment = next(predecessor_cache.rglob("*.parsed.jsonl.zst"))
    donor_segment.write_bytes(donor_segment.read_bytes() + b"post-materialization-drift")
    with pytest.raises(Exception, match="bridge failed"):
        replay_v4._load_expected_parsed_asset_bridge_v4(
            cache_root=current_cache,
            current_context=current_context,
            predecessor_context=predecessor_context,
            compatibility_policy=policy,
            validate_segment_transport=True,
        )


def test_donor_mutation_between_inventory_and_publish_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    dolma = _dolma_assets(source_cache)[:1]
    fineweb = _fineweb_asset(source_cache)
    predecessor_cache = tmp_path / "predecessor-cache"
    current_cache = tmp_path / "current-cache"
    predecessor_cache.mkdir()
    current_cache.mkdir()
    predecessor_context = _context(code_identity_sha256="4" * 64)
    current_context = _context()
    donor_inputs = _instance(
        tmp_path / "donor-inputs",
        assets=dolma,
        source_cache_root=source_cache,
        parsed_cache_root=predecessor_cache,
    ).inputs
    materializer.prefill_production_parsed_asset_cache_v1(
        inputs=donor_inputs,
        parsed_asset_cache_root=predecessor_cache,
        parsed_asset_recovery_context=predecessor_context,
        allow_writes=True,
    )
    donor_segment = next(predecessor_cache.rglob("*.parsed.jsonl.zst"))
    inputs = _instance(
        tmp_path / "bridge-inputs",
        assets=(*dolma, fineweb),
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    ).inputs
    policy = _compatibility_policy(
        current=current_context,
        predecessor=predecessor_context,
        predecessor_count=1,
        current_count=1,
    )
    original_resolver = materializer.resolve_production_parser_binding_v3
    original_parser = materializer.iter_source_asset_events_v3
    fixture_binding = _fixture_fineweb_binding(source_cache, fineweb)

    def fixture_resolver(asset: VerifiedLocalCacheAssetV3) -> object:
        return (
            fixture_binding
            if asset.expected.source_family == "fineweb_edu"
            else original_resolver(asset)
        )

    mutated = False

    def mutating_parser(
        asset: VerifiedLocalCacheAssetV3,
        root: Path,
        *,
        binding: object,
    ) -> object:
        nonlocal mutated
        if asset.expected.source_family == "fineweb_edu" and not mutated:
            donor_segment.write_bytes(donor_segment.read_bytes() + b"drift")
            mutated = True
        return original_parser(
            asset,
            root,
            binding=binding,
            allow_fixture_binding=(
                getattr(binding, "authority", None) == "FIXTURE_ONLY"
            ),
        )

    monkeypatch.setattr(
        materializer,
        "resolve_production_parser_binding_v3",
        fixture_resolver,
    )
    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        mutating_parser,
    )
    with pytest.raises(
        Exception,
        match="physical bytes drifted|segment type or size differs",
    ):
        materializer.prefill_production_parsed_asset_cache_v1(
            inputs=inputs,
            parsed_asset_cache_root=current_cache,
            parsed_asset_recovery_context=current_context,
            allow_writes=True,
            predecessor_cache_root=predecessor_cache,
            predecessor_recovery_context=predecessor_context,
            compatibility_policy=policy,
        )
    assert mutated
    assert not parsed_asset_composite_bridge_path_v1(current_cache).exists()


def test_donor_always_rejects_existing_current_segment_before_fineweb_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    dolma = _dolma_assets(source_cache)[:1]
    fineweb = _fineweb_asset(source_cache)
    predecessor_cache = tmp_path / "predecessor-cache"
    current_cache = tmp_path / "current-cache"
    predecessor_cache.mkdir()
    current_cache.mkdir()
    predecessor_context = _context(code_identity_sha256="4" * 64)
    current_context = _context()
    for cache, context, label in (
        (predecessor_cache, predecessor_context, "donor"),
        (current_cache, current_context, "current"),
    ):
        inputs = _instance(
            tmp_path / label,
            assets=dolma,
            source_cache_root=source_cache,
            parsed_cache_root=cache,
        ).inputs
        materializer.prefill_production_parsed_asset_cache_v1(
            inputs=inputs,
            parsed_asset_cache_root=cache,
            parsed_asset_recovery_context=context,
            allow_writes=True,
        )
    policy = _compatibility_policy(
        current=current_context,
        predecessor=predecessor_context,
        predecessor_count=1,
        current_count=1,
    )
    inputs = _instance(
        tmp_path / "ambiguous",
        assets=(*dolma, fineweb),
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    ).inputs

    def forbidden_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("donor/current ambiguity must precede FineWeb parsing")

    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        forbidden_parser,
    )
    with pytest.raises(
        materializer.CorpusMaterializationError,
        match="conflicts with donor-always authority",
    ):
        materializer.prefill_production_parsed_asset_cache_v1(
            inputs=inputs,
            parsed_asset_cache_root=current_cache,
            parsed_asset_recovery_context=current_context,
            allow_writes=True,
            predecessor_cache_root=predecessor_cache,
            predecessor_recovery_context=predecessor_context,
            compatibility_policy=policy,
        )


def test_bridge_policy_rejects_swapped_resolution_with_preserved_counts(
    tmp_path: Path,
) -> None:
    current = _context()
    predecessor = _context(code_identity_sha256="4" * 64)
    policy = _compatibility_policy(
        current=current,
        predecessor=predecessor,
        predecessor_count=1,
        current_count=1,
    )
    donor = parsed_cache.ParsedAssetCompositeBridgeRowV1(
        source_family="dolma_web",
        asset_order_ordinal=0,
        source_asset_identity_sha256="5" * 64,
        source_asset_sha256="6" * 64,
        parser_binding_sha256=(
            PRODUCTION_PARSER_BINDINGS_V3["dolma_web"].binding_sha256
        ),
        first_event_ordinal=0,
        next_event_ordinal=1,
        resolution=READ_ONLY_PREDECESSOR_RESOLUTION_V1,
        selected_context_identity_sha256=predecessor.identity_sha256,
        selected_code_identity_sha256=predecessor.code_identity_sha256,
        segment_relative_path="donor/row.parsed.jsonl.zst",
        segment_physical_bytes=1,
        segment_physical_sha256="7" * 64,
        segment_receipt_sha256="8" * 64,
        segment_receipt_physical_bytes=1,
        segment_receipt_physical_sha256="9" * 64,
    )
    current_row = replace(
        donor,
        source_family="fineweb_edu",
        asset_order_ordinal=0,
        source_asset_identity_sha256="a" * 64,
        source_asset_sha256="b" * 64,
        parser_binding_sha256=(
            PRODUCTION_PARSER_BINDINGS_V3["fineweb_edu"].binding_sha256
        ),
        resolution=CURRENT_CONTEXT_RESOLUTION_V1,
        selected_context_identity_sha256=current.identity_sha256,
        selected_code_identity_sha256=current.code_identity_sha256,
        segment_relative_path="current/row.parsed.jsonl.zst",
    )
    swapped_donor = replace(
        donor,
        resolution=CURRENT_CONTEXT_RESOLUTION_V1,
        selected_context_identity_sha256=current.identity_sha256,
        selected_code_identity_sha256=current.code_identity_sha256,
    )
    swapped_current = replace(
        current_row,
        resolution=READ_ONLY_PREDECESSOR_RESOLUTION_V1,
        selected_context_identity_sha256=predecessor.identity_sha256,
        selected_code_identity_sha256=predecessor.code_identity_sha256,
    )
    bridge = parsed_cache.ParsedAssetCompositeBridgeV1(
        schema=parsed_cache.PARSED_ASSET_COMPOSITE_BRIDGE_SCHEMA_V1,
        recovery_domain=parsed_cache.PARSED_ASSET_RECOVERY_DOMAIN_V1,
        current_context=current,
        predecessor_context=predecessor,
        compatibility_policy_sha256=policy.identity_sha256,
        rows=(swapped_donor, swapped_current),
        current_asset_count=1,
        predecessor_asset_count=1,
    )
    with pytest.raises(
        Exception,
        match="unauthorized predecessor row|current code outside an excluded family",
    ):
        validate_parsed_asset_composite_bridge_policy_v1(bridge, policy)


def test_missing_registered_donor_fails_before_any_current_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    dolma = _dolma_assets(source_cache)[:1]
    fineweb = _fineweb_asset(source_cache)
    predecessor_cache = tmp_path / "missing-predecessor"
    current_cache = tmp_path / "current-cache"
    predecessor_cache.mkdir()
    current_cache.mkdir()
    predecessor_context = _context(code_identity_sha256="4" * 64)
    current_context = _context()
    inputs = _instance(
        tmp_path / "inputs",
        assets=(*dolma, fineweb),
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    ).inputs
    policy = _compatibility_policy(
        current=current_context,
        predecessor=predecessor_context,
        predecessor_count=1,
        current_count=1,
    )

    def forbidden_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("missing donor must fail before a current parser opens")

    monkeypatch.setattr(materializer, "iter_source_asset_events_v3", forbidden_parser)
    with pytest.raises(
        materializer.CorpusMaterializationError,
        match="registered predecessor parsed-asset segment is missing",
    ):
        materializer.prefill_production_parsed_asset_cache_v1(
            inputs=inputs,
            parsed_asset_cache_root=current_cache,
            parsed_asset_recovery_context=current_context,
            allow_writes=True,
            predecessor_cache_root=predecessor_cache,
            predecessor_recovery_context=predecessor_context,
            compatibility_policy=policy,
        )
    assert tuple(current_cache.rglob("*")) == ()


def test_compatibility_policy_rejects_cross_lane_context(
    tmp_path: Path,
) -> None:
    source_cache = tmp_path / "source-cache"
    assets = _dolma_assets(source_cache)[:1]
    predecessor_cache = tmp_path / "predecessor-cache"
    current_cache = tmp_path / "current-cache"
    predecessor_cache.mkdir()
    current_cache.mkdir()
    current_context = _context()
    predecessor_context = _context(
        run_id="production-v4-replay-b",
        code_identity_sha256="4" * 64,
    )
    policy = _compatibility_policy(
        current=current_context,
        predecessor=_context(code_identity_sha256="4" * 64),
        predecessor_count=1,
        current_count=0,
    )
    inputs = _instance(
        tmp_path / "inputs",
        assets=assets,
        source_cache_root=source_cache,
        parsed_cache_root=current_cache,
    ).inputs
    with pytest.raises(Exception, match="differ by more than"):
        materializer.prefill_production_parsed_asset_cache_v1(
            inputs=inputs,
            parsed_asset_cache_root=current_cache,
            parsed_asset_recovery_context=current_context,
            allow_writes=True,
            predecessor_cache_root=predecessor_cache,
            predecessor_recovery_context=predecessor_context,
            compatibility_policy=policy,
        )
    assert tuple(current_cache.rglob("*")) == ()


def test_compatibility_policy_artifact_is_canonical_and_tamper_evident(
    tmp_path: Path,
) -> None:
    current = _context()
    predecessor = _context(
        code_identity_sha256=(
            "89a8b42dbe06edad2db7c67ae126c779a356612a3ed9e94587a98befb0d94657"
        )
    )
    policy = _compatibility_policy(
        current=current,
        predecessor=predecessor,
        predecessor_count=394,
        current_count=3,
    )
    payload = {
        "policy": asdict(policy),
        "policy_sha256": policy.identity_sha256,
        "schema": PARSED_ASSET_COMPATIBILITY_POLICY_ARTIFACT_SCHEMA_V1,
    }
    path = tmp_path / "authority.json"
    raw = canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)
    loaded, physical_bytes, physical_sha256 = (
        load_parsed_asset_compatibility_policy_v1(path)
    )
    assert loaded == policy
    assert physical_bytes == len(raw)
    assert physical_sha256 == _sha(raw)

    drifted = dict(payload)
    drifted["policy_sha256"] = "f" * 64
    path.write_bytes(canonical_json_bytes(drifted) + b"\n")
    with pytest.raises(Exception, match="identity drifted"):
        load_parsed_asset_compatibility_policy_v1(path)


def test_read_only_materialization_rejects_a_miss_before_parser_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cache = tmp_path / "source-cache"
    assets = _dolma_assets(source_cache)
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    instance = _instance(
        tmp_path / "read-only",
        assets=assets,
        source_cache_root=source_cache,
        parsed_cache_root=parsed_cache,
    )
    instance.parsed_asset_cache_read_only = True

    def forbidden_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("read-only materialization opened a source parser")

    monkeypatch.setattr(
        materializer,
        "iter_source_asset_events_v3",
        forbidden_parser,
    )
    with pytest.raises(
        materializer.CorpusMaterializationError,
        match="read-only parsed-asset cache is incomplete",
    ):
        instance._prefill_production_parsed_asset_cache()


@pytest.mark.parametrize("overlap", ("output", "work", "source"))
def test_parsed_cache_root_overlap_fails_before_fresh_root_creation(
    tmp_path: Path,
    overlap: str,
) -> None:
    durable = tmp_path / "durable"
    durable.mkdir()
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    output_root = tmp_path / "output"
    work_root = tmp_path / "work"
    if overlap == "output":
        parsed_cache = durable
        output_root = durable / "output"
    elif overlap == "work":
        parsed_cache = durable
        work_root = durable / "work"
    else:
        parsed_cache = source_cache

    instance = object.__new__(materializer._Materializer)
    instance.inputs = SimpleNamespace(cache_root=source_cache)
    instance.output_root = output_root
    instance.work_root = work_root
    instance.parsed_asset_cache_root = parsed_cache
    instance.parsed_asset_recovery_context = _context()

    with pytest.raises(materializer.CorpusMaterializationError, match="overlaps"):
        instance._prepare_roots()
    assert not output_root.exists()
    assert not work_root.exists()


@pytest.mark.parametrize("supply_root", (False, True))
def test_parsed_cache_arguments_are_paired_and_fixture_forbidden(
    tmp_path: Path,
    supply_root: bool,
) -> None:
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    kwargs = {
        "parsed_asset_cache_root": parsed_cache if supply_root else None,
        "parsed_asset_recovery_context": None if supply_root else _context(),
    }
    with pytest.raises(
        materializer.CorpusMaterializationError,
        match="must be supplied together",
    ):
        materializer._Materializer(
            inputs=SimpleNamespace(mode=materializer.FIXTURE_MODE),
            plan=SimpleNamespace(mode=materializer.FIXTURE_MODE),
            language_classifier=SimpleNamespace(classify=lambda _document: None),
            output_root=tmp_path / "output",
            work_root=tmp_path / "work",
            global_execution_provenance=None,
            runtime_build_receipt=None,
            **kwargs,
        )


def test_fixture_rejects_a_complete_parsed_cache_assignment(tmp_path: Path) -> None:
    parsed_cache = tmp_path / "parsed-cache"
    parsed_cache.mkdir()
    with pytest.raises(
        materializer.CorpusMaterializationError,
        match="fixture materialization",
    ):
        materializer._Materializer(
            inputs=SimpleNamespace(mode=materializer.FIXTURE_MODE),
            plan=SimpleNamespace(mode=materializer.FIXTURE_MODE),
            language_classifier=SimpleNamespace(classify=lambda _document: None),
            output_root=tmp_path / "output",
            work_root=tmp_path / "work",
            global_execution_provenance=None,
            runtime_build_receipt=None,
            parsed_asset_cache_root=parsed_cache,
            parsed_asset_recovery_context=_context(),
        )
