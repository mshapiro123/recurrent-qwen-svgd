from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import zstandard

import training.weft1_corpus_materialize_a2 as materializer
from training.weft1_corpus_parsed_asset_cache_v1 import (
    ParsedAssetRecoveryContextV1,
)
from training.weft1_corpus_sources_a2 import (
    SourceCacheAssetV3,
    VerifiedLocalCacheAssetV3,
    load_exact_source_routes_v3,
)
from training.weft1_gtok_a1_contract import SOURCE_FAMILIES


def _sha(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _context() -> ParsedAssetRecoveryContextV1:
    return ParsedAssetRecoveryContextV1(
        run_id="production-v4-replay-a",
        durable_marker_physical_sha256="0" * 64,
        runtime_identity_sha256="1" * 64,
        code_identity_sha256="2" * 64,
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
