from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from scripts import run_weft1_stackedu_collision_audit_v1 as cli
from training.weft1_corpus_a3 import A3_AUTHORITY_SHA256, A3_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_fetch_a3 import (
    SOURCE_CACHE_MANIFEST_SCHEMA_V4,
    PAExecutionBindingV4,
    SourceCacheAssetV4,
    SourceCacheManifestV4,
)
from training.weft1_corpus_materialize_a3 import (
    SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4,
)
from training.weft1_corpus_source_io_a2 import SourceIOError
from training.weft1_corpus_sources_a2 import asset_order_digest_v3
from training.weft1_gtok_a1_contract import load_source_route_manifest
from training.weft1_gtok_contract import canonical_json_bytes
from training.weft1_stackedu_collision_audit_v1 import (
    CONTENT_DIVERGENCE,
    EXACT_REPEAT,
    GREEN_COMPLETE,
    LEDGER_NAME_V1,
    RECEIPT_NAME_V1,
    SCORE_ONLY_VARIANCE,
    STOP_CONTENT_DIVERGENCE,
    StackEduCollisionAuditError,
    load_stackedu_collision_audit_v1,
    run_stackedu_collision_audit_v1,
)


def _row(native_id: str, text: str, score: int) -> dict[str, object]:
    return {
        "added": "2026-01-01T00:00:00Z",
        "created": "2025-01-01T00:00:00Z",
        "id": native_id,
        "metadata": {"int_score": score},
        "source": "stackedu-fixture",
        "text": text,
    }


def _direct_python_row(
    native_id: str, text: str, score: int
) -> dict[str, object]:
    return {
        "blob_id": native_id,
        "detected_licenses": ["mit"],
        "download_success": True,
        "int_score": score,
        "language": "Python",
        "length_bytes": len(text.encode("utf-8")),
        "license_type": "permissive",
        "path": "fixture.py",
        "repo_name": "owner/repository",
        "score": float(score),
        "src_encoding": "UTF-8",
        "text": text,
    }


def _binding() -> PAExecutionBindingV4:
    return PAExecutionBindingV4(
        authority_sha256=A3_AUTHORITY_SHA256,
        effective_route_identity_sha256="d" * 64,
        breakdown_artifact_physical_sha256="e" * 64,
        breakdown_artifact_receipt_sha256="a" * 64,
        family_projection_sha256s=(
            ("dolma_web", "b" * 64),
            ("fineweb_edu", "c" * 64),
        ),
        campaign_root_seed=A3_CAMPAIGN_ROOT_SEED,
    )


def _fixture(
    tmp_path: Path,
    assets_rows: list[list[dict[str, object]]],
) -> tuple[Path, Path, Path]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    work_root = tmp_path / "work"
    work_root.mkdir()
    binding = _binding()
    route = next(
        row
        for row in load_source_route_manifest().routes
        if row.source_family == "stackedu"
    )
    assets: list[SourceCacheAssetV4] = []
    compressor = zstandard.ZstdCompressor(level=1)
    for index, rows in enumerate(assets_rows):
        decompressed = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        compressed = compressor.compress(decompressed)
        digest = hashlib.sha256(compressed).hexdigest()
        relative = f"assets/stackedu/{digest}.jsonl.zst"
        path = cache_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
        direct_python = bool(rows) and "blob_id" in rows[0]
        locator = (
            f"data/stack_edu-Python/part-{index:09d}.jsonl.zst"
            if direct_python
            else f"data/stack_edu-Java/shard_{index:08d}.jsonl.zst"
        )
        assets.append(
            SourceCacheAssetV4(
                source_family="stackedu",
                repository=route.repository,
                config=route.config,
                revision=route.revision,
                split=route.split,
                asset_locator=locator,
                relative_path=relative,
                bytes=len(compressed),
                sha256=digest,
                effective_route_receipt_sha256="f" * 64,
                execution_binding_sha256=binding.receipt_sha256,
            )
        )
    ordered = tuple(
        sorted(
            assets,
            key=lambda asset: (
                asset_order_digest_v3(asset.asset_locator),
                asset.source_family.encode("utf-8"),
                asset.asset_locator.encode("utf-8"),
                asset.sha256,
            ),
        )
    )
    manifest = SourceCacheManifestV4(
        schema=SOURCE_CACHE_MANIFEST_SCHEMA_V4,
        execution_binding=binding,
        effective_route_identity_sha256=binding.effective_route_identity_sha256,
        selection_plan_sha256="1" * 64,
        assets=ordered,
    )
    manifest_path = tmp_path / "source-cache-manifest-v4.json"
    envelope = {
        "manifest": asdict(manifest),
        "manifest_sha256": manifest.receipt_sha256,
        "schema": SOURCE_CACHE_MANIFEST_ARTIFACT_SCHEMA_V4,
    }
    manifest_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    return manifest_path, cache_root, work_root


def _run(
    tmp_path: Path,
    assets_rows: list[list[dict[str, object]]],
    *,
    output_name: str = "output",
):
    manifest, cache, work = _fixture(tmp_path, assets_rows)
    output = tmp_path / output_name
    receipt = run_stackedu_collision_audit_v1(
        source_manifest_path=manifest,
        cache_root=cache,
        work_root=work,
        output_root=output,
    )
    return manifest, cache, work, output, receipt


def _ledger_rows(output: Path) -> list[dict[str, object]]:
    raw = (output / LEDGER_NAME_V1).read_bytes().splitlines()
    return [json.loads(row) for row in raw]


def test_exact_cross_asset_repeat_is_hash_only_green_and_replay_stable(
    tmp_path: Path,
) -> None:
    rows = [
        [
            _row("unique-native-secret", "unique-secret-code", 3),
            _row("duplicate-native-secret", "duplicate-secret-code", 4),
        ],
        [_row("duplicate-native-secret", "duplicate-secret-code", 4)],
    ]
    manifest, cache, work, output_a, first = _run(tmp_path, rows, output_name="a")
    output_b = tmp_path / "b"
    second = run_stackedu_collision_audit_v1(
        source_manifest_path=manifest,
        cache_root=cache,
        work_root=work,
        output_root=output_b,
    )

    assert first.status == GREEN_COMPLETE
    assert first.eligible_record_count == 3
    assert first.distinct_eligible_native_id_count == 2
    assert first.exact_repeat_count == 1
    assert first.score_only_variance_count == 0
    assert first.content_divergence_count == 0
    assert first.receipt_sha256 == second.receipt_sha256
    assert (output_a / RECEIPT_NAME_V1).read_bytes() == (
        output_b / RECEIPT_NAME_V1
    ).read_bytes()
    assert (output_a / LEDGER_NAME_V1).read_bytes() == (
        output_b / LEDGER_NAME_V1
    ).read_bytes()
    durable = (output_a / RECEIPT_NAME_V1).read_bytes() + (
        output_a / LEDGER_NAME_V1
    ).read_bytes()
    assert b"native-secret" not in durable
    assert b"secret-code" not in durable
    evidence = _ledger_rows(output_a)
    assert [row["classification"] for row in evidence] == [EXACT_REPEAT]
    loaded = load_stackedu_collision_audit_v1(
        receipt_path=output_a / RECEIPT_NAME_V1,
        collision_ledger_path=output_a / LEDGER_NAME_V1,
    )
    assert loaded == first


def test_score_only_variance_is_evidenced_but_first_occurrence_remains_canonical(
    tmp_path: Path,
) -> None:
    _, _, _, output, receipt = _run(
        tmp_path,
        [
            [
                _row("same-id", "same-text", 3),
                _row("same-id", "same-text", 3),
                _row("same-id", "same-text", 4),
                _row("not-scanned", "later", 3),
            ]
        ],
    )

    assert receipt.status == GREEN_COMPLETE
    assert receipt.parse_event_count == 4
    assert receipt.eligible_record_count == 4
    assert receipt.distinct_eligible_native_id_count == 2
    assert receipt.exact_repeat_count == 1
    assert receipt.score_only_variance_count == 1
    assert receipt.content_divergence_count == 0
    assert [row["classification"] for row in _ledger_rows(output)] == [
        EXACT_REPEAT,
        SCORE_ONLY_VARIANCE,
    ]


def test_normalized_and_direct_python_assets_use_distinct_pinned_bindings(
    tmp_path: Path,
) -> None:
    _, _, _, output, receipt = _run(
        tmp_path,
        [
            [_row("shared-blob", "same-code", 3)],
            [_direct_python_row("shared-blob", "same-code", 3)],
        ],
    )

    assert receipt.status == GREEN_COMPLETE
    assert receipt.eligible_record_count == 2
    assert receipt.distinct_eligible_native_id_count == 1
    assert receipt.exact_repeat_count == 1
    assert len(receipt.parser_binding_asset_counts) == 2
    assert {row[1:] for row in receipt.parser_binding_asset_counts} == {(1, 1)}
    assert _ledger_rows(output)[0]["classification"] == EXACT_REPEAT


def test_content_divergence_stops_and_different_ids_same_text_do_not_collide(
    tmp_path: Path,
) -> None:
    _, _, _, output, divergent = _run(
        tmp_path,
        [[_row("same-id", "first", 3), _row("same-id", "second", 3)]],
    )
    assert divergent.status == STOP_CONTENT_DIVERGENCE
    assert divergent.content_divergence_count == 1
    assert divergent.terminal_evidence_sha256 == _ledger_rows(output)[0][
        "evidence_sha256"
    ]
    assert _ledger_rows(output)[0]["classification"] == CONTENT_DIVERGENCE

    other = tmp_path / "other"
    other.mkdir()
    _, _, _, _, clean = _run(
        other,
        [[_row("id-a", "shared", 3), _row("id-b", "shared", 3)]],
    )
    assert clean.status == GREEN_COMPLETE
    assert clean.exact_repeat_count == 0
    assert clean.distinct_eligible_native_id_count == 2


def test_quality_drops_are_counted_but_not_eligible_identity_rows(
    tmp_path: Path,
) -> None:
    _, _, _, _, receipt = _run(
        tmp_path,
        [[_row("dropped", "low", 2), _row("retained", "high", 3)]],
    )
    counts = dict(receipt.parse_disposition_counts)
    assert receipt.status == GREEN_COMPLETE
    assert receipt.parse_event_count == 2
    assert receipt.eligible_record_count == 1
    assert counts["DROP_QUALITY_LT3"] == 1
    assert counts["RETAIN"] == 1


def test_cache_and_schema_drift_fail_before_minting_receipt(tmp_path: Path) -> None:
    manifest, cache, work = _fixture(
        tmp_path,
        [[_row("stable", "text", 3)]],
    )
    asset = next((cache / "assets" / "stackedu").iterdir())
    raw = bytearray(asset.read_bytes())
    raw[-1] ^= 1
    asset.write_bytes(bytes(raw))
    output = tmp_path / "tampered-output"
    with pytest.raises(SourceIOError, match="changed hash or size"):
        run_stackedu_collision_audit_v1(
            source_manifest_path=manifest,
            cache_root=cache,
            work_root=work,
            output_root=output,
        )
    assert not output.exists()

    schema_root = tmp_path / "schema"
    schema_root.mkdir()
    bad = _row("stable", "text", 3)
    bad["unexpected"] = True
    bad_manifest, bad_cache, bad_work = _fixture(schema_root, [[bad]])
    bad_output = schema_root / "output"
    with pytest.raises(SourceIOError, match="schema drifted"):
        run_stackedu_collision_audit_v1(
            source_manifest_path=bad_manifest,
            cache_root=bad_cache,
            work_root=bad_work,
            output_root=bad_output,
        )
    assert not bad_output.exists()


def test_loader_rejects_ledger_mutation_and_output_root_reuse(tmp_path: Path) -> None:
    manifest, cache, work, output, receipt = _run(
        tmp_path,
        [[_row("same", "text", 3), _row("same", "text", 3)]],
    )
    with pytest.raises(StackEduCollisionAuditError, match="must be fresh"):
        run_stackedu_collision_audit_v1(
            source_manifest_path=manifest,
            cache_root=cache,
            work_root=work,
            output_root=output,
        )
    ledger = output / LEDGER_NAME_V1
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    with pytest.raises(StackEduCollisionAuditError, match="drifted"):
        load_stackedu_collision_audit_v1(
            receipt_path=output / RECEIPT_NAME_V1,
            collision_ledger_path=ledger,
        )
    assert receipt.exact_repeat_count == 1


def test_cli_mints_stop_receipt_and_reports_physical_failure(
    tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    manifest, cache, work = _fixture(
        tmp_path,
        [[_row("same", "text", 3), _row("same", "other", 3)]],
    )
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "--source-cache-manifest",
                str(manifest),
                "--source-cache",
                str(cache),
                "--work-root",
                str(work),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    stdout, stderr = capsysbinary.readouterr()
    assert stderr == b""
    assert json.loads(stdout)["receipt"]["status"] == STOP_CONTENT_DIVERGENCE

    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    broken_manifest, broken_cache, broken_work = _fixture(
        broken_root, [[_row("stable", "text", 3)]]
    )
    broken_asset = next((broken_cache / "assets" / "stackedu").iterdir())
    broken_asset.write_bytes(b"not-zstd")
    assert (
        cli.main(
            [
                "--source-cache-manifest",
                str(broken_manifest),
                "--source-cache",
                str(broken_cache),
                "--work-root",
                str(broken_work),
                "--output-root",
                str(broken_root / "output"),
            ]
        )
        == 2
    )
    stdout, stderr = capsysbinary.readouterr()
    assert stdout == b""
    assert json.loads(stderr)["error"] == "SourceTransportError"
