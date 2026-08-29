from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import training.weft1_corpus_a3 as corpus_a3

from training.weft1_corpus_a2 import (
    A2_CAMPAIGN_ROOT_SEED,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
)
from training.weft1_corpus_a3 import (
    A1_ROUTE_MANIFEST_PHYSICAL_SHA256,
    A1_ROUTE_MANIFEST_RECEIPT_SHA256,
    A2_BINDINGS_SHA256,
    A3_AUTHORITY_SHA256,
    A3BreakdownPending,
    A3_CAMPAIGN_ROOT_SEED,
    A3_EFFECTIVE_ROUTE_OVERLAY_PATH,
    A3_EFFECTIVE_ROUTE_OVERLAY_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
    OVERLAY_FAMILIES,
    OVERLAY_MODE,
    PASSTHROUGH_MODE,
    execution_authority_v4_bound_sha256,
    finalize_effective_route_overlay_a3,
    load_effective_route_overlay_a3,
    load_effective_route_overlay_template_a3,
    verify_a3_authority_artifact,
)
from training.weft1_corpus_breakdown_a3 import (
    DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
    FIXTURE_OBSERVATION_MODE_A3,
    FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
    PathMemberReceiptA3,
    PinnedSemanticEvidenceA3,
    PriorRouteDeclarationA3,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
    build_dolma_path_breakdown_a3,
    build_fineweb_path_breakdown_a3,
    build_upstream_path_breakdown_a3,
    write_upstream_path_breakdown_a3,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    load_source_route_manifest,
)
from training.weft1_gtok_contract import canonical_json_bytes


def _payload() -> dict[str, object]:
    return json.loads(A3_EFFECTIVE_ROUTE_OVERLAY_PATH.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_pending_template_binds_forward_only_chain_seed_and_predecessors() -> None:
    manifest = load_effective_route_overlay_template_a3()
    assert verify_a3_authority_artifact() == A3_AUTHORITY_SHA256
    assert GTOK_EXECUTION_AUTHORITY_CHAIN_V4[:-1] == GTOK_EXECUTION_AUTHORITY_CHAIN_V3
    assert GTOK_EXECUTION_AUTHORITY_CHAIN_V4[-1] == A3_AUTHORITY_SHA256
    assert A3_CAMPAIGN_ROOT_SEED == A2_CAMPAIGN_ROOT_SEED == 17843936115933234841
    assert manifest.predecessors.a2_bindings_sha256 == A2_BINDINGS_SHA256
    assert (
        manifest.predecessors.a1_route_manifest_physical_sha256
        == A1_ROUTE_MANIFEST_PHYSICAL_SHA256
    )
    assert (
        manifest.predecessors.a1_route_manifest_receipt_sha256
        == A1_ROUTE_MANIFEST_RECEIPT_SHA256
    )


def test_pending_template_has_two_overlays_five_passthroughs_and_no_fake_hashes() -> None:
    manifest = load_effective_route_overlay_template_a3()
    assert tuple(row.source_family for row in manifest.overlay_rows) == SOURCE_FAMILIES
    overlays = tuple(row for row in manifest.overlay_rows if row.mode == OVERLAY_MODE)
    passthroughs = tuple(
        row for row in manifest.overlay_rows if row.mode == PASSTHROUGH_MODE
    )
    assert tuple(row.source_family for row in overlays) == OVERLAY_FAMILIES
    assert len(passthroughs) == 5
    for row in overlays:
        assert row.breakdown_artifact is not None
        assert not row.breakdown_artifact.is_bound
        assert row.family_projection_sha256 is None
        assert not row.effective_declaration.is_bound
    assert overlays[0].breakdown_artifact == overlays[1].breakdown_artifact


def test_checked_in_template_is_byte_pinned_and_production_fails_pending() -> None:
    raw = A3_EFFECTIVE_ROUTE_OVERLAY_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == A3_EFFECTIVE_ROUTE_OVERLAY_SHA256
    with pytest.raises(A3BreakdownPending):
        load_effective_route_overlay_a3()


def test_alternate_ledger_requires_explicit_fixture_mode(tmp_path: Path) -> None:
    path = tmp_path / "overlay.json"
    _write(path, _payload())
    with pytest.raises(ValueError, match="fixture mode"):
        load_effective_route_overlay_a3(path, allow_pending_template=True)
    replayed = load_effective_route_overlay_a3(
        path,
        allow_pending_template=True,
        nonproduction_fixture=True,
    )
    assert replayed.overlay_rows[0].source_family == "dolma_web"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(authority_sha256="0" * 64),
        lambda value: value["campaign_seed_policy"].update(campaign_root_seed=1),
        lambda value: value["predecessors"].update(a2_bindings_sha256="0" * 64),
        lambda value: value["predecessors"].update(
            a1_route_manifest_physical_sha256="0" * 64
        ),
        lambda value: value["predecessors"].update(
            a1_route_manifest_receipt_sha256="0" * 64
        ),
        lambda value: value["overlay_rows"][0].update(
            base_route_receipt_sha256="0" * 64
        ),
        lambda value: value["overlay_rows"].reverse(),
        lambda value: value["overlay_rows"][0].update(
            family_projection_sha256="0" * 64
        ),
    ),
)
def test_authority_seed_order_and_placeholder_tampering_fail_closed(
    tmp_path: Path,
    mutation,
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "tampered.json"
    _write(path, payload)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        load_effective_route_overlay_a3(
            path,
            allow_pending_template=True,
            nonproduction_fixture=True,
        )


def test_v4_hash_domain_rejects_v3_schema_and_changes_with_a3_authority() -> None:
    with pytest.raises(ValueError, match="explicit v4"):
        execution_authority_v4_bound_sha256("legacy_v3", {"x": 1})
    assert execution_authority_v4_bound_sha256(
        "test_a3_v4", {"x": 1}
    ) != execution_authority_v4_bound_sha256("test_other_a3_v4", {"x": 1})


def test_duplicate_json_key_is_rejected_before_overlay_validation(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_bytes(
        b'{"schema":"weft1_corpus_effective_route_overlay_a3_v4",'
        b'"schema":"weft1_corpus_effective_route_overlay_a3_v4"}\n'
    )
    with pytest.raises(ValueError):
        load_effective_route_overlay_a3(
            path,
            allow_pending_template=True,
            nonproduction_fixture=True,
        )


def _member(path: str, size: int) -> PathMemberReceiptA3:
    return PathMemberReceiptA3(
        path=path,
        upstream_bytes=size,
        blob_identity_kind="git_sha1",
        blob_identity=hashlib.sha1(path.encode("utf-8")).hexdigest(),
    )


def test_finalizer_derives_both_routes_and_self_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = load_effective_route_overlay_template_a3()
    base = {route.source_family: route for route in load_source_route_manifest().routes}
    dolma_route = base["dolma_web"]
    fineweb_route = base["fineweb_edu"]
    dump_ids = tuple(
        f"CC-MAIN-{2010 + index // 10:04d}-{index % 10:02d}"
        for index in range(110)
    )
    dolma = build_dolma_path_breakdown_a3(
        (
            _member("data/common_crawl-news-0018/a.jsonl.zst", 1),
            _member("data/common_crawl-news-0019/a.jsonl.zst", 7_000_000_000),
        ),
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
            evidence_id="fixture-dolma",
            locator="fixture#dolma",
            pin="f" * 40,
            content_sha256="d" * 64,
            assertion=DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
        ),
    )
    fineweb = build_fineweb_path_breakdown_a3(
        tuple(
            _member(f"data/{dump_id}/000_00000.parquet", 100_000_000)
            for dump_id in dump_ids
        ),
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
            evidence_id="fixture-fineweb",
            locator="fixture#fineweb",
            pin="e" * 40,
            content_sha256="e" * 64,
            assertion=FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
        ),
        configured_main_dump_ids=dump_ids,
    )
    receipt = build_upstream_path_breakdown_a3(
        authority_sha256=A3_AUTHORITY_SHA256,
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity=PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
        dolma=dolma,
        fineweb=fineweb,
    )
    artifact = tmp_path / "training" / "breakdown.json"
    artifact.parent.mkdir(parents=True)
    write_upstream_path_breakdown_a3(receipt, artifact)
    monkeypatch.setattr(corpus_a3, "REPOSITORY_ROOT", tmp_path)

    fixture_receipt = build_upstream_path_breakdown_a3(
        authority_sha256=A3_AUTHORITY_SHA256,
        observation_mode=FIXTURE_OBSERVATION_MODE_A3,
        observation_client_identity=FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
        dolma=dolma,
        fineweb=fineweb,
    )
    fixture_artifact = tmp_path / "training" / "fixture-breakdown.json"
    write_upstream_path_breakdown_a3(fixture_receipt, fixture_artifact)
    with pytest.raises(corpus_a3.A3RouteError, match="rejects nonproduction"):
        finalize_effective_route_overlay_a3(
            pending,
            breakdown_path=fixture_artifact,
        )

    finalized = finalize_effective_route_overlay_a3(
        pending,
        breakdown_path=artifact,
    )
    declarations = {
        row.source_family: row.effective_declaration
        for row in finalized.overlay_rows
    }
    assert finalized.status == corpus_a3.RESOLVED_STATUS
    assert (
        declarations["dolma_web"].resolution
        == "CONFIRM_TOP_BUCKET_SELECTOR_REMINT_DECLARATION"
    )
    assert (
        declarations["fineweb_edu"].resolution
        == "WIDEN_TO_ALL_MAIN_DATA_CC_DUMPS"
    )
    resolved = corpus_a3.resolve_effective_routes_a3(
        finalized,
        breakdown_root=tmp_path,
    )
    assert (
        resolved.effective_route_identity_sha256
        == finalized.claimed_effective_route_identity_sha256
    )
