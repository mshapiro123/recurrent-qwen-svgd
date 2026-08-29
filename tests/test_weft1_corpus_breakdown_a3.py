from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.weft1_corpus_breakdown_a3 import (
    DOLMA_EXCLUDED_CLASSIFICATION,
    DOLMA_CONFIRM_RESOLUTION_A3_V1,
    DOLMA_NON_COMMON_CRAWL_CLASSIFICATION,
    DOLMA_SELECTED_CLASSIFICATION,
    DOLMA_SOURCE_FAMILY,
    DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
    FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
    FINEWEB_SAMPLE_CLASSIFICATION,
    FINEWEB_SCORE_CLASSIFICATION,
    FINEWEB_SELECTED_CLASSIFICATION,
    FINEWEB_SOURCE_FAMILY,
    FINEWEB_WIDEN_RESOLUTION_A3_V1,
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_SHA256_A3,
    FIXTURE_OBSERVATION_MODE_A3,
    PATH_BREAKDOWN_STATUS_A3_V1,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3,
    PRODUCTION_OBSERVATION_MODE_A3,
    REPOSITORY_METADATA_CLASSIFICATION,
    PathBreakdownError,
    PathMemberReceiptA3,
    PinnedSemanticEvidenceA3,
    PriorRouteDeclarationA3,
    build_dolma_path_breakdown_a3,
    build_fineweb_path_breakdown_a3,
    build_upstream_path_breakdown_a3,
    load_upstream_path_breakdown_a3,
    load_upstream_path_breakdown_snapshot_a3,
    observe_hf_tree_files_a3,
    project_family_resolution_a3,
    replay_upstream_path_breakdown_a3,
    write_upstream_path_breakdown_a3,
)
from training.weft1_corpus_a3 import A3_AUTHORITY_SHA256
from training.weft1_gtok_contract import canonical_json_bytes


AUTHORITY_SHA256 = A3_AUTHORITY_SHA256
DOLMA_REVISION = "b" * 40
FINEWEB_REVISION = "c" * 40


def _member(path: str, size: int = 10) -> PathMemberReceiptA3:
    return PathMemberReceiptA3(
        path=path,
        upstream_bytes=size,
        blob_identity_kind="git_sha1",
        blob_identity=hashlib.sha1(path.encode("utf-8")).hexdigest(),
    )


def _dump_ids() -> tuple[str, ...]:
    values = tuple(
        f"CC-MAIN-{2013 + index // 10:04d}-{10 + index % 10:02d}"
        for index in range(110)
    )
    assert len(values) == len(set(values)) == 110
    return values


def _evidence(source_family: str) -> PinnedSemanticEvidenceA3:
    if source_family == DOLMA_SOURCE_FAMILY:
        return PinnedSemanticEvidenceA3(
            evidence_id="dolma3-quality-ordering",
            locator="https://example.invalid/dolma3-construction",
            pin="pinned-document-v1",
            content_sha256="d" * 64,
            assertion=DOLMA_TOP_QUALITY_ASSERTION_A3_V1,
        )
    return PinnedSemanticEvidenceA3(
        evidence_id="fineweb-edu-main-data",
        locator="https://example.invalid/fineweb-edu-card",
        pin=FINEWEB_REVISION,
        content_sha256="e" * 64,
        assertion=FINEWEB_MAIN_DATA_ASSERTION_A3_V1,
    )


def _prior(source_family: str) -> PriorRouteDeclarationA3:
    selector = (
        "data/common_crawl-*-0019/*.jsonl.zst"
        if source_family == DOLMA_SOURCE_FAMILY
        else "data/*/train-*.parquet"
    )
    return PriorRouteDeclarationA3(
        source_family=source_family,
        asset_selector=selector,
        asset_count=999,
        available_bytes=999_999,
        declaration_receipt_sha256=("1" if source_family == DOLMA_SOURCE_FAMILY else "2")
        * 64,
    )


def _dolma_members() -> tuple[PathMemberReceiptA3, ...]:
    return (
        _member("README.md", 1),
        _member("data/common_crawl-news-0000/part-000.jsonl.zst", 20),
        _member("data/common_crawl-news-0018/part-000.jsonl.zst", 30),
        _member("data/common_crawl-news-0019/part-000.jsonl.zst", 40),
        _member("data/common_crawl-books-0019/part-000.jsonl.zst", 50),
        _member("data/books/part-000.jsonl.zst", 60),
    )


def _fineweb_members() -> tuple[PathMemberReceiptA3, ...]:
    members = [
        _member(
            f"data/{dump_id}/"
            + (f"train-{index:05d}.parquet" if index % 2 else f"{index:05d}.parquet"),
            index + 1,
        )
        for index, dump_id in enumerate(_dump_ids())
    ]
    members.extend(
        (
            _member(f"data/{_dump_ids()[0]}/numeric-shard-7.parquet", 500),
            _member("data/sample-10BT/train-00000.parquet", 600),
            _member("data/sample-10BT-score-3/00000.parquet", 700),
            _member("sample/10BT/00000.parquet", 800),
            _member("README.md", 2),
        )
    )
    return tuple(members)


def _families(*, reverse: bool = False):
    dolma_members = _dolma_members()
    fineweb_members = _fineweb_members()
    dump_ids = _dump_ids()
    if reverse:
        dolma_members = tuple(reversed(dolma_members))
        fineweb_members = tuple(reversed(fineweb_members))
        dump_ids = tuple(reversed(dump_ids))
    dolma = build_dolma_path_breakdown_a3(
        dolma_members,
        repository="allenai/dolma3_pool",
        revision=DOLMA_REVISION,
        prior_declaration=_prior(DOLMA_SOURCE_FAMILY),
        semantic_evidence=_evidence(DOLMA_SOURCE_FAMILY),
    )
    fineweb = build_fineweb_path_breakdown_a3(
        fineweb_members,
        repository="HuggingFaceFW/fineweb-edu",
        revision=FINEWEB_REVISION,
        prior_declaration=_prior(FINEWEB_SOURCE_FAMILY),
        semantic_evidence=_evidence(FINEWEB_SOURCE_FAMILY),
        configured_main_dump_ids=dump_ids,
    )
    return dolma, fineweb


def _receipt(*, reverse: bool = False):
    dolma, fineweb = _families(reverse=reverse)
    return build_upstream_path_breakdown_a3(
        authority_sha256=AUTHORITY_SHA256,
        observation_mode=FIXTURE_OBSERVATION_MODE_A3,
        observation_client_identity=FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
        dolma=dolma,
        fineweb=fineweb,
    )


def test_dolma_selects_only_0019_and_preserves_explained_lower_buckets() -> None:
    dolma, _ = _families()
    selected = tuple(group for group in dolma.groups if group.selected)
    excluded = tuple(group for group in dolma.groups if not group.selected)

    assert {group.classification for group in selected} == {
        DOLMA_SELECTED_CLASSIFICATION
    }
    assert {group.group_id for group in selected} == {
        "common_crawl-books-0019",
        "common_crawl-news-0019",
    }
    assert {group.classification for group in excluded} == {
        DOLMA_EXCLUDED_CLASSIFICATION,
        DOLMA_NON_COMMON_CRAWL_CLASSIFICATION,
        REPOSITORY_METADATA_CLASSIFICATION,
    }
    assert dolma.selected_asset_count == 2
    assert dolma.selected_upstream_bytes == 90
    assert dolma.repository_asset_count == sum(
        group.asset_count for group in dolma.groups
    )
    assert dolma.repository_upstream_bytes == sum(
        group.upstream_bytes for group in dolma.groups
    )
    assert dolma.reminted_asset_count != dolma.prior_declaration.asset_count
    assert dolma.reminted_available_bytes != dolma.prior_declaration.available_bytes


def test_dolma_requires_pinned_top_quality_evidence_and_exact_paths() -> None:
    wrong = PinnedSemanticEvidenceA3(
        evidence_id="wrong",
        locator="https://example.invalid/wrong",
        pin="wrong",
        content_sha256="3" * 64,
        assertion="numeric_suffix_probably_increases_quality",
    )
    with pytest.raises(PathBreakdownError, match="requires pinned 0019"):
        build_dolma_path_breakdown_a3(
            _dolma_members(),
            repository="allenai/dolma3_pool",
            revision=DOLMA_REVISION,
            prior_declaration=_prior(DOLMA_SOURCE_FAMILY),
            semantic_evidence=wrong,
        )

    with pytest.raises(PathBreakdownError, match="maximum bucket is 0018"):
        build_dolma_path_breakdown_a3(
            (_member("data/common_crawl-news-0018/part.jsonl.zst"),),
            repository="allenai/dolma3_pool",
            revision=DOLMA_REVISION,
            prior_declaration=_prior(DOLMA_SOURCE_FAMILY),
            semantic_evidence=_evidence(DOLMA_SOURCE_FAMILY),
        )

    with pytest.raises(PathBreakdownError, match="maximum bucket is 0020"):
        build_dolma_path_breakdown_a3(
            (
                *_dolma_members(),
                _member("data/common_crawl-news-0020/part.jsonl.zst"),
            ),
            repository="allenai/dolma3_pool",
            revision=DOLMA_REVISION,
            prior_declaration=_prior(DOLMA_SOURCE_FAMILY),
            semantic_evidence=_evidence(DOLMA_SOURCE_FAMILY),
        )

    with pytest.raises(PathBreakdownError, match="unclassified Dolma"):
        build_dolma_path_breakdown_a3(
            (_member("data/common_crawl-news/not-a-shard.jsonl.zst"),),
            repository="allenai/dolma3_pool",
            revision=DOLMA_REVISION,
            prior_declaration=_prior(DOLMA_SOURCE_FAMILY),
            semantic_evidence=_evidence(DOLMA_SOURCE_FAMILY),
        )


def test_fineweb_exactly_covers_110_main_dumps_and_all_parquet_names() -> None:
    _, fineweb = _families()
    selected = tuple(group for group in fineweb.groups if group.selected)
    excluded = {group.group_id: group for group in fineweb.groups if not group.selected}

    assert len(selected) == 110
    assert tuple(sorted(group.group_id for group in selected)) == tuple(
        sorted(_dump_ids())
    )
    assert fineweb.selected_asset_count == 111
    assert fineweb.repository_asset_count == sum(
        group.asset_count for group in fineweb.groups
    )
    assert fineweb.repository_upstream_bytes == sum(
        group.upstream_bytes for group in fineweb.groups
    )
    first_dump = next(group for group in selected if group.group_id == _dump_ids()[0])
    assert first_dump.asset_count == 2
    assert (
        excluded["data/sample-10BT"].classification
        == FINEWEB_SAMPLE_CLASSIFICATION
    )
    assert excluded["data/sample-10BT"].selected is False
    assert excluded["sample/10BT"].classification == FINEWEB_SAMPLE_CLASSIFICATION
    assert (
        excluded["data/sample-10BT-score-3"].classification
        == FINEWEB_SCORE_CLASSIFICATION
    )
    assert (
        excluded["repository_metadata"].classification
        == REPOSITORY_METADATA_CLASSIFICATION
    )
    assert all(
        group.classification == FINEWEB_SELECTED_CLASSIFICATION
        for group in selected
    )


@pytest.mark.parametrize("case", ("omitted", "extra", "unknown", "nested"))
def test_fineweb_coverage_fails_closed(case: str) -> None:
    members = list(_fineweb_members())
    configured = _dump_ids()
    if case == "omitted":
        missing = configured[-1]
        members = [member for member in members if f"data/{missing}/" not in member.path]
        match = "omits 1 configured"
    elif case == "extra":
        members.append(_member("data/CC-MAIN-2099-99/train-00000.parquet"))
        match = "extra main dump"
    elif case == "unknown":
        members.append(_member("data/production/train-00000.parquet"))
        match = "unknown data group"
    else:
        members.append(
            _member(f"data/{configured[0]}/nested/train-00000.parquet")
        )
        match = "unclassified FineWeb"
    with pytest.raises(PathBreakdownError, match=match):
        build_fineweb_path_breakdown_a3(
            tuple(members),
            repository="HuggingFaceFW/fineweb-edu",
            revision=FINEWEB_REVISION,
            prior_declaration=_prior(FINEWEB_SOURCE_FAMILY),
            semantic_evidence=_evidence(FINEWEB_SOURCE_FAMILY),
            configured_main_dump_ids=configured,
        )


def test_fineweb_configuration_must_be_exactly_110_canonical_unique_ids() -> None:
    for invalid in (
        _dump_ids()[:-1],
        (*_dump_ids()[:-1], _dump_ids()[0]),
        (*_dump_ids()[:-1], "sample-10BT"),
    ):
        with pytest.raises(PathBreakdownError, match="FineWeb"):
            build_fineweb_path_breakdown_a3(
                _fineweb_members(),
                repository="HuggingFaceFW/fineweb-edu",
                revision=FINEWEB_REVISION,
                prior_declaration=_prior(FINEWEB_SOURCE_FAMILY),
                semantic_evidence=_evidence(FINEWEB_SOURCE_FAMILY),
                configured_main_dump_ids=invalid,
            )


def test_duplicate_repository_path_fails_before_group_aggregation() -> None:
    duplicate = _fineweb_members()[0]
    with pytest.raises(PathBreakdownError, match="repeats a repository file"):
        build_fineweb_path_breakdown_a3(
            (*_fineweb_members(), duplicate),
            repository="HuggingFaceFW/fineweb-edu",
            revision=FINEWEB_REVISION,
            prior_declaration=_prior(FINEWEB_SOURCE_FAMILY),
            semantic_evidence=_evidence(FINEWEB_SOURCE_FAMILY),
            configured_main_dump_ids=_dump_ids(),
        )


def test_input_order_cannot_change_receipt_or_artifact_bytes(tmp_path: Path) -> None:
    forward = _receipt(reverse=False)
    reverse = _receipt(reverse=True)
    assert reverse == forward
    assert reverse.receipt_sha256 == forward.receipt_sha256
    assert forward.status == PATH_BREAKDOWN_STATUS_A3_V1
    assert forward.authorizes_downloads is False
    assert forward.observation_mode == FIXTURE_OBSERVATION_MODE_A3
    assert (
        forward.observation_client_identity_sha256
        == FIXTURE_OBSERVATION_CLIENT_IDENTITY_SHA256_A3
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_sha = write_upstream_path_breakdown_a3(forward, first)
    second_sha = write_upstream_path_breakdown_a3(reverse, second)
    assert second.read_bytes() == first.read_bytes()
    assert second_sha == first_sha == hashlib.sha256(first.read_bytes()).hexdigest()
    assert b'"members"' not in first.read_bytes()


def test_observation_mode_is_bound_to_an_exact_client_identity() -> None:
    dolma, fineweb = _families()
    production = build_upstream_path_breakdown_a3(
        authority_sha256=AUTHORITY_SHA256,
        observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
        observation_client_identity=PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
        dolma=dolma,
        fineweb=fineweb,
    )
    assert production.observation_client_identity.expand is False
    assert (
        production.observation_client_identity_sha256
        == PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3
    )
    assert (
        production.observation_client_identity_sha256
        != FIXTURE_OBSERVATION_CLIENT_IDENTITY_SHA256_A3
    )

    with pytest.raises(PathBreakdownError, match="does not match its mode"):
        build_upstream_path_breakdown_a3(
            authority_sha256=AUTHORITY_SHA256,
            observation_mode=PRODUCTION_OBSERVATION_MODE_A3,
            observation_client_identity=FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3,
            dolma=dolma,
            fineweb=fineweb,
        )


def test_breakdown_artifact_round_trip_and_nested_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    path = tmp_path / "breakdown.json"
    artifact_sha256 = write_upstream_path_breakdown_a3(receipt, path)
    assert artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    replayed = load_upstream_path_breakdown_a3(
        path,
        expected_authority_sha256=AUTHORITY_SHA256,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert replayed == receipt
    raw, snapshot = load_upstream_path_breakdown_snapshot_a3(
        path,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert raw == path.read_bytes()
    assert snapshot == receipt

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipt"]["families"][1]["groups"][0]["upstream_bytes"] += 1
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(PathBreakdownError):
        load_upstream_path_breakdown_a3(
            tampered,
            expected_authority_sha256=AUTHORITY_SHA256,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(PathBreakdownError, match="not canonical"):
        load_upstream_path_breakdown_a3(
            noncanonical,
            expected_authority_sha256=AUTHORITY_SHA256,
        )

    with pytest.raises(PathBreakdownError, match="differs from expected"):
        load_upstream_path_breakdown_a3(
            path,
            expected_authority_sha256=AUTHORITY_SHA256,
            expected_receipt_sha256="f" * 64,
        )


def test_compact_projection_and_live_replay_are_deterministic() -> None:
    receipt = _receipt()
    dolma = project_family_resolution_a3(receipt, DOLMA_SOURCE_FAMILY)
    fineweb = project_family_resolution_a3(receipt, FINEWEB_SOURCE_FAMILY)

    assert dolma.selected_asset_count == 2
    assert dolma.selected_upstream_bytes == 90
    assert dolma.resolution == DOLMA_CONFIRM_RESOLUTION_A3_V1
    assert dolma.breakdown_receipt_sha256 == receipt.receipt_sha256
    assert fineweb.selected_asset_count == 111
    assert len(fineweb.selected_path_patterns) == 110
    assert fineweb.resolution == FINEWEB_WIDEN_RESOLUTION_A3_V1
    assert fineweb.breakdown_receipt_sha256 == receipt.receipt_sha256

    replayed = replay_upstream_path_breakdown_a3(
        receipt,
        dolma_members=tuple(reversed(_dolma_members())),
        fineweb_members=tuple(reversed(_fineweb_members())),
    )
    assert replayed == receipt

    changed = list(_dolma_members())
    changed[-1] = _member(changed[-1].path, changed[-1].upstream_bytes + 1)
    with pytest.raises(PathBreakdownError, match="replay differs"):
        replay_upstream_path_breakdown_a3(
            receipt,
            dolma_members=tuple(changed),
            fineweb_members=_fineweb_members(),
        )


def test_hf_tree_observer_handles_pinned_objects_zero_bytes_and_duplicates() -> None:
    path = "data/CC-MAIN-2024-10/00000.parquet"
    rows = (
        SimpleNamespace(
            path=path,
            size=0,
            blob_id=hashlib.sha1(path.encode("utf-8")).hexdigest(),
            lfs=None,
            xet_hash=None,
        ),
        SimpleNamespace(path="data", tree_id="4" * 40),
    )
    observed = observe_hf_tree_files_a3(rows)
    assert len(observed) == 1
    assert observed[0].path == path
    assert observed[0].upstream_bytes == 0

    with pytest.raises(PathBreakdownError, match="repeats a repository file"):
        observe_hf_tree_files_a3((rows[0], rows[0]))


def test_writer_refuses_overwrite_and_stale_partial(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "breakdown.json"
    write_upstream_path_breakdown_a3(receipt, path)
    with pytest.raises(PathBreakdownError, match="overwrite"):
        write_upstream_path_breakdown_a3(receipt, path)

    other = tmp_path / "other.json"
    other.with_name(other.name + ".partial").write_bytes(b"stale")
    with pytest.raises(PathBreakdownError, match="stale"):
        write_upstream_path_breakdown_a3(receipt, other)
