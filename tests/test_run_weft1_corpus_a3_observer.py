from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_weft1_corpus_a3_observer as cli
from training.weft1_corpus_breakdown_a3 import PathBreakdownError
from training.weft1_gtok_contract import canonical_sha256


def _dump_ids() -> list[str]:
    return [
        f"CC-MAIN-{2010 + index // 10:04d}-{index % 10:02d}"
        for index in range(110)
    ]


def _evidence() -> dict[str, object]:
    families: list[dict[str, object]] = []
    for source_family, assertion, configured in (
        (
            "dolma_web",
            "dolma3_bucket_0019_is_top_quality",
            [],
        ),
        (
            "fineweb_edu",
            "fineweb_edu_main_data_is_all_configured_cc_main_dumps",
            _dump_ids(),
        ),
    ):
        core = {
            "assertion": assertion,
            "configured_group_ids": configured,
            "derived_facts": ["fixture evidence"],
            "pin": "f" * 40,
            "source_family": source_family,
            "upstream_documents": [
                {
                    "bytes": 1,
                    "content_sha256": "d" * 64,
                    "path": "README.md",
                    "repository": "owner/repository",
                    "revision": "f" * 40,
                    "source_family": source_family,
                    "supports": "fixture proposition",
                    "url": "https://example.invalid/README.md",
                }
            ],
        }
        families.append(
            {
                **core,
                "family_evidence_sha256": canonical_sha256(
                    {"payload": core, "schema": cli.EVIDENCE_FAMILY_SCHEMA}
                ),
            }
        )
    core = {
        "authority_sha256": cli.A3_AUTHORITY_SHA256,
        "families": families,
        "schema": cli.EVIDENCE_SCHEMA,
    }
    return {
        **core,
        "receipt_sha256": canonical_sha256(
            {"payload": core, "schema": cli.EVIDENCE_RECEIPT_SCHEMA}
        ),
    }


def _file(path: str, size: int) -> dict[str, object]:
    return {
        "blob_id": ("a" if path.startswith("data/common_crawl") else "b") * 40,
        "path": path,
        "size": size,
        "type": "file",
    }


class _Api:
    def __init__(self, *, mutate_dolma: bool = False) -> None:
        self.mutate_dolma = mutate_dolma

    def list_repo_tree(self, repository: str, **_: object):
        if repository == cli.DOLMA_POOL_REPOSITORY:
            extra = 1 if self.mutate_dolma else 0
            return [
                _file("README.md", 5),
                _file("data/common_crawl-news-0018/part.jsonl.zst", 10),
                _file("data/common_crawl-news-0019/part.jsonl.zst", 20 + extra),
            ]
        assert repository == cli.FINEWEB_REPOSITORY
        return [
            _file(
                f"data/{dump_id}/"
                + ("000_00000.parquet" if index % 2 else "train-00000.parquet"),
                index + 1,
            )
            for index, dump_id in enumerate(_dump_ids())
        ]


def test_mint_then_replay_is_byte_stable_and_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    monkeypatch.setattr(cli, "verify_a3_authority_artifact", lambda: cli.A3_AUTHORITY_SHA256)
    monkeypatch.setattr(cli, "build_semantic_evidence_payload", lambda fetch: evidence)

    minted = cli.run_nonproduction_fixture(
        "mint", root=tmp_path, fetch=lambda _: b"", api=_Api()
    )
    replayed = cli.run_nonproduction_fixture(
        "replay", root=tmp_path, fetch=lambda _: b"", api=_Api()
    )

    assert minted["status"] == "A3_PATH_BREAKDOWN_MINTED"
    assert replayed["status"] == "A3_PATH_BREAKDOWN_REPLAYED"
    assert minted["breakdown_physical_sha256"] == replayed["breakdown_physical_sha256"]
    assert minted["breakdown_receipt_sha256"] == replayed["breakdown_receipt_sha256"]
    assert minted["dolma_selected_asset_count"] == 1
    assert minted["fineweb_selected_asset_count"] == 110
    assert minted["authorizes_downloads"] is False
    with pytest.raises(PathBreakdownError, match="both governed"):
        cli.run_nonproduction_fixture(
            "mint", root=tmp_path, fetch=lambda _: b"", api=_Api()
        )


def test_replay_detects_tree_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    monkeypatch.setattr(cli, "verify_a3_authority_artifact", lambda: cli.A3_AUTHORITY_SHA256)
    monkeypatch.setattr(cli, "build_semantic_evidence_payload", lambda fetch: evidence)
    cli.run_nonproduction_fixture(
        "mint", root=tmp_path, fetch=lambda _: b"", api=_Api()
    )

    with pytest.raises(PathBreakdownError, match="replay differs"):
        cli.run_nonproduction_fixture(
            "replay",
            root=tmp_path,
            fetch=lambda _: b"",
            api=_Api(mutate_dolma=True),
        )


def test_semantic_evidence_rejects_any_pinned_content_drift() -> None:
    with pytest.raises(PathBreakdownError, match="semantic evidence drifted"):
        cli.build_semantic_evidence_payload(lambda _: b"wrong")
