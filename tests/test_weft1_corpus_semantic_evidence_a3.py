from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.weft1_corpus_a3 import A3_AUTHORITY_SHA256
from training.weft1_corpus_semantic_evidence_a3 import (
    SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1,
    SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1,
    SEMANTIC_EVIDENCE_SCHEMA_A3_V1,
    SemanticEvidenceA3Error,
    load_semantic_evidence_snapshot_a3,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256


def _payload() -> dict[str, object]:
    dump_ids = [
        f"CC-MAIN-{2010 + index // 10:04d}-{index % 10:02d}"
        for index in range(110)
    ]
    families: list[dict[str, object]] = []
    for source_family, assertion, configured in (
        ("dolma_web", "dolma3_bucket_0019_is_top_quality", []),
        (
            "fineweb_edu",
            "fineweb_edu_main_data_is_all_configured_cc_main_dumps",
            dump_ids,
        ),
    ):
        core = {
            "assertion": assertion,
            "configured_group_ids": configured,
            "derived_facts": ["pinned fixture fact"],
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
                    "supports": "pinned fixture proposition",
                    "url": "https://example.invalid/README.md",
                }
            ],
        }
        families.append(
            {
                **core,
                "family_evidence_sha256": canonical_sha256(
                    {
                        "payload": core,
                        "schema": SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1,
                    }
                ),
            }
        )
    core = {
        "authority_sha256": A3_AUTHORITY_SHA256,
        "families": families,
        "schema": SEMANTIC_EVIDENCE_SCHEMA_A3_V1,
    }
    return {
        **core,
        "receipt_sha256": canonical_sha256(
            {
                "payload": core,
                "schema": SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1,
            }
        ),
    }


def test_semantic_evidence_loader_binds_physical_typed_and_family_identities(
    tmp_path: Path,
) -> None:
    payload = _payload()
    path = tmp_path / "evidence.json"
    raw = canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)

    observed_raw, observed, identity = load_semantic_evidence_snapshot_a3(path)

    assert observed_raw == raw
    assert observed == payload
    assert identity.physical_bytes == len(raw)
    assert identity.receipt_sha256 == payload["receipt_sha256"]
    assert tuple(family for family, _ in identity.family_receipt_sha256s) == (
        "dolma_web",
        "fineweb_edu",
    )


def test_semantic_evidence_rejects_nested_or_noncanonical_tamper(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["families"][1]["configured_group_ids"].pop()  # type: ignore[index,union-attr]
    path = tmp_path / "tampered.json"
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(SemanticEvidenceA3Error):
        load_semantic_evidence_snapshot_a3(path)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")
    with pytest.raises(SemanticEvidenceA3Error, match="not canonical"):
        load_semantic_evidence_snapshot_a3(noncanonical)
