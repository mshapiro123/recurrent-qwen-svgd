from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.weft1_release import (
    MODEL_CARD_TEMPLATE_PATH,
    NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE,
    RELEASE_AUTHORITY_SHA256,
    REQUIRED_PROVENANCE_SENTENCE,
    WEIGHTS_LICENSE_SPDX,
    load_release_bindings,
    release_manifest_section,
    verify_model_card_template,
    verify_release_authority_artifact,
)


def _payload() -> dict[str, object]:
    return json.loads(
        Path("training/weft1_release_bindings_20260830.json").read_text(
            encoding="utf-8"
        )
    )


def _write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_authority_and_model_card_are_exactly_bound() -> None:
    assert verify_release_authority_artifact() == RELEASE_AUTHORITY_SHA256
    verify_model_card_template()
    text = MODEL_CARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert text.count(REQUIRED_PROVENANCE_SENTENCE) == 1
    assert text.count(NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE) == 1


def test_release_manifest_section_carries_publication_and_claim_controls() -> None:
    section = release_manifest_section()
    assert section["public_release"]["weights_license_spdx"] == WEIGHTS_LICENSE_SPDX
    assert section["public_release"]["never_publish"] == ["raw_text_shards"]
    assert section["public_release"]["public_corpus_identity"] == "manifest_sha256"
    assert (
        section["claims"]["required_provenance_sentence"]
        == REQUIRED_PROVENANCE_SENTENCE
    )
    assert (
        section["claims"]["no_named_public_model_comparisons_rule"]
        == NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE
    )


def test_three_attributions_have_pinned_hashed_card_evidence() -> None:
    payload = load_release_bindings()
    assert [row["source_family"] for row in payload["attributions"]] == [
        "dolma3",
        "fineweb_edu",
        "stackedu",
    ]
    assert all(row["exact_text"] for row in payload["attributions"])
    for row in payload["attributions"]:
        assert row["cards"]
        for card in row["cards"]:
            assert f"/blob/{card['revision']}/README.md" in card["url"]
            assert len(card["content_sha256"]) == 64


def test_execution_card_pins_match_the_committed_source_routes() -> None:
    routes = {
        row["source_family"]: row
        for row in json.loads(
            Path("training/weft1_gtok_source_routes_20260828.json").read_text(
                encoding="utf-8"
            )
        )["routes"]
    }
    attributions = {
        row["source_family"]: row for row in load_release_bindings()["attributions"]
    }
    dolma_pool = attributions["dolma3"]["cards"][0]
    dolma_mix = attributions["dolma3"]["cards"][1]
    fineweb = attributions["fineweb_edu"]["cards"][0]
    assert (dolma_pool["revision"], dolma_pool["content_sha256"]) == (
        routes["dolma_web"]["revision"],
        routes["dolma_web"]["card_sha256"],
    )
    assert (dolma_mix["revision"], dolma_mix["content_sha256"]) == (
        routes["stackedu"]["revision"],
        routes["stackedu"]["card_sha256"],
    )
    assert (fineweb["revision"], fineweb["content_sha256"]) == (
        routes["fineweb_edu"]["revision"],
        routes["fineweb_edu"]["card_sha256"],
    )


def test_stackedu_separates_execution_pin_from_source_snapshot() -> None:
    disclosure = load_release_bindings()["stackedu_disclosure"]
    assert disclosure["execution_revision"] == (
        "689a3ea2d8217e64d73a5058913fa43ad15e81aa"
    )
    assert disclosure["execution_revision_date_utc"] == "2026-01-15T05:36:27Z"
    assert disclosure["software_heritage_graph_snapshot_date"] == "2023-09-06"
    assert disclosure["github_archive_metadata_through_date"] == "2023-09-14"
    assert disclosure["starcoder2_opt_out_cutoff"] == "2023-10-20"
    assert disclosure["upstream_removal_posture"] == "INHERITED_AND_MUST_BE_HONORED"


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value["public_release"].update(
                weights_license_spdx="other"
            ),
            "Apache-2.0",
        ),
        (
            lambda value: value["public_release"].update(never_publish=[]),
            "raw text shards",
        ),
        (
            lambda value: value["claims"].update(
                required_provenance_sentence="paraphrased"
            ),
            "verbatim",
        ),
        (
            lambda value: value["claims"].update(
                no_named_public_model_comparisons_rule="best in class"
            ),
            "verbatim",
        ),
        (
            lambda value: value["attributions"][0]["cards"][0].update(
                revision="main"
            ),
            "40-character",
        ),
        (
            lambda value: value["stackedu_disclosure"].update(
                software_heritage_graph_snapshot_date="2024-01-01"
            ),
            "snapshot",
        ),
    ),
)
def test_release_binding_mutations_fail_closed(
    tmp_path: Path, mutation, match: str
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "release.json"
    _write(path, payload)
    with pytest.raises(ValueError, match=match):
        load_release_bindings(path)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"weft1_release_bindings_v1",'
        '"schema":"weft1_release_bindings_v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_release_bindings(path)
