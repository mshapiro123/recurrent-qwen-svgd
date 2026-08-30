from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.weft1_corpus_a2 import A2_LANGUAGE_ID_BINDING
from training.weft1_corpus_a3 import execution_authority_v4_bound_sha256
from training.weft1_release import (
    FASTTEXT_RETENTION_SENTENCE,
    MODEL_CARD_TEMPLATE_PATH,
    NAMED_PUBLIC_MODEL_COMPARISON_STATUS,
    NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE,
    RELEASE_CARD_EVIDENCE_PATH,
    RELEASE_AUTHORITY_SHA256,
    REQUIRED_PROVENANCE_SENTENCE,
    WEIGHTS_LICENSE_SPDX,
    load_release_card_evidence,
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
    assert text.count(NAMED_PUBLIC_MODEL_COMPARISON_STATUS) == 1
    assert text.count(FASTTEXT_RETENTION_SENTENCE) == 1


def test_release_manifest_section_carries_publication_and_claim_controls() -> None:
    section = release_manifest_section()
    assert execution_authority_v4_bound_sha256(
        "weft1_release_manifest_section_v4", section
    ) == "14c90a7d1391dbbc05e8301c07e249152f0d652fa248f4931f6d37eb4ccf7c2b"
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
    assert "language_id" not in section
    assert "named_public_model_comparisons" not in section["claims"]
    assert all(
        "evidence_id" not in card
        for row in section["attributions"]
        for card in row["cards"]
    )


def test_fasttext_retention_is_cross_bound_to_the_a2_runtime_contract() -> None:
    language_id = load_release_bindings()["language_id"]
    assert language_id == {
        "decision": "RETAIN_FASTTEXT_NO_SWAP",
        "binding_receipt_sha256": A2_LANGUAGE_ID_BINDING.receipt_sha256,
        "package": A2_LANGUAGE_ID_BINDING.package,
        "package_version": A2_LANGUAGE_ID_BINDING.package_version,
        "adapter": A2_LANGUAGE_ID_BINDING.adapter,
        "model_bytes": A2_LANGUAGE_ID_BINDING.model_bytes,
        "model_sha256": A2_LANGUAGE_ID_BINDING.model_sha256,
        "scope": A2_LANGUAGE_ID_BINDING.scope,
    }


def test_three_attributions_have_pinned_hashed_card_evidence() -> None:
    payload = load_release_bindings()
    evidence = load_release_card_evidence()
    assert [row["source_family"] for row in payload["attributions"]] == [
        "dolma3",
        "fineweb_edu",
        "stackedu",
    ]
    assert all(row["exact_text"] for row in payload["attributions"])
    for row in payload["attributions"]:
        assert row["cards"]
        for card in row["cards"]:
            assert card["evidence_id"] in evidence
            assert f"/blob/{card['revision']}/README.md" in card["url"]
            assert len(card["content_sha256"]) == 64
            evidence_card = evidence[card["evidence_id"]]
            assert card["repository"] == evidence_card["repository"]
            assert card["revision"] == evidence_card["revision"]
            assert card["content_sha256"] == evidence_card["content_sha256"]

    assert [
        card["evidence_id"]
        for row in payload["attributions"]
        for card in row["cards"]
    ] == list(evidence)


def test_card_evidence_physically_pins_every_source_card(tmp_path: Path) -> None:
    raw = bytearray(RELEASE_CARD_EVIDENCE_PATH.read_bytes())
    marker = b'"content_bytes": 14847'
    offset = raw.index(marker)
    raw[offset + len(marker) - 1] = ord("8")
    changed = tmp_path / "card-evidence.json"
    changed.write_bytes(raw)

    with pytest.raises(RuntimeError, match="card evidence identity SHA-256"):
        load_release_card_evidence(changed)


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
    "claim",
    (
        "WEFT-1 outperforms Llama 4 on the public benchmark.",
        "WEFT-1 achieves higher accuracy than Qwen3.",
        "Gemma is outperformed by WEFT-1.",
        "Compared with Mistral, WEFT-1 performs better.",
        "WEFT-1 performs better compared with DeepSeek-V3.",
        "WEFT-1 > Phi-4.",
        "WEFT-1 is superior to Claude 5.",
        "OLMo trails WEFT-1.",
    ),
)
def test_model_card_rejects_named_public_model_comparisons(
    tmp_path: Path, claim: str
) -> None:
    text = MODEL_CARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    path = tmp_path / "README.md"
    path.write_text(text + "\n" + claim + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="named-public-model comparison"):
        verify_model_card_template(path)


def test_model_card_allows_the_registered_matched_control_comparison(
    tmp_path: Path,
) -> None:
    text = MODEL_CARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    path = tmp_path / "README.md"
    path.write_text(
        text
        + "\nWEFT-1 outperforms the matched-compute control on the registered evaluation.\n",
        encoding="utf-8",
    )

    verify_model_card_template(path)


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
            lambda value: value["public_release"].update(
                publish_raw_text_shards=True
            ),
            "public release posture must contain exactly",
        ),
        (
            lambda value: value["public_release"].update(
                reproducibility_claim="copy_shards_instead"
            ),
            "D1 replay reproducibility claim",
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
            lambda value: value["claims"].update(
                named_public_model_comparisons=["Llama 4"]
            ),
            "comparison ledger must remain empty",
        ),
        (
            lambda value: value["claims"].update(p4_generator_rule="current freeze"),
            "claim controls must contain exactly",
        ),
        (
            lambda value: value["language_id"].update(package="replacement"),
            "fastText retention binding",
        ),
        (
            lambda value: value["attributions"][0]["cards"][0].update(
                revision="main"
            ),
            "40-character",
        ),
        (
            lambda value: value["attributions"][2]["cards"][0].update(
                content_sha256="0" * 64
            ),
            "differs from pinned card evidence",
        ),
        (
            lambda value: value["stackedu_disclosure"].update(
                software_heritage_graph_snapshot_date="2024-01-01"
            ),
            "snapshot",
        ),
        (
            lambda value: value["stackedu_disclosure"].update(
                p4_generator_rule="included in current freeze"
            ),
            "StackEdu disclosure must contain exactly",
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
