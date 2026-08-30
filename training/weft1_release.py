"""Fail-closed WEFT-1 public-release bindings.

This module turns the 2026-08-30 release ratification into an immutable section
that the P-A corpus manifest and later model-card builder can inherit.  It does
not publish an artifact, mint P-B, or access corpus text.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_BINDINGS_PATH = REPO_ROOT / "training" / "weft1_release_bindings_20260830.json"
RELEASE_AUTHORITY_PATH = (
    REPO_ROOT / "docs" / "STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md"
)
MODEL_CARD_TEMPLATE_PATH = REPO_ROOT / "docs" / "WEFT1_MODEL_CARD_TEMPLATE.md"

RELEASE_AUTHORITY_BYTES = 5_746
RELEASE_AUTHORITY_SHA256 = (
    "d8c4f3bf8829bbe48e2464bf758ec3594ef730a0f952712099b45d183ca2ab3e"
)
REQUIRED_PROVENANCE_SENTENCE = (
    "from-scratch in weights, not in data provenance — trained from random "
    "initialization on an open corpus including model-generated reasoning traces "
    "in a declared final phase."
)
NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE = (
    'no sentence of the form "WEFT-1 outperforms [named public model]" is ever written.'
)
WEIGHTS_LICENSE_SPDX = "Apache-2.0"
ATTRIBUTION_TEXT_BY_SOURCE = {
    "dolma3": (
        "Dolma 3 — This training corpus includes Dolma 3 Pool and Dolma 3 Mix "
        "material from the Allen Institute for AI, licensed under the Open Data "
        "Commons Attribution License v1.0 (ODC-By); cite Team Olmo et al., “Olmo "
        "3” (2025), arXiv:2512.13961."
    ),
    "fineweb_edu": (
        "FineWeb-Edu — This training corpus includes FineWeb-Edu by Anton Lozhkov, "
        "Loubna Ben Allal, Leandro von Werra, and Thomas Wolf, licensed under "
        "ODC-By v1.0 and subject to Common Crawl’s Terms of Use; cite “FineWeb-Edu: "
        "the Finest Collection of Educational Content” (2024), DOI 10.57967/hf/2497."
    ),
    "stackedu": (
        "Stack-Edu — The code stratum includes Stack-Edu material routed through "
        "the pinned Dolma 3 Mix; Stack-Edu is a 125B-token educational-code corpus "
        "filtered from The Stack v2. Cite Loubna Ben Allal et al., “SmolLM2: When "
        "Smol Goes Big — Data-Centric Training of a Small Language Model” (2025), "
        "arXiv:2502.02737. The Stack v2’s original per-file licenses, attribution "
        "requirements, and removal/opt-out process remain applicable."
    ),
}
STACKEDU_DISCLOSURE_TEXT = (
    "Stack-Edu route — Executed from allenai/dolma3_mix-6T at revision "
    "689a3ea2d8217e64d73a5058913fa43ad15e81aa (2026-01-15T05:36:27Z). "
    "Stack-Edu is derived from The Stack v2/StarCoder2Data: the underlying source "
    "snapshot uses the Software Heritage graph dated 2023-09-06 and GitHub Archive "
    "metadata through 2023-09-14. The upstream card records that StarCoder2 used "
    "v2.0.1, which incorporated validated opt-outs through 2023-10-20; WEFT-1 "
    "inherits The Stack v2’s continuing removal/takedown posture."
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FAMILIES = ("dolma3", "fineweb_edu", "stackedu")
_PUBLIC_ARTIFACTS = (
    "pipeline_code",
    "corpus_manifest_with_pins_shas_seeds_and_dedup_rates",
    "d1_replay_instructions",
)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _require_sha(value: Any, *, width: int, name: str) -> str:
    pattern = _SHA1 if width == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase {width}-character hash")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_without_duplicates,
    )
    if not isinstance(value, dict):
        raise TypeError("release bindings must be one JSON object")
    return value


def verify_release_authority_artifact(path: Path = RELEASE_AUTHORITY_PATH) -> str:
    raw = path.read_bytes()
    if len(raw) != RELEASE_AUTHORITY_BYTES:
        raise RuntimeError("release authority byte count drifted")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RELEASE_AUTHORITY_SHA256:
        raise RuntimeError("release authority SHA-256 drifted")
    return digest


def _validate_card(card: Mapping[str, Any], *, name: str) -> None:
    required = {"repository", "revision", "url", "content_sha256"}
    optional = {"revision_date_utc"}
    if not required.issubset(card) or not set(card).issubset(required | optional):
        raise ValueError(f"{name} has an invalid card evidence shape")
    if not isinstance(card["repository"], str) or "/" not in card["repository"]:
        raise ValueError(f"{name} repository must be an owner/name pair")
    _require_sha(card["revision"], width=40, name=f"{name} revision")
    _require_sha(card["content_sha256"], width=64, name=f"{name} card SHA-256")
    expected_prefix = f"https://huggingface.co/datasets/{card['repository']}/blob/{card['revision']}/"
    if not isinstance(card["url"], str) or not card["url"].startswith(expected_prefix):
        raise ValueError(f"{name} URL is not pinned to its declared revision")
    if "revision_date_utc" in card:
        date = card["revision_date_utc"]
        if not isinstance(date, str) or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", date
        ) is None:
            raise ValueError(f"{name} revision date must be canonical UTC")


def load_release_bindings(path: Path = RELEASE_BINDINGS_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    _require_exact_keys(
        payload,
        {
            "schema",
            "authority",
            "public_release",
            "claims",
            "attributions",
            "stackedu_disclosure",
        },
        "release bindings",
    )
    if payload["schema"] != "weft1_release_bindings_v1":
        raise ValueError("release binding schema drifted")

    authority = payload["authority"]
    if not isinstance(authority, dict):
        raise TypeError("authority binding must be an object")
    _require_exact_keys(authority, {"path", "bytes", "sha256"}, "authority")
    if authority != {
        "path": "docs/STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md",
        "bytes": RELEASE_AUTHORITY_BYTES,
        "sha256": RELEASE_AUTHORITY_SHA256,
    }:
        raise ValueError("release authority binding drifted")

    public_release = payload["public_release"]
    if not isinstance(public_release, dict):
        raise TypeError("public release posture must be an object")
    if public_release.get("weights_license_spdx") != WEIGHTS_LICENSE_SPDX:
        raise ValueError("weights license must remain Apache-2.0")
    if tuple(public_release.get("publish", ())) != _PUBLIC_ARTIFACTS:
        raise ValueError("public corpus artifact allowlist drifted")
    if public_release.get("never_publish") != ["raw_text_shards"]:
        raise ValueError("raw text shards must remain prohibited from publication")
    if public_release.get("public_corpus_identity") != "manifest_sha256":
        raise ValueError("the manifest SHA must remain the public corpus identity")

    claims = payload["claims"]
    if not isinstance(claims, dict):
        raise TypeError("claim controls must be an object")
    if claims.get("required_provenance_sentence") != REQUIRED_PROVENANCE_SENTENCE:
        raise ValueError("required provenance sentence must remain verbatim")
    if (
        claims.get("no_named_public_model_comparisons_rule")
        != NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE
    ):
        raise ValueError("no-named-public-model rule must remain verbatim")
    if claims.get("matched_comparison_rule") != (
        "Comparative claims ride on the matched-compute control only."
    ):
        raise ValueError("matched-compute comparison rule drifted")

    attributions = payload["attributions"]
    if not isinstance(attributions, list):
        raise TypeError("attributions must be a list")
    if tuple(row.get("source_family") for row in attributions) != _SOURCE_FAMILIES:
        raise ValueError("attribution source order or coverage drifted")
    for index, row in enumerate(attributions):
        if not isinstance(row, dict):
            raise TypeError("each attribution must be an object")
        _require_exact_keys(row, {"source_family", "exact_text", "cards"}, "attribution")
        if not isinstance(row["exact_text"], str) or not row["exact_text"].strip():
            raise ValueError("attribution exact text may not be empty")
        if row["exact_text"] != ATTRIBUTION_TEXT_BY_SOURCE[row["source_family"]]:
            raise ValueError("attribution exact text drifted")
        cards = row["cards"]
        if not isinstance(cards, list) or not cards:
            raise ValueError("each attribution must bind at least one pinned card")
        for card_index, card in enumerate(cards):
            if not isinstance(card, dict):
                raise TypeError("card evidence must be an object")
            _validate_card(card, name=f"attributions[{index}].cards[{card_index}]")

    disclosure = payload["stackedu_disclosure"]
    if not isinstance(disclosure, dict):
        raise TypeError("StackEdu disclosure must be an object")
    if disclosure.get("execution_repository") != "allenai/dolma3_mix-6T":
        raise ValueError("StackEdu execution repository drifted")
    if disclosure.get("execution_revision") != (
        "689a3ea2d8217e64d73a5058913fa43ad15e81aa"
    ):
        raise ValueError("StackEdu execution revision drifted")
    if disclosure.get("execution_revision_date_utc") != "2026-01-15T05:36:27Z":
        raise ValueError("StackEdu execution revision date drifted")
    if disclosure.get("software_heritage_graph_snapshot_date") != "2023-09-06":
        raise ValueError("StackEdu source snapshot date drifted")
    if disclosure.get("github_archive_metadata_through_date") != "2023-09-14":
        raise ValueError("StackEdu metadata cutoff drifted")
    if disclosure.get("starcoder2_opt_out_cutoff") != "2023-10-20":
        raise ValueError("StackEdu inherited opt-out cutoff drifted")
    if disclosure.get("upstream_removal_posture") != "INHERITED_AND_MUST_BE_HONORED":
        raise ValueError("StackEdu upstream removal posture drifted")
    if disclosure.get("exact_text") != STACKEDU_DISCLOSURE_TEXT:
        raise ValueError("StackEdu exact disclosure text drifted")

    return payload


def release_manifest_section(path: Path = RELEASE_BINDINGS_PATH) -> dict[str, Any]:
    """Return the immutable release section to embed in the P-A manifest."""

    verify_release_authority_artifact()
    payload = load_release_bindings(path)
    return deepcopy(
        {
            "schema": payload["schema"],
            "public_release": payload["public_release"],
            "claims": payload["claims"],
            "attributions": payload["attributions"],
            "stackedu_disclosure": payload["stackedu_disclosure"],
        }
    )


def verify_model_card_template(path: Path = MODEL_CARD_TEMPLATE_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    payload = load_release_bindings()
    required = [
        REQUIRED_PROVENANCE_SENTENCE,
        NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE,
        *(row["exact_text"] for row in payload["attributions"]),
        payload["stackedu_disclosure"]["exact_text"],
        "license: apache-2.0",
        "Raw text shards are never published.",
    ]
    missing = [value for value in required if text.count(value) != 1]
    if missing:
        raise RuntimeError("model-card template is missing or duplicates a release binding")
