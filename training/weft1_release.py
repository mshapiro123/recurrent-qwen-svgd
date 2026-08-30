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
RELEASE_CARD_EVIDENCE_PATH = (
    REPO_ROOT / "training" / "weft1_release_card_evidence_20260830.json"
)
SOURCE_ROUTE_MANIFEST_PATH = (
    REPO_ROOT / "training" / "weft1_gtok_source_routes_20260828.json"
)
RELEASE_AUTHORITY_PATH = (
    REPO_ROOT / "docs" / "STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md"
)
MODEL_CARD_TEMPLATE_PATH = REPO_ROOT / "docs" / "WEFT1_MODEL_CARD_TEMPLATE.md"

RELEASE_AUTHORITY_BYTES = 5_746
RELEASE_AUTHORITY_SHA256 = (
    "d8c4f3bf8829bbe48e2464bf758ec3594ef730a0f952712099b45d183ca2ab3e"
)
RELEASE_CARD_EVIDENCE_IDENTITY_SHA256 = (
    "2170c35db83ac05bd676fe73d7e3f7cc3a52cd206e7958b69317a280651e7a21"
)
SOURCE_ROUTE_MANIFEST_BYTES = 9_114
SOURCE_ROUTE_MANIFEST_SHA256 = (
    "1cf99ea33b72013f4bf07101aad8c9b5124879afe3de9f28991e6427ea861a6c"
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
REPRODUCIBILITY_CLAIM = "replay_the_public_manifest_and_verify_d1"
FASTTEXT_RETENTION_SENTENCE = (
    "Language identification retains the pinned fastText lid.176.bin classifier; "
    "no substitute model is authorized."
)
NAMED_PUBLIC_MODEL_COMPARISON_STATUS = "Named public-model comparisons: none."
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
_NAMED_COMPARATIVE_PATTERNS = (
    re.compile(
        r"\bWEFT[- ]?1\b[^\n.!?]{0,160}?\b(?:outperform(?:s|ed|ing)?|"
        r"beat(?:s|en|ing)?|surpass(?:es|ed|ing)?|exceed(?:s|ed|ing)?)\b\s+"
        r"(?P<target>[^\n.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bWEFT[- ]?1\b[^\n.!?]{0,160}?\b(?:better|stronger|superior|"
        r"higher|lower|faster|more\s+accurate)\b[^\n.!?]{0,80}?\bthan\b\s+"
        r"(?P<target>[^\n.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<target>[^\n.!?]{1,120}?)\b(?:is|are|was|were)\s+"
        r"(?:outperformed|beaten|surpassed|exceeded)\s+by\s+\bWEFT[- ]?1\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bWEFT[- ]?1\b[^\n.!?]{0,160}?\b(?:compared\s+(?:with|to)|"
        r"relative\s+to|versus|vs\.?|ranks?\s+(?:above|ahead\s+of)|"
        r"edges?\s+out|(?:superior|inferior)\s+to|ahead\s+of|behind)\s+"
        r"(?P<target>[^\n.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcompared\s+(?:with|to)\s+(?P<target>[^,\n.!?;]+),"
        r"[^\n.!?]{0,160}\bWEFT[- ]?1\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bWEFT[- ]?1\b\s*(?:>|≥)\s*(?P<target>[^\n.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<target>[^\n.!?]{1,120}?)\s*(?:<|≤)\s*\bWEFT[- ]?1\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<target>[^\n.!?]{1,120}?)\b(?:trails?|lags?\s+behind)\s+"
        r"\bWEFT[- ]?1\b",
        re.IGNORECASE,
    ),
)
_SOURCE_FAMILIES = ("dolma3", "fineweb_edu", "stackedu")
_PUBLIC_ARTIFACTS = (
    "pipeline_code",
    "corpus_manifest_with_pins_shas_seeds_and_dedup_rates",
    "d1_replay_instructions",
)
_CARD_EVIDENCE_IDS_BY_SOURCE = {
    "dolma3": ("dolma3_pool_execution_card", "dolma3_mix_execution_card"),
    "fineweb_edu": ("fineweb_edu_execution_card",),
    "stackedu": ("stackedu_source_card", "the_stack_v2_source_card"),
}
_RELEASE_MANIFEST_CLAIM_KEYS = (
    "required_provenance_sentence",
    "matched_comparison_rule",
    "no_named_public_model_comparisons_rule",
)
_RELEASE_MANIFEST_CARD_KEYS = (
    "repository",
    "revision",
    "url",
    "content_sha256",
    "revision_date_utc",
)


def _is_allowed_matched_control_target(value: str) -> bool:
    target = " ".join(value.casefold().strip().split())
    if target.startswith("the "):
        target = target[4:]
    prefix = "matched-compute control"
    if not target.startswith(prefix):
        return False
    tail = target[len(prefix) :].strip()
    if not tail:
        return True
    if re.search(r"\b(?:and|or|versus|vs|compared|than)\b", tail):
        return False
    return tail.startswith(("on ", "by ", "at ", "under ", "for ", "in ", "("))


def _reject_named_public_model_comparisons(text: str) -> None:
    scrubbed = text.replace(NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE, "")
    for pattern in _NAMED_COMPARATIVE_PATTERNS:
        for match in pattern.finditer(scrubbed):
            if not _is_allowed_matched_control_target(match.group("target")):
                raise RuntimeError(
                    "model card contains a prohibited named-public-model comparison"
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


def _verified_file_bytes(
    path: Path, *, expected_bytes: int, expected_sha256: str, name: str
) -> bytes:
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise RuntimeError(f"{name} byte count drifted")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"{name} SHA-256 drifted")
    return raw


def _load_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs_without_duplicates,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be one JSON object")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _validate_card_evidence_row(card: Mapping[str, Any], *, index: int) -> None:
    common = {
        "evidence_id",
        "source_family",
        "evidence_kind",
        "repository",
        "revision",
        "url",
        "resolve_url",
        "content_bytes",
        "content_sha256",
    }
    kind = card.get("evidence_kind")
    if kind == "committed_execution_route":
        expected = common | {"route_source_family"}
    elif kind == "pinned_upstream_card":
        expected = common | {"revision_date_utc"}
    else:
        raise ValueError(f"card evidence row {index} has an unknown evidence kind")
    _require_exact_keys(card, expected, f"card evidence row {index}")
    if card.get("source_family") not in _SOURCE_FAMILIES:
        raise ValueError(f"card evidence row {index} has an unknown source family")
    if not isinstance(card.get("evidence_id"), str) or not card["evidence_id"]:
        raise ValueError(f"card evidence row {index} lacks an evidence ID")
    if not isinstance(card.get("repository"), str) or "/" not in card["repository"]:
        raise ValueError(f"card evidence row {index} has an invalid repository")
    _require_sha(card.get("revision"), width=40, name="card evidence revision")
    _require_sha(
        card.get("content_sha256"), width=64, name="card evidence content SHA-256"
    )
    if type(card.get("content_bytes")) is not int or card["content_bytes"] <= 0:
        raise ValueError(f"card evidence row {index} has an invalid byte count")
    blob_prefix = (
        f"https://huggingface.co/datasets/{card['repository']}/blob/"
        f"{card['revision']}/"
    )
    resolve_prefix = (
        f"https://huggingface.co/datasets/{card['repository']}/resolve/"
        f"{card['revision']}/"
    )
    if not isinstance(card.get("url"), str) or not card["url"].startswith(
        blob_prefix
    ):
        raise ValueError(f"card evidence row {index} has an unpinned blob URL")
    if not isinstance(card.get("resolve_url"), str) or not card[
        "resolve_url"
    ].startswith(resolve_prefix):
        raise ValueError(f"card evidence row {index} has an unpinned resolve URL")
    if "revision_date_utc" in card and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", card["revision_date_utc"]
    ) is None:
        raise ValueError(f"card evidence row {index} has an invalid revision date")


def load_release_card_evidence(
    path: Path = RELEASE_CARD_EVIDENCE_PATH,
    route_path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> dict[str, dict[str, Any]]:
    raw = path.read_bytes()
    payload = _load_json_bytes(raw, name="release card evidence")
    if (
        hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != RELEASE_CARD_EVIDENCE_IDENTITY_SHA256
    ):
        raise RuntimeError("release card evidence identity SHA-256 drifted")
    _require_exact_keys(
        payload,
        {"schema", "authority_sha256", "source_route_manifest", "cards"},
        "release card evidence",
    )
    if payload.get("schema") != "weft1_release_card_evidence_v1":
        raise ValueError("release card evidence schema drifted")
    if payload.get("authority_sha256") != RELEASE_AUTHORITY_SHA256:
        raise ValueError("release card evidence authority drifted")

    route_binding = payload.get("source_route_manifest")
    if not isinstance(route_binding, dict):
        raise TypeError("source route evidence binding must be an object")
    _require_exact_keys(
        route_binding, {"path", "bytes", "sha256", "schema"}, "source route binding"
    )
    if route_binding != {
        "path": "training/weft1_gtok_source_routes_20260828.json",
        "bytes": SOURCE_ROUTE_MANIFEST_BYTES,
        "sha256": SOURCE_ROUTE_MANIFEST_SHA256,
        "schema": "weft1_gtok_source_route_manifest_v2",
    }:
        raise ValueError("source route evidence binding drifted")
    route_raw = _verified_file_bytes(
        route_path,
        expected_bytes=SOURCE_ROUTE_MANIFEST_BYTES,
        expected_sha256=SOURCE_ROUTE_MANIFEST_SHA256,
        name="source route manifest",
    )
    route_payload = _load_json_bytes(route_raw, name="source route manifest")
    if route_payload.get("schema") != route_binding["schema"]:
        raise ValueError("source route manifest schema drifted")
    raw_routes = route_payload.get("routes")
    if not isinstance(raw_routes, list):
        raise TypeError("source route manifest routes must be a list")
    routes: dict[str, Mapping[str, Any]] = {}
    for route in raw_routes:
        if not isinstance(route, dict) or not isinstance(
            route.get("source_family"), str
        ):
            raise TypeError("source route row must name its source family")
        source_family = route["source_family"]
        if source_family in routes:
            raise ValueError("source route manifest has duplicate source families")
        routes[source_family] = route

    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise TypeError("release card evidence cards must be a list")
    evidence: dict[str, dict[str, Any]] = {}
    for index, raw_card in enumerate(raw_cards):
        if not isinstance(raw_card, dict):
            raise TypeError("release card evidence row must be an object")
        _validate_card_evidence_row(raw_card, index=index)
        evidence_id = raw_card["evidence_id"]
        if evidence_id in evidence:
            raise ValueError("release card evidence has duplicate evidence IDs")
        if raw_card["evidence_kind"] == "committed_execution_route":
            route = routes.get(raw_card["route_source_family"])
            if route is None:
                raise ValueError("card evidence references an absent source route")
            if (
                route.get("repository") != raw_card["repository"]
                or route.get("revision") != raw_card["revision"]
                or route.get("card_url") != raw_card["url"]
                or route.get("card_sha256") != raw_card["content_sha256"]
                or route.get("declared_license") != "odc-by"
            ):
                raise ValueError("card evidence differs from its committed source route")
        evidence[evidence_id] = dict(raw_card)

    expected_ids = tuple(
        evidence_id
        for source_family in _SOURCE_FAMILIES
        for evidence_id in _CARD_EVIDENCE_IDS_BY_SOURCE[source_family]
    )
    if tuple(evidence) != expected_ids:
        raise ValueError("release card evidence order or coverage drifted")
    return evidence


def verify_release_authority_artifact(path: Path = RELEASE_AUTHORITY_PATH) -> str:
    raw = path.read_bytes()
    if len(raw) != RELEASE_AUTHORITY_BYTES:
        raise RuntimeError("release authority byte count drifted")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RELEASE_AUTHORITY_SHA256:
        raise RuntimeError("release authority SHA-256 drifted")
    return digest


def _validate_card(card: Mapping[str, Any], *, name: str) -> None:
    required = {"evidence_id", "repository", "revision", "url", "content_sha256"}
    optional = {"revision_date_utc"}
    if not required.issubset(card) or not set(card).issubset(required | optional):
        raise ValueError(f"{name} has an invalid card evidence shape")
    if not isinstance(card["repository"], str) or "/" not in card["repository"]:
        raise ValueError(f"{name} repository must be an owner/name pair")
    if not isinstance(card["evidence_id"], str) or not card["evidence_id"]:
        raise ValueError(f"{name} evidence ID must be nonempty")
    _require_sha(card["revision"], width=40, name=f"{name} revision")
    _require_sha(card["content_sha256"], width=64, name=f"{name} card SHA-256")
    expected_prefix = (
        f"https://huggingface.co/datasets/{card['repository']}/blob/"
        f"{card['revision']}/"
    )
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
            "language_id",
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

    language_id = payload["language_id"]
    if not isinstance(language_id, dict):
        raise TypeError("language-ID release binding must be an object")
    _require_exact_keys(
        language_id,
        {
            "decision",
            "binding_receipt_sha256",
            "package",
            "package_version",
            "adapter",
            "model_bytes",
            "model_sha256",
            "scope",
        },
        "language-ID release binding",
    )
    from training.weft1_corpus_a2 import A2_LANGUAGE_ID_BINDING

    expected_language_id = {
        "decision": "RETAIN_FASTTEXT_NO_SWAP",
        "binding_receipt_sha256": A2_LANGUAGE_ID_BINDING.receipt_sha256,
        "package": A2_LANGUAGE_ID_BINDING.package,
        "package_version": A2_LANGUAGE_ID_BINDING.package_version,
        "adapter": A2_LANGUAGE_ID_BINDING.adapter,
        "model_bytes": A2_LANGUAGE_ID_BINDING.model_bytes,
        "model_sha256": A2_LANGUAGE_ID_BINDING.model_sha256,
        "scope": A2_LANGUAGE_ID_BINDING.scope,
    }
    if language_id != expected_language_id:
        raise ValueError("fastText retention binding drifted")

    public_release = payload["public_release"]
    if not isinstance(public_release, dict):
        raise TypeError("public release posture must be an object")
    _require_exact_keys(
        public_release,
        {
            "weights_license_spdx",
            "publish",
            "never_publish",
            "public_corpus_identity",
            "reproducibility_claim",
        },
        "public release posture",
    )
    if public_release.get("weights_license_spdx") != WEIGHTS_LICENSE_SPDX:
        raise ValueError("weights license must remain Apache-2.0")
    if tuple(public_release.get("publish", ())) != _PUBLIC_ARTIFACTS:
        raise ValueError("public corpus artifact allowlist drifted")
    if public_release.get("never_publish") != ["raw_text_shards"]:
        raise ValueError("raw text shards must remain prohibited from publication")
    if public_release.get("public_corpus_identity") != "manifest_sha256":
        raise ValueError("the manifest SHA must remain the public corpus identity")
    if public_release.get("reproducibility_claim") != REPRODUCIBILITY_CLAIM:
        raise ValueError("the D1 replay reproducibility claim drifted")

    claims = payload["claims"]
    if not isinstance(claims, dict):
        raise TypeError("claim controls must be an object")
    _require_exact_keys(
        claims,
        {
            "required_provenance_sentence",
            "matched_comparison_rule",
            "no_named_public_model_comparisons_rule",
            "named_public_model_comparisons",
        },
        "claim controls",
    )
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
    if claims.get("named_public_model_comparisons") != []:
        raise ValueError("named public-model comparison ledger must remain empty")

    attributions = payload["attributions"]
    if not isinstance(attributions, list):
        raise TypeError("attributions must be a list")
    if tuple(row.get("source_family") for row in attributions) != _SOURCE_FAMILIES:
        raise ValueError("attribution source order or coverage drifted")
    evidence = load_release_card_evidence()
    seen_evidence_ids: list[str] = []
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
            evidence_id = card["evidence_id"]
            evidence_card = evidence.get(evidence_id)
            if evidence_card is None:
                raise ValueError("attribution references absent pinned card evidence")
            expected_card = {
                key: evidence_card[key]
                for key in _RELEASE_MANIFEST_CARD_KEYS
                if key in evidence_card
            }
            observed_card = {
                key: card[key]
                for key in _RELEASE_MANIFEST_CARD_KEYS
                if key in card
            }
            if (
                evidence_card["source_family"] != row["source_family"]
                or observed_card != expected_card
            ):
                raise ValueError("attribution card differs from pinned card evidence")
            seen_evidence_ids.append(evidence_id)
        if tuple(seen_evidence_ids[-len(cards) :]) != _CARD_EVIDENCE_IDS_BY_SOURCE[
            row["source_family"]
        ]:
            raise ValueError("attribution card evidence order or coverage drifted")
    if tuple(seen_evidence_ids) != tuple(evidence):
        raise ValueError("attributions do not consume the complete card evidence set")

    disclosure = payload["stackedu_disclosure"]
    if not isinstance(disclosure, dict):
        raise TypeError("StackEdu disclosure must be an object")
    _require_exact_keys(
        disclosure,
        {
            "exact_text",
            "execution_repository",
            "execution_revision",
            "execution_revision_date_utc",
            "software_heritage_graph_snapshot_date",
            "github_archive_metadata_through_date",
            "starcoder2_opt_out_cutoff",
            "upstream_removal_posture",
        },
        "StackEdu disclosure",
    )
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
    manifest_attributions = []
    for row in payload["attributions"]:
        manifest_attributions.append(
            {
                "source_family": row["source_family"],
                "exact_text": row["exact_text"],
                "cards": [
                    {
                        key: card[key]
                        for key in _RELEASE_MANIFEST_CARD_KEYS
                        if key in card
                    }
                    for card in row["cards"]
                ],
            }
        )
    return deepcopy(
        {
            "schema": payload["schema"],
            "public_release": payload["public_release"],
            "claims": {
                key: payload["claims"][key]
                for key in _RELEASE_MANIFEST_CLAIM_KEYS
            },
            "attributions": manifest_attributions,
            "stackedu_disclosure": payload["stackedu_disclosure"],
        }
    )


def verify_model_card_template(path: Path = MODEL_CARD_TEMPLATE_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    payload = load_release_bindings()
    required = [
        REQUIRED_PROVENANCE_SENTENCE,
        NO_NAMED_PUBLIC_MODEL_COMPARISONS_RULE,
        NAMED_PUBLIC_MODEL_COMPARISON_STATUS,
        FASTTEXT_RETENTION_SENTENCE,
        *(row["exact_text"] for row in payload["attributions"]),
        payload["stackedu_disclosure"]["exact_text"],
        "license: apache-2.0",
        "Raw text shards are never published.",
    ]
    missing = [value for value in required if text.count(value) != 1]
    if missing:
        raise RuntimeError("model-card template is missing or duplicates a release binding")
    _reject_named_public_model_comparisons(text)
