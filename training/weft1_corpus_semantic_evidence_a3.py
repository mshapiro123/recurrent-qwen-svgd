"""Typed identity checks for WEFT-1 A3 semantic-evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from training.weft1_corpus_a3 import A3_AUTHORITY_SHA256
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import load_canonical_json_snapshot


SEMANTIC_EVIDENCE_SCHEMA_A3_V1 = "weft1_corpus_semantic_evidence_a3_v1"
SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1 = (
    "weft1_corpus_semantic_evidence_receipt_a3_v1"
)
SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1 = (
    "weft1_corpus_semantic_evidence_family_a3_v1"
)
SEMANTIC_EVIDENCE_RELATIVE_PATH_A3 = (
    "training/weft1_corpus_semantic_evidence_a3_20260829.json"
)
SEMANTIC_EVIDENCE_FAMILIES_A3 = ("dolma_web", "fineweb_edu")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINEWEB_DUMP = re.compile(r"^CC-MAIN-[0-9]{4}-[0-9]{2}$")
_ASSERTIONS = {
    "dolma_web": "dolma3_bucket_0019_is_top_quality",
    "fineweb_edu": "fineweb_edu_main_data_is_all_configured_cc_main_dumps",
}


class SemanticEvidenceA3Error(ValueError):
    """An A3 semantic-evidence artifact or binding is not exact."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticEvidenceA3Error(f"{name} must be a lowercase SHA-256")
    return value


def _exact_mapping(
    value: object,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SemanticEvidenceA3Error(f"{name} shape drifted")
    return value


@dataclass(frozen=True)
class SemanticEvidenceTypedIdentityA3:
    receipt_sha256: str
    family_receipt_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_sha256, "semantic evidence typed identity")
        if tuple(family for family, _ in self.family_receipt_sha256s) != (
            SEMANTIC_EVIDENCE_FAMILIES_A3
        ):
            raise SemanticEvidenceA3Error(
                "semantic evidence lacks both families in canonical order"
            )
        for _, receipt in self.family_receipt_sha256s:
            _require_sha256(receipt, "semantic evidence family identity")


@dataclass(frozen=True)
class SemanticEvidenceArtifactIdentityA3(SemanticEvidenceTypedIdentityA3):
    physical_bytes: int
    physical_sha256: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.physical_bytes) is not int or self.physical_bytes < 1:
            raise SemanticEvidenceA3Error("semantic evidence bytes must be positive")
        _require_sha256(self.physical_sha256, "semantic evidence physical identity")


def validate_semantic_evidence_payload_a3(
    payload: Mapping[str, Any],
) -> SemanticEvidenceTypedIdentityA3:
    row = _exact_mapping(
        payload,
        {"authority_sha256", "families", "receipt_sha256", "schema"},
        "semantic evidence ledger",
    )
    if (
        row["schema"] != SEMANTIC_EVIDENCE_SCHEMA_A3_V1
        or row["authority_sha256"] != A3_AUTHORITY_SHA256
    ):
        raise SemanticEvidenceA3Error("semantic evidence authority drifted")
    families = row["families"]
    if not isinstance(families, list) or tuple(
        item.get("source_family") if isinstance(item, Mapping) else None
        for item in families
    ) != SEMANTIC_EVIDENCE_FAMILIES_A3:
        raise SemanticEvidenceA3Error("semantic evidence family order drifted")
    family_receipts: list[tuple[str, str]] = []
    for value in families:
        family = dict(
            _exact_mapping(
                value,
                {
                    "assertion",
                    "configured_group_ids",
                    "derived_facts",
                    "family_evidence_sha256",
                    "pin",
                    "source_family",
                    "upstream_documents",
                },
                "semantic evidence family",
            )
        )
        for name in ("assertion", "pin", "source_family"):
            if not isinstance(family[name], str) or not family[name]:
                raise SemanticEvidenceA3Error(
                    f"semantic evidence family {name} is invalid"
                )
        for name in ("configured_group_ids", "derived_facts", "upstream_documents"):
            if not isinstance(family[name], list):
                raise SemanticEvidenceA3Error(
                    f"semantic evidence family {name} must be a list"
                )
        source_family = str(family["source_family"])
        if family["assertion"] != _ASSERTIONS[source_family]:
            raise SemanticEvidenceA3Error(
                "semantic evidence family assertion drifted"
            )
        configured = family["configured_group_ids"]
        if source_family == "dolma_web":
            if configured:
                raise SemanticEvidenceA3Error(
                    "Dolma semantic evidence must not configure path groups"
                )
        elif (
            len(configured) != 110
            or any(
                not isinstance(value, str)
                or _FINEWEB_DUMP.fullmatch(value) is None
                for value in configured
            )
            or configured
            != sorted(configured, key=lambda value: value.encode("utf-8"))
            or len(set(configured)) != 110
        ):
            raise SemanticEvidenceA3Error(
                "FineWeb semantic evidence requires exactly 110 canonical dumps"
            )
        if (
            not family["derived_facts"]
            or any(
                not isinstance(value, str) or not value
                for value in family["derived_facts"]
            )
            or not family["upstream_documents"]
        ):
            raise SemanticEvidenceA3Error(
                "semantic evidence family lacks facts or documents"
            )
        for document in family["upstream_documents"]:
            document_row = _exact_mapping(
                document,
                {
                    "bytes",
                    "content_sha256",
                    "path",
                    "repository",
                    "revision",
                    "source_family",
                    "supports",
                    "url",
                },
                "semantic evidence document",
            )
            if type(document_row["bytes"]) is not int or document_row["bytes"] < 1:
                raise SemanticEvidenceA3Error(
                    "semantic evidence document bytes are invalid"
                )
            _require_sha256(
                document_row["content_sha256"],
                "semantic evidence document content",
            )
            if document_row["source_family"] != source_family or any(
                not isinstance(document_row[name], str) or not document_row[name]
                for name in (
                    "path",
                    "repository",
                    "revision",
                    "source_family",
                    "supports",
                    "url",
                )
            ):
                raise SemanticEvidenceA3Error(
                    "semantic evidence document family or locator drifted"
                )
        claimed = family.pop("family_evidence_sha256")
        expected = canonical_sha256(
            {
                "payload": family,
                "schema": SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1,
            }
        )
        if claimed != expected:
            raise SemanticEvidenceA3Error(
                "semantic evidence family receipt drifted"
            )
        family_receipts.append((str(family["source_family"]), expected))
    core = dict(row)
    claimed_receipt = core.pop("receipt_sha256")
    expected_receipt = canonical_sha256(
        {
            "payload": core,
            "schema": SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1,
        }
    )
    if claimed_receipt != expected_receipt:
        raise SemanticEvidenceA3Error("semantic evidence ledger receipt drifted")
    return SemanticEvidenceTypedIdentityA3(
        receipt_sha256=expected_receipt,
        family_receipt_sha256s=tuple(family_receipts),
    )


def load_semantic_evidence_snapshot_a3(
    path: Path,
) -> tuple[bytes, Mapping[str, Any], SemanticEvidenceArtifactIdentityA3]:
    """Single-read canonical loader returning physical and typed identities."""

    try:
        raw, payload = load_canonical_json_snapshot(path)
    except ValueError as error:
        raise SemanticEvidenceA3Error(
            "semantic evidence artifact is not canonical JSON"
        ) from error
    if raw != canonical_json_bytes(payload) + b"\n":
        raise SemanticEvidenceA3Error(
            "semantic evidence artifact is not canonical JSON"
        )
    typed = validate_semantic_evidence_payload_a3(payload)
    identity = SemanticEvidenceArtifactIdentityA3(
        physical_bytes=len(raw),
        physical_sha256=_sha256(raw),
        receipt_sha256=typed.receipt_sha256,
        family_receipt_sha256s=typed.family_receipt_sha256s,
    )
    return raw, payload, identity


def semantic_evidence_relative_path_from_breakdown_a3(
    breakdown: object,
) -> str:
    families = getattr(breakdown, "families", ())
    locators = tuple(
        getattr(getattr(family, "semantic_evidence", None), "locator", None)
        for family in families
    )
    if len(locators) != 2 or any(not isinstance(value, str) for value in locators):
        raise SemanticEvidenceA3Error(
            "breakdown lacks both semantic-evidence locators"
        )
    paths = tuple(str(value).split("#", 1)[0] for value in locators)
    if paths[0] != paths[1] or paths[0] != SEMANTIC_EVIDENCE_RELATIVE_PATH_A3:
        raise SemanticEvidenceA3Error(
            "breakdown semantic-evidence artifact path drifted"
        )
    pure = PurePosixPath(paths[0])
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SemanticEvidenceA3Error(
            "breakdown semantic-evidence path is noncanonical"
        )
    return paths[0]


def verify_semantic_evidence_breakdown_binding_a3(
    payload: Mapping[str, Any],
    identity: SemanticEvidenceArtifactIdentityA3,
    breakdown: object,
) -> None:
    """Require every family object in the breakdown to match the ledger."""

    semantic_evidence_relative_path_from_breakdown_a3(breakdown)
    rows = payload.get("families")
    families = getattr(breakdown, "families", ())
    if not isinstance(rows, list) or len(rows) != len(families) != 2:
        raise SemanticEvidenceA3Error("semantic-evidence breakdown shape drifted")
    for row, family in zip(rows, families, strict=True):
        evidence = getattr(family, "semantic_evidence", None)
        observed = (
            row.get("source_family"),
            row.get("assertion"),
            row.get("pin"),
            row.get("family_evidence_sha256"),
        )
        expected = (
            getattr(family, "source_family", None),
            getattr(evidence, "assertion", None),
            getattr(evidence, "pin", None),
            getattr(evidence, "content_sha256", None),
        )
        if observed != expected:
            raise SemanticEvidenceA3Error(
                "semantic-evidence ledger differs from the path breakdown"
            )
    if identity.family_receipt_sha256s != tuple(
        (family.source_family, family.semantic_evidence.content_sha256)
        for family in families
    ):
        raise SemanticEvidenceA3Error(
            "semantic-evidence family identities differ from the breakdown"
        )


__all__ = [
    "SEMANTIC_EVIDENCE_FAMILY_SCHEMA_A3_V1",
    "SEMANTIC_EVIDENCE_RECEIPT_SCHEMA_A3_V1",
    "SEMANTIC_EVIDENCE_RELATIVE_PATH_A3",
    "SEMANTIC_EVIDENCE_SCHEMA_A3_V1",
    "SemanticEvidenceA3Error",
    "SemanticEvidenceArtifactIdentityA3",
    "SemanticEvidenceTypedIdentityA3",
    "load_semantic_evidence_snapshot_a3",
    "semantic_evidence_relative_path_from_breakdown_a3",
    "validate_semantic_evidence_payload_a3",
    "verify_semantic_evidence_breakdown_binding_a3",
]
