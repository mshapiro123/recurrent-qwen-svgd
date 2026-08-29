"""Forward-only Amendment-A3 corpus-route authority for WEFT-1.

A1 and A2 receipts are banked and immutable.  A3 therefore appends a V4 hash
domain and overlays only the two declarations whose first live enumeration
diverged from the A1 hypothesis.  The other five routes are exact passthroughs.

The checked-in ledger is intentionally pending until the two durable breakdown
artifacts exist.  Pending values are represented by JSON ``null`` rather than
invented hashes.  Production loading fails closed while any placeholder remains;
the explicit template loader exists only so the pending contract can be tested
and finalized after live enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from training.weft1_corpus_a2 import (
    A2_CAMPAIGN_ROOT_SEED,
    GTOK_AMENDMENT_A2_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
)
from training.weft1_gtok_a1_contract import (
    SOURCE_FAMILIES,
    SOURCE_ROUTE_MANIFEST_PATH,
    SourceRouteBindingV2,
    SourceRouteManifestV2,
)
from training.weft1_gtok_contract import canonical_sha256
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
A3_AUTHORITY_PATH = (
    REPOSITORY_ROOT / "docs" / "STRATEGY_CORPUS_GTOK_AMENDMENT_A3_20260829.md"
)
A3_AUTHORITY_SHA256 = (
    "4e7b18ec676c6d613c7a0f85ece4c7b8fcc1daab48d5ce0b8cd11bc06875b6c0"
)
A2_BINDINGS_PATH = Path(__file__).with_name(
    "weft1_corpus_gtok_a2_bindings_20260828.json"
)
A2_BINDINGS_SHA256 = (
    "ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b"
)
A1_ROUTE_MANIFEST_PHYSICAL_SHA256 = (
    "1cf99ea33b72013f4bf07101aad8c9b5124879afe3de9f28991e6427ea861a6c"
)
A1_ROUTE_MANIFEST_RECEIPT_SHA256 = (
    "8455b63f8b0dde7f5a5bdb599bec7563ce2b8c9159a26b09f6302e6e326bb663"
)
A3_EFFECTIVE_ROUTE_OVERLAY_PATH = Path(__file__).with_name(
    "weft1_corpus_effective_routes_a3_20260829.json"
)
# Filled after the canonical template file is written.  A live A3 finalization
# changes this pin together with the same new ledger; no A1/A2 file changes.
A3_EFFECTIVE_ROUTE_OVERLAY_SHA256 = (
    "4cf68f276429b15f929ff54b911e7fadab3c67bc0377ae1e0868048e053053d1"
)

GTOK_EXECUTION_AUTHORITY_CHAIN_V4 = (
    *GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
    A3_AUTHORITY_SHA256,
)
A3_CAMPAIGN_ROOT_SEED = A2_CAMPAIGN_ROOT_SEED
A3_CAMPAIGN_SEED_RULE = "PRESERVE_A2_CAMPAIGN_ROOT_SEED_EXACTLY"

OVERLAY_SCHEMA_V4 = "weft1_corpus_effective_route_overlay_a3_v4"
OVERLAY_MANIFEST_IDENTITY_SCHEMA_V4 = (
    "weft1_corpus_effective_route_overlay_manifest_v4"
)
EFFECTIVE_ROUTE_SCHEMA_V4 = "weft1_corpus_effective_source_route_v4"
EFFECTIVE_ROUTES_SCHEMA_V4 = "weft1_corpus_effective_routes_v4"
BREAKDOWN_FAMILY_PROJECTION_SCHEMA_V4 = (
    "weft1_corpus_breakdown_family_projection_a3_v4"
)
PENDING_STATUS = "AWAITING_BREAKDOWN_ARTIFACTS"
RESOLVED_STATUS = "RESOLVED"
PASSTHROUGH_MODE = "PASSTHROUGH_A1"
OVERLAY_MODE = "BREAKDOWN_OVERLAY"
PASSTHROUGH_RESOLUTION = "PASSTHROUGH_A1_UNCHANGED"
OVERLAY_FAMILIES = ("dolma_web", "fineweb_edu")

_A2_BINDINGS_REPO_PATH = "training/weft1_corpus_gtok_a2_bindings_20260828.json"
_A1_ROUTES_REPO_PATH = "training/weft1_gtok_source_routes_20260828.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOLMA_DEFINITION = "Dolma 3 web, top quality bucket"
_FINEWEB_DEFINITION = (
    "FineWeb-Edu, main data, all CC dumps; exclude sample-* and score-variant configs"
)
_FAMILY_DEFINITIONS = {
    "dolma_web": _DOLMA_DEFINITION,
    "fineweb_edu": _FINEWEB_DEFINITION,
}
_DOLMA_RESOLUTIONS = {
    "NARROW_TO_TOP_QUALITY_BUCKET",
    "CONFIRM_TOP_BUCKET_SELECTOR_REMINT_DECLARATION",
}
_FINEWEB_RESOLUTIONS = {
    "ACCEPT_OBSERVED_MAIN_DATA_ALL_CC_DUMPS",
    "WIDEN_TO_ALL_MAIN_DATA_CC_DUMPS",
}


class A3RouteError(RuntimeError):
    """An A3 authority, declaration, or artifact boundary failed closed."""


class A3BreakdownPending(A3RouteError):
    """The checked-in A3 overlay still contains honest pending placeholders."""


class A3StrategyEscalationRequired(A3RouteError):
    """A3's explicitly reserved design branch was reached."""


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _require_exact_mapping(
    value: object,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} fields drifted")
    return value


def _canonical_relative_path(value: object, name: str) -> str:
    text = _require_nonempty(value, name)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be canonical relative POSIX")
    return text


def execution_authority_v4_bound_sha256(schema: str, value: object) -> str:
    """Hash one A3-era receipt without re-keying V1/V2/V3 receipts."""

    _require_nonempty(schema, "receipt schema")
    if not schema.endswith("_v4"):
        raise ValueError("A3 execution receipts require an explicit v4 schema")
    return canonical_sha256(
        {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V4,
            "payload": value,
            "schema": schema,
        }
    )


@dataclass(frozen=True)
class A3Predecessors:
    a2_authority_sha256: str
    a2_bindings_path: str
    a2_bindings_sha256: str
    a1_route_manifest_path: str
    a1_route_manifest_physical_sha256: str
    a1_route_manifest_receipt_sha256: str

    def __post_init__(self) -> None:
        expected = (
            GTOK_AMENDMENT_A2_SHA256,
            _A2_BINDINGS_REPO_PATH,
            A2_BINDINGS_SHA256,
            _A1_ROUTES_REPO_PATH,
            A1_ROUTE_MANIFEST_PHYSICAL_SHA256,
            A1_ROUTE_MANIFEST_RECEIPT_SHA256,
        )
        observed = (
            self.a2_authority_sha256,
            self.a2_bindings_path,
            self.a2_bindings_sha256,
            self.a1_route_manifest_path,
            self.a1_route_manifest_physical_sha256,
            self.a1_route_manifest_receipt_sha256,
        )
        if observed != expected:
            raise ValueError("A3 predecessor authority chain drifted")

    @classmethod
    def from_mapping(cls, value: object) -> "A3Predecessors":
        keys = {
            "a2_authority_sha256",
            "a2_bindings_path",
            "a2_bindings_sha256",
            "a1_route_manifest_path",
            "a1_route_manifest_physical_sha256",
            "a1_route_manifest_receipt_sha256",
        }
        return cls(**dict(_require_exact_mapping(value, keys, "A3 predecessors")))


@dataclass(frozen=True)
class A3CampaignSeedPolicy:
    rule: str
    campaign_root_seed: int

    def __post_init__(self) -> None:
        if (
            self.rule != A3_CAMPAIGN_SEED_RULE
            or type(self.campaign_root_seed) is not int
            or self.campaign_root_seed != A2_CAMPAIGN_ROOT_SEED
        ):
            raise ValueError("A3 must preserve the exact A2 campaign root seed")

    @classmethod
    def from_mapping(cls, value: object) -> "A3CampaignSeedPolicy":
        return cls(
            **dict(
                _require_exact_mapping(
                    value,
                    {"rule", "campaign_root_seed"},
                    "A3 campaign seed policy",
                )
            )
        )


@dataclass(frozen=True)
class A3BreakdownArtifactBinding:
    relative_path: str | None
    physical_bytes: int | None
    physical_sha256: str | None
    typed_receipt_sha256: str | None

    def __post_init__(self) -> None:
        values = (
            self.relative_path,
            self.physical_bytes,
            self.physical_sha256,
            self.typed_receipt_sha256,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("A3 breakdown binding may not be partially populated")
        _canonical_relative_path(self.relative_path, "breakdown artifact path")
        _require_positive_int(self.physical_bytes, "breakdown artifact bytes")
        _require_sha256(self.physical_sha256, "breakdown physical SHA-256")
        _require_sha256(self.typed_receipt_sha256, "breakdown typed receipt SHA-256")

    @property
    def is_bound(self) -> bool:
        return self.relative_path is not None

    @classmethod
    def from_mapping(cls, value: object) -> "A3BreakdownArtifactBinding":
        keys = {
            "relative_path",
            "physical_bytes",
            "physical_sha256",
            "typed_receipt_sha256",
        }
        return cls(
            **dict(_require_exact_mapping(value, keys, "A3 breakdown artifact binding"))
        )


@dataclass(frozen=True)
class A3EffectiveDeclaration:
    asset_selector: str | None
    asset_count: int | None
    available_bytes: int | None
    available_bytes_basis: str | None
    resolution: str | None

    def __post_init__(self) -> None:
        values = (
            self.asset_selector,
            self.asset_count,
            self.available_bytes,
            self.available_bytes_basis,
            self.resolution,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("A3 effective declaration may not be partially populated")
        _require_nonempty(self.asset_selector, "effective asset selector")
        _require_positive_int(self.asset_count, "effective asset count")
        _require_positive_int(self.available_bytes, "effective available bytes")
        _require_nonempty(self.available_bytes_basis, "effective byte basis")
        _require_nonempty(self.resolution, "effective declaration resolution")

    @property
    def is_bound(self) -> bool:
        return self.asset_selector is not None

    @classmethod
    def from_mapping(cls, value: object) -> "A3EffectiveDeclaration":
        keys = {
            "asset_selector",
            "asset_count",
            "available_bytes",
            "available_bytes_basis",
            "resolution",
        }
        return cls(
            **dict(_require_exact_mapping(value, keys, "A3 effective declaration"))
        )


@dataclass(frozen=True)
class A3RouteOverlayRow:
    source_family: str
    mode: str
    base_route_receipt_sha256: str
    family_definition: str | None
    breakdown_artifact: A3BreakdownArtifactBinding | None
    family_projection_sha256: str | None
    effective_declaration: A3EffectiveDeclaration

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("A3 overlay contains an unknown source family")
        _require_sha256(
            self.base_route_receipt_sha256,
            "A3 base-route receipt SHA-256",
        )
        if not isinstance(self.effective_declaration, A3EffectiveDeclaration):
            raise TypeError("A3 row requires a typed effective declaration")
        if self.source_family in OVERLAY_FAMILIES:
            if (
                self.mode != OVERLAY_MODE
                or self.family_definition != _FAMILY_DEFINITIONS[self.source_family]
                or not isinstance(
                    self.breakdown_artifact, A3BreakdownArtifactBinding
                )
                or self.breakdown_artifact.is_bound
                != self.effective_declaration.is_bound
                or (self.family_projection_sha256 is not None)
                != self.effective_declaration.is_bound
            ):
                raise ValueError("A3 breakdown-overlay row drifted")
            if self.family_projection_sha256 is not None:
                _require_sha256(
                    self.family_projection_sha256,
                    "A3 family-projection SHA-256",
                )
                allowed = (
                    _DOLMA_RESOLUTIONS
                    if self.source_family == "dolma_web"
                    else _FINEWEB_RESOLUTIONS
                )
                if self.effective_declaration.resolution not in allowed:
                    raise ValueError("A3 effective declaration resolution drifted")
        elif (
            self.mode != PASSTHROUGH_MODE
            or self.family_definition is not None
            or self.breakdown_artifact is not None
            or self.family_projection_sha256 is not None
            or not self.effective_declaration.is_bound
            or self.effective_declaration.resolution != PASSTHROUGH_RESOLUTION
        ):
            raise ValueError("A3 passthrough row drifted")

    @classmethod
    def from_mapping(cls, value: object) -> "A3RouteOverlayRow":
        keys = {
            "source_family",
            "mode",
            "base_route_receipt_sha256",
            "family_definition",
            "breakdown_artifact",
            "family_projection_sha256",
            "effective_declaration",
        }
        row = dict(_require_exact_mapping(value, keys, "A3 overlay row"))
        raw_breakdown = row.pop("breakdown_artifact")
        return cls(
            breakdown_artifact=(
                None
                if raw_breakdown is None
                else A3BreakdownArtifactBinding.from_mapping(raw_breakdown)
            ),
            effective_declaration=A3EffectiveDeclaration.from_mapping(
                row.pop("effective_declaration")
            ),
            **row,
        )


@dataclass(frozen=True)
class A3EffectiveRouteOverlayManifest:
    schema: str
    status: str
    authority_sha256: str
    predecessors: A3Predecessors
    campaign_seed_policy: A3CampaignSeedPolicy
    overlay_rows: tuple[A3RouteOverlayRow, ...]
    claimed_effective_route_identity_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema != OVERLAY_SCHEMA_V4:
            raise ValueError("unexpected A3 effective-route overlay schema")
        if self.authority_sha256 != A3_AUTHORITY_SHA256:
            raise ValueError("A3 overlay is not bound to the verified authority")
        if not isinstance(self.predecessors, A3Predecessors):
            raise TypeError("A3 overlay requires typed predecessors")
        if not isinstance(self.campaign_seed_policy, A3CampaignSeedPolicy):
            raise TypeError("A3 overlay requires a typed campaign-seed policy")
        if tuple(row.source_family for row in self.overlay_rows) != SOURCE_FAMILIES:
            raise ValueError("A3 overlay must preserve the canonical family order")
        pending = tuple(
            row.source_family
            for row in self.overlay_rows
            if row.mode == OVERLAY_MODE and not row.effective_declaration.is_bound
        )
        bindings = tuple(
            row.breakdown_artifact
            for row in self.overlay_rows
            if row.mode == OVERLAY_MODE
        )
        if len(bindings) != 2 or bindings[0] != bindings[1]:
            raise ValueError(
                "both A3 overlay rows must bind the same combined breakdown artifact"
            )
        if self.status == PENDING_STATUS:
            if pending != OVERLAY_FAMILIES or self.claimed_effective_route_identity_sha256 is not None:
                raise ValueError("pending A3 overlay placeholders drifted")
        elif self.status == RESOLVED_STATUS:
            if pending:
                raise ValueError("resolved A3 overlay retains pending declarations")
            _require_sha256(
                self.claimed_effective_route_identity_sha256,
                "claimed effective-route identity",
            )
        else:
            raise ValueError("A3 overlay status is unknown")

    @property
    def is_resolved(self) -> bool:
        return self.status == RESOLVED_STATUS

    @property
    def overlay_identity_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            OVERLAY_MANIFEST_IDENTITY_SCHEMA_V4,
            self,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "A3EffectiveRouteOverlayManifest":
        keys = {
            "schema",
            "status",
            "authority_sha256",
            "predecessors",
            "campaign_seed_policy",
            "overlay_rows",
            "claimed_effective_route_identity_sha256",
        }
        payload = dict(_require_exact_mapping(value, keys, "A3 overlay manifest"))
        raw_rows = payload.pop("overlay_rows")
        if not isinstance(raw_rows, list):
            raise TypeError("A3 overlay rows must be a JSON list")
        return cls(
            predecessors=A3Predecessors.from_mapping(payload.pop("predecessors")),
            campaign_seed_policy=A3CampaignSeedPolicy.from_mapping(
                payload.pop("campaign_seed_policy")
            ),
            overlay_rows=tuple(A3RouteOverlayRow.from_mapping(row) for row in raw_rows),
            **payload,
        )


@dataclass(frozen=True)
class A3BreakdownFamilyProjection:
    """Compact route projection mechanically derived from the combined observer."""

    source_family: str
    combined_breakdown_receipt_sha256: str
    repository: str
    revision: str
    base_route_receipt_sha256: str
    family_definition: str
    family_definition_id: str
    semantic_evidence_sha256: str
    configured_group_ids_sha256: str
    repository_member_set_sha256: str
    selected_group_ids: tuple[str, ...]
    selected_path_patterns: tuple[str, ...]
    selected_group_receipt_sha256s: tuple[str, ...]
    selected_member_set_sha256: str
    effective_asset_selector: str
    asset_count: int
    available_bytes: int
    available_bytes_basis: str
    resolution: str

    def __post_init__(self) -> None:
        if self.source_family not in OVERLAY_FAMILIES:
            raise ValueError("A3 projection uses an unchanged source family")
        for name in (
            "repository",
            "revision",
            "family_definition",
            "family_definition_id",
            "effective_asset_selector",
            "available_bytes_basis",
            "resolution",
        ):
            _require_nonempty(getattr(self, name), f"A3 projection {name}")
        for name in (
            "combined_breakdown_receipt_sha256",
            "base_route_receipt_sha256",
            "selected_member_set_sha256",
            "semantic_evidence_sha256",
            "configured_group_ids_sha256",
            "repository_member_set_sha256",
        ):
            _require_sha256(getattr(self, name), f"A3 projection {name}")
        _require_positive_int(self.asset_count, "A3 projection asset count")
        _require_positive_int(self.available_bytes, "A3 projection bytes")
        lengths = {
            len(self.selected_group_ids),
            len(self.selected_path_patterns),
            len(self.selected_group_receipt_sha256s),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("A3 projection selected-group vectors drifted")
        keys = tuple(
            zip(
                self.selected_path_patterns,
                self.selected_group_ids,
                self.selected_group_receipt_sha256s,
            )
        )
        if keys != tuple(sorted(keys, key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")))):
            raise ValueError("A3 projection groups are not in canonical order")
        if len(set(self.selected_path_patterns)) != len(self.selected_path_patterns):
            raise ValueError("A3 projection repeats a selected path pattern")
        for receipt in self.selected_group_receipt_sha256s:
            _require_sha256(receipt, "A3 projection group receipt")
        if self.source_family == "fineweb_edu" and len(self.selected_group_ids) != 110:
            raise ValueError("A3 FineWeb projection must retain all 110 main dumps")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(
            BREAKDOWN_FAMILY_PROJECTION_SCHEMA_V4,
            self,
        )


@dataclass(frozen=True)
class EffectiveSourceRouteA3:
    source_family: str
    stratum: str
    role: str
    repository: str
    config: str
    revision: str
    split: str
    asset_selector: str
    selection_rule: str
    declared_license: str
    card_url: str
    card_sha256: str
    asset_count: int
    available_bytes: int
    available_bytes_basis: str
    required_bytes: int
    lineage_evidence: str
    parse_policy: str
    external_locator_manifest_sha256: str | None
    base_route_receipt_sha256: str
    route_resolution: str
    breakdown_artifact_receipt_sha256: str | None
    breakdown_family_projection_sha256: str | None

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError("effective A3 route uses an unknown family")
        for name in (
            "stratum",
            "role",
            "repository",
            "config",
            "revision",
            "split",
            "asset_selector",
            "selection_rule",
            "declared_license",
            "card_url",
            "card_sha256",
            "available_bytes_basis",
            "lineage_evidence",
            "parse_policy",
            "route_resolution",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_positive_int(self.asset_count, "effective route asset count")
        _require_positive_int(self.available_bytes, "effective route available bytes")
        if self.available_bytes <= self.required_bytes:
            raise ValueError("effective A3 route has no positive byte margin")
        _require_sha256(self.base_route_receipt_sha256, "base route receipt")
        evidence = (
            self.breakdown_artifact_receipt_sha256,
            self.breakdown_family_projection_sha256,
        )
        if any(value is None for value in evidence) != all(
            value is None for value in evidence
        ):
            raise ValueError("effective A3 route has a partial breakdown binding")
        for value in evidence:
            if value is not None:
                _require_sha256(value, "effective-route breakdown receipt")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(EFFECTIVE_ROUTE_SCHEMA_V4, self)


@dataclass(frozen=True)
class A3EffectiveRouteResolution:
    routes: tuple[EffectiveSourceRouteA3, ...]
    effective_route_identity_sha256: str
    breakdown_artifact_physical_sha256: str
    breakdown_artifact_receipt_sha256: str
    family_projection_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if tuple(route.source_family for route in self.routes) != SOURCE_FAMILIES:
            raise ValueError("resolved A3 routes are not in canonical family order")
        _require_sha256(
            self.effective_route_identity_sha256,
            "effective route-set identity",
        )
        _require_sha256(
            self.breakdown_artifact_physical_sha256,
            "A3 breakdown artifact physical SHA-256",
        )
        _require_sha256(
            self.breakdown_artifact_receipt_sha256,
            "A3 breakdown artifact typed receipt",
        )
        if tuple(family for family, _ in self.family_projection_sha256s) != OVERLAY_FAMILIES:
            raise ValueError("A3 resolution lacks both breakdown receipt identities")
        for _, receipt in self.family_projection_sha256s:
            _require_sha256(receipt, "breakdown receipt identity")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_a3_authority_artifact(path: Path = A3_AUTHORITY_PATH) -> str:
    if not isinstance(path, Path):
        raise TypeError("A3 authority path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    raw = path.read_bytes()
    observed = _sha256_bytes(raw)
    if observed != A3_AUTHORITY_SHA256:
        raise A3RouteError("A3 authority artifact differs from its ratified SHA-256")
    return observed


def _load_a1_route_snapshot() -> tuple[SourceRouteManifestV2, bytes]:
    raw, payload = load_canonical_json_snapshot(SOURCE_ROUTE_MANIFEST_PATH)
    if _sha256_bytes(raw) != A1_ROUTE_MANIFEST_PHYSICAL_SHA256:
        raise A3RouteError("A1 source-route physical bytes drifted")
    value = dict(payload)
    raw_routes = value.pop("routes", None)
    raw_findings = value.pop("known_route_findings", None)
    if not isinstance(raw_routes, list) or not isinstance(raw_findings, list):
        raise A3RouteError("A1 source-route payload shape drifted")
    manifest = SourceRouteManifestV2(
        routes=tuple(SourceRouteBindingV2.from_mapping(row) for row in raw_routes),
        known_route_findings=tuple(raw_findings),
        **value,
    )
    if manifest.manifest_sha256 != A1_ROUTE_MANIFEST_RECEIPT_SHA256:
        raise A3RouteError("A1 source-route semantic receipt drifted")
    return manifest, raw


def _verify_a2_bindings_snapshot() -> None:
    raw, payload = load_canonical_json_snapshot(A2_BINDINGS_PATH)
    if _sha256_bytes(raw) != A2_BINDINGS_SHA256:
        raise A3RouteError("A2 bindings physical bytes drifted")
    if (
        payload.get("schema") != "weft1_corpus_gtok_a2_bindings_v3"
        or payload.get("authority_sha256") != GTOK_AMENDMENT_A2_SHA256
        or payload.get("a1_route_manifest_receipt_sha256")
        != A1_ROUTE_MANIFEST_RECEIPT_SHA256
        or not isinstance(payload.get("rng"), Mapping)
        or payload["rng"].get("campaign_root_seed") != A2_CAMPAIGN_ROOT_SEED
    ):
        raise A3RouteError("A2 bindings predecessor edges drifted")


def _validate_rows_against_a1(
    manifest: A3EffectiveRouteOverlayManifest,
    base: SourceRouteManifestV2,
) -> None:
    by_family = {route.source_family: route for route in base.routes}
    for row in manifest.overlay_rows:
        route = by_family[row.source_family]
        if row.base_route_receipt_sha256 != route.receipt_sha256:
            raise A3RouteError(f"A3 base-route receipt drifted for {row.source_family}")
        if row.mode == PASSTHROUGH_MODE:
            declaration = row.effective_declaration
            expected = (
                route.asset_selector,
                route.asset_count,
                route.available_bytes,
                route.available_bytes_basis,
                PASSTHROUGH_RESOLUTION,
            )
            observed = (
                declaration.asset_selector,
                declaration.asset_count,
                declaration.available_bytes,
                declaration.available_bytes_basis,
                declaration.resolution,
            )
            if observed != expected:
                raise A3RouteError(
                    f"A3 passthrough declaration drifted for {row.source_family}"
                )


def load_effective_route_overlay_a3(
    path: Path | None = None,
    *,
    allow_pending_template: bool = False,
    nonproduction_fixture: bool = False,
) -> A3EffectiveRouteOverlayManifest:
    """Load A3's overlay; production rejects pending or alternate ledgers."""

    if type(allow_pending_template) is not bool or type(nonproduction_fixture) is not bool:
        raise TypeError("A3 loader flags must be exact bools")
    if path is None and nonproduction_fixture:
        raise ValueError("the checked-in A3 ledger may not use fixture mode")
    if path is not None and not nonproduction_fixture:
        raise ValueError("alternate A3 ledgers require explicit nonproduction fixture mode")
    production_load = path is None
    selected = A3_EFFECTIVE_ROUTE_OVERLAY_PATH if production_load else path
    if not isinstance(selected, Path):
        raise TypeError("A3 overlay path must be a pathlib.Path")
    raw, payload = load_canonical_json_snapshot(selected)
    if production_load and _sha256_bytes(raw) != A3_EFFECTIVE_ROUTE_OVERLAY_SHA256:
        raise A3RouteError("checked-in A3 overlay physical SHA-256 drifted")
    manifest = A3EffectiveRouteOverlayManifest.from_mapping(payload)
    verify_a3_authority_artifact()
    base, _ = _load_a1_route_snapshot()
    _verify_a2_bindings_snapshot()
    _validate_rows_against_a1(manifest, base)
    if not manifest.is_resolved and not allow_pending_template:
        raise A3BreakdownPending(
            "A3 effective routes cannot be consumed before both breakdowns are bound"
        )
    return manifest


def load_effective_route_overlay_template_a3() -> A3EffectiveRouteOverlayManifest:
    """Inspect the pinned pending ledger without granting production authority."""

    return load_effective_route_overlay_a3(allow_pending_template=True)


def _load_combined_breakdown(
    manifest: A3EffectiveRouteOverlayManifest,
    breakdown_root: Path,
) -> tuple[object, A3BreakdownArtifactBinding]:
    """Load the observer's combined artifact once and verify both identities."""

    bindings = tuple(
        row.breakdown_artifact
        for row in manifest.overlay_rows
        if row.mode == OVERLAY_MODE
    )
    if len(bindings) != 2 or bindings[0] is None or bindings[0] != bindings[1]:
        raise A3RouteError("A3 overlay does not bind one combined breakdown")
    binding = bindings[0]
    if not binding.is_bound:
        raise A3BreakdownPending("A3 combined breakdown artifact is pending")
    if not isinstance(breakdown_root, Path):
        raise TypeError("A3 breakdown root must be a pathlib.Path")
    root = assert_no_symlink_ancestors(breakdown_root).resolve(strict=True)
    if not root.is_dir():
        raise A3RouteError("A3 breakdown root must be a real directory")
    relative = PurePosixPath(
        _canonical_relative_path(binding.relative_path, "breakdown artifact path")
    )
    lexical = root.joinpath(*relative.parts)
    assert_no_symlink_ancestors(lexical)
    path = lexical.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise A3RouteError("A3 breakdown artifact escapes its governed root")

    # Lazy import avoids a module cycle: the observer imports the V4 authority
    # domain from this module, while this execution layer consumes its output.
    from training.weft1_corpus_breakdown_a3 import (
        PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
        PRODUCTION_OBSERVATION_MODE_A3,
        load_upstream_path_breakdown_snapshot_a3,
    )

    raw, receipt = load_upstream_path_breakdown_snapshot_a3(
        path,
        expected_receipt_sha256=binding.typed_receipt_sha256,
    )
    if (
        len(raw) != binding.physical_bytes
        or _sha256_bytes(raw) != binding.physical_sha256
    ):
        raise A3RouteError("A3 breakdown physical identity drifted")
    if receipt.receipt_sha256 != binding.typed_receipt_sha256:
        raise A3RouteError("A3 breakdown typed receipt identity drifted")
    if (
        receipt.observation_mode != PRODUCTION_OBSERVATION_MODE_A3
        or receipt.observation_client_identity
        != PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3
    ):
        raise A3RouteError("A3 execution rejects nonproduction path observation")
    return receipt, binding


def project_family_resolution_a3(
    receipt: object,
    base: SourceRouteBindingV2,
) -> A3BreakdownFamilyProjection:
    """Derive a compact effective declaration from the validated observer."""

    from training.weft1_corpus_breakdown_a3 import (
        UpstreamPathBreakdownReceiptA3,
        project_family_resolution_a3 as project_observer_family_a3,
    )

    if not isinstance(receipt, UpstreamPathBreakdownReceiptA3):
        raise TypeError("A3 family projection requires the typed combined observer")
    if base.source_family not in OVERLAY_FAMILIES:
        raise ValueError("A3 projection requested for an unchanged family")
    family = next(
        item for item in receipt.families if item.source_family == base.source_family
    )
    prior = family.prior_declaration
    if (
        family.repository != base.repository
        or family.revision != base.revision
        or prior.source_family != base.source_family
        or prior.asset_selector != base.asset_selector
        or prior.asset_count != base.asset_count
        or prior.available_bytes != base.available_bytes
        or prior.declaration_receipt_sha256 != base.receipt_sha256
    ):
        raise A3RouteError("A3 observer is not a replay of the bound A1 route")
    selected = tuple(group for group in family.groups if group.selected)
    observed = project_observer_family_a3(receipt, base.source_family)
    projection = A3BreakdownFamilyProjection(
        source_family=base.source_family,
        combined_breakdown_receipt_sha256=receipt.receipt_sha256,
        repository=family.repository,
        revision=family.revision,
        base_route_receipt_sha256=base.receipt_sha256,
        family_definition=_FAMILY_DEFINITIONS[base.source_family],
        family_definition_id=observed.family_definition_id,
        semantic_evidence_sha256=observed.semantic_evidence_sha256,
        configured_group_ids_sha256=observed.configured_group_ids_sha256,
        repository_member_set_sha256=observed.repository_member_set_sha256,
        selected_group_ids=tuple(group.group_id for group in selected),
        selected_path_patterns=tuple(group.path_pattern for group in selected),
        selected_group_receipt_sha256s=tuple(
            group.receipt_sha256 for group in selected
        ),
        selected_member_set_sha256=family.selected_member_set_sha256,
        effective_asset_selector=observed.effective_asset_selector,
        asset_count=observed.selected_asset_count,
        available_bytes=observed.selected_upstream_bytes,
        available_bytes_basis=observed.available_bytes_basis,
        resolution=observed.resolution,
    )
    if projection.available_bytes <= base.required_bytes:
        raise A3RouteError("A3 re-minted declaration has no positive byte margin")
    return projection


def _effective_route(
    base: SourceRouteBindingV2,
    row: A3RouteOverlayRow,
    projection: A3BreakdownFamilyProjection | None,
    breakdown_artifact_receipt_sha256: str | None,
) -> EffectiveSourceRouteA3:
    declaration = row.effective_declaration
    if not declaration.is_bound:
        raise A3BreakdownPending(f"A3 declaration is pending for {row.source_family}")
    return EffectiveSourceRouteA3(
        source_family=base.source_family,
        stratum=base.stratum,
        role=base.role,
        repository=base.repository,
        config=base.config,
        revision=base.revision,
        split=base.split,
        asset_selector=str(declaration.asset_selector),
        selection_rule=base.selection_rule,
        declared_license=base.declared_license,
        card_url=base.card_url,
        card_sha256=base.card_sha256,
        asset_count=int(declaration.asset_count),
        available_bytes=int(declaration.available_bytes),
        available_bytes_basis=str(declaration.available_bytes_basis),
        required_bytes=base.required_bytes,
        lineage_evidence=base.lineage_evidence,
        parse_policy=base.parse_policy,
        external_locator_manifest_sha256=base.external_locator_manifest_sha256,
        base_route_receipt_sha256=base.receipt_sha256,
        route_resolution=str(declaration.resolution),
        breakdown_artifact_receipt_sha256=breakdown_artifact_receipt_sha256,
        breakdown_family_projection_sha256=(
            None if projection is None else projection.receipt_sha256
        ),
    )


def effective_route_identity_sha256(
    routes: Sequence[EffectiveSourceRouteA3],
) -> str:
    if not isinstance(routes, Sequence) or tuple(
        route.source_family for route in routes
    ) != SOURCE_FAMILIES:
        raise ValueError("effective A3 routes must cover every family in order")
    if not all(isinstance(route, EffectiveSourceRouteA3) for route in routes):
        raise TypeError("effective A3 route set must contain typed routes")
    return execution_authority_v4_bound_sha256(EFFECTIVE_ROUTES_SCHEMA_V4, tuple(routes))


def finalize_effective_route_overlay_a3(
    pending: A3EffectiveRouteOverlayManifest,
    *,
    breakdown_path: Path,
) -> A3EffectiveRouteOverlayManifest:
    """Mechanically replace the two honest placeholders from one observer.

    This derives, rather than accepts, every changed declaration and the final
    route-set identity.  It does not write the ledger or authorize downloads;
    callers must persist, pin, reload, and resolve the returned manifest.
    """

    if not isinstance(pending, A3EffectiveRouteOverlayManifest):
        raise TypeError("A3 finalization requires a typed pending overlay")
    if pending.status != PENDING_STATUS or pending.is_resolved:
        raise A3RouteError("A3 finalization requires the exact pending state")
    if not isinstance(breakdown_path, Path):
        raise TypeError("A3 finalization breakdown path must be pathlib.Path")
    root = REPOSITORY_ROOT.resolve(strict=True)
    assert_no_symlink_ancestors(breakdown_path)
    resolved_path = breakdown_path.resolve(strict=True)
    if root not in resolved_path.parents or not resolved_path.is_file():
        raise A3RouteError("A3 finalization breakdown escapes the repository")
    relative_path = resolved_path.relative_to(root).as_posix()

    from training.weft1_corpus_breakdown_a3 import (
        PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3,
        PRODUCTION_OBSERVATION_MODE_A3,
        load_upstream_path_breakdown_snapshot_a3,
    )

    raw, receipt = load_upstream_path_breakdown_snapshot_a3(resolved_path)
    if (
        receipt.observation_mode != PRODUCTION_OBSERVATION_MODE_A3
        or receipt.observation_client_identity
        != PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3
    ):
        raise A3RouteError("A3 finalization rejects nonproduction path observation")
    binding = A3BreakdownArtifactBinding(
        relative_path=relative_path,
        physical_bytes=len(raw),
        physical_sha256=_sha256_bytes(raw),
        typed_receipt_sha256=receipt.receipt_sha256,
    )
    base, _ = _load_a1_route_snapshot()
    _verify_a2_bindings_snapshot()
    _validate_rows_against_a1(pending, base)
    by_family = {route.source_family: route for route in base.routes}
    projections: dict[str, A3BreakdownFamilyProjection] = {}
    rows: list[A3RouteOverlayRow] = []
    for row in pending.overlay_rows:
        if row.mode == PASSTHROUGH_MODE:
            rows.append(row)
            continue
        projection = project_family_resolution_a3(
            receipt,
            by_family[row.source_family],
        )
        projections[row.source_family] = projection
        rows.append(
            A3RouteOverlayRow(
                source_family=row.source_family,
                mode=row.mode,
                base_route_receipt_sha256=row.base_route_receipt_sha256,
                family_definition=row.family_definition,
                breakdown_artifact=binding,
                family_projection_sha256=projection.receipt_sha256,
                effective_declaration=A3EffectiveDeclaration(
                    asset_selector=projection.effective_asset_selector,
                    asset_count=projection.asset_count,
                    available_bytes=projection.available_bytes,
                    available_bytes_basis=projection.available_bytes_basis,
                    resolution=projection.resolution,
                ),
            )
        )
    ordered_rows = tuple(rows)
    routes = tuple(
        _effective_route(
            by_family[row.source_family],
            row,
            projections.get(row.source_family),
            (
                binding.typed_receipt_sha256
                if row.source_family in projections
                else None
            ),
        )
        for row in ordered_rows
    )
    identity = effective_route_identity_sha256(routes)
    finalized = A3EffectiveRouteOverlayManifest(
        schema=pending.schema,
        status=RESOLVED_STATUS,
        authority_sha256=pending.authority_sha256,
        predecessors=pending.predecessors,
        campaign_seed_policy=pending.campaign_seed_policy,
        overlay_rows=ordered_rows,
        claimed_effective_route_identity_sha256=identity,
    )
    resolved = resolve_effective_routes_a3(finalized, breakdown_root=root)
    if resolved.effective_route_identity_sha256 != identity:
        raise A3RouteError("A3 finalized route identity failed self-resolution")
    return finalized


def resolve_effective_routes_a3(
    manifest: A3EffectiveRouteOverlayManifest,
    *,
    breakdown_root: Path,
) -> A3EffectiveRouteResolution:
    """Verify both physical breakdowns and mint only the V4 effective identity."""

    if not isinstance(manifest, A3EffectiveRouteOverlayManifest):
        raise TypeError("A3 resolution requires a typed overlay manifest")
    if not manifest.is_resolved:
        raise A3BreakdownPending("A3 overlay is not resolved")
    base, _ = _load_a1_route_snapshot()
    _verify_a2_bindings_snapshot()
    _validate_rows_against_a1(manifest, base)
    combined_receipt, artifact_binding = _load_combined_breakdown(
        manifest,
        breakdown_root,
    )
    by_family = {route.source_family: route for route in base.routes}
    routes: list[EffectiveSourceRouteA3] = []
    projections: list[tuple[str, str]] = []
    for row in manifest.overlay_rows:
        base_route = by_family[row.source_family]
        projection: A3BreakdownFamilyProjection | None = None
        if row.mode == OVERLAY_MODE:
            projection = project_family_resolution_a3(
                combined_receipt,
                base_route,
            )
            declaration = row.effective_declaration
            if (
                row.family_definition != projection.family_definition
                or row.family_projection_sha256 != projection.receipt_sha256
                or (
                    declaration.asset_selector,
                    declaration.asset_count,
                    declaration.available_bytes,
                    declaration.available_bytes_basis,
                    declaration.resolution,
                )
                != (
                    projection.effective_asset_selector,
                    projection.asset_count,
                    projection.available_bytes,
                    projection.available_bytes_basis,
                    projection.resolution,
                )
            ):
                raise A3RouteError(
                    f"A3 effective declaration is not derived for {row.source_family}"
                )
            projections.append((row.source_family, projection.receipt_sha256))
        routes.append(
            _effective_route(
                base_route,
                row,
                projection,
                (
                    None
                    if projection is None
                    else artifact_binding.typed_receipt_sha256
                ),
            )
        )
    ordered = tuple(routes)
    identity = effective_route_identity_sha256(ordered)
    if identity != manifest.claimed_effective_route_identity_sha256:
        raise A3RouteError("claimed A3 effective-route identity drifted")
    return A3EffectiveRouteResolution(
        routes=ordered,
        effective_route_identity_sha256=identity,
        breakdown_artifact_physical_sha256=str(artifact_binding.physical_sha256),
        breakdown_artifact_receipt_sha256=str(
            artifact_binding.typed_receipt_sha256
        ),
        family_projection_sha256s=tuple(projections),
    )


__all__ = [
    "A1_ROUTE_MANIFEST_PHYSICAL_SHA256",
    "A1_ROUTE_MANIFEST_RECEIPT_SHA256",
    "A2_BINDINGS_SHA256",
    "A3_AUTHORITY_PATH",
    "A3_AUTHORITY_SHA256",
    "A3BreakdownPending",
    "A3BreakdownArtifactBinding",
    "A3CampaignSeedPolicy",
    "A3EffectiveDeclaration",
    "A3EffectiveRouteOverlayManifest",
    "A3EffectiveRouteResolution",
    "A3Predecessors",
    "A3RouteError",
    "A3RouteOverlayRow",
    "A3StrategyEscalationRequired",
    "A3_CAMPAIGN_ROOT_SEED",
    "A3_CAMPAIGN_SEED_RULE",
    "A3_EFFECTIVE_ROUTE_OVERLAY_PATH",
    "A3_EFFECTIVE_ROUTE_OVERLAY_SHA256",
    "A3BreakdownFamilyProjection",
    "BREAKDOWN_FAMILY_PROJECTION_SCHEMA_V4",
    "EFFECTIVE_ROUTES_SCHEMA_V4",
    "EffectiveSourceRouteA3",
    "GTOK_EXECUTION_AUTHORITY_CHAIN_V4",
    "OVERLAY_FAMILIES",
    "OVERLAY_MODE",
    "PASSTHROUGH_MODE",
    "PENDING_STATUS",
    "RESOLVED_STATUS",
    "effective_route_identity_sha256",
    "execution_authority_v4_bound_sha256",
    "finalize_effective_route_overlay_a3",
    "load_effective_route_overlay_a3",
    "load_effective_route_overlay_template_a3",
    "project_family_resolution_a3",
    "resolve_effective_routes_a3",
    "verify_a3_authority_artifact",
]
