"""Forward-only A3 path-breakdown evidence for WEFT-1 P-A.

This module observes already-pinned repository-tree metadata.  It does not
perform network I/O, download corpus bytes, amend an A1/A2 route, or authorize
materialization.  Its receipt explains which Dolma and FineWeb-Edu path groups
are in-family, preserves excluded groups, and makes the selected totals a
deterministic consequence of exact member metadata.

The A3 surface is deliberately isolated from every banked V1/V2/V3 type.  A
future A3 route ledger may bind one of these receipts, but this observer cannot
mint that ledger itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from training.weft1_corpus_a3 import (
    A3_AUTHORITY_SHA256,
    execution_authority_v4_bound_sha256,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import (
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


PATH_BREAKDOWN_MEMBER_SCHEMA_A3_V1 = "weft1_path_member_a3_v1"
PATH_BREAKDOWN_GROUP_SCHEMA_A3_V1 = "weft1_path_group_a3_v1"
PATH_BREAKDOWN_EVIDENCE_SCHEMA_A3_V1 = "weft1_semantic_evidence_a3_v1"
PATH_BREAKDOWN_CLIENT_IDENTITY_SCHEMA_A3_V1 = (
    "weft1_path_observation_client_identity_a3_v1"
)
PATH_BREAKDOWN_CONFIGURATION_SCHEMA_A3_V1 = (
    "weft1_path_breakdown_configuration_a3_v1"
)
PATH_BREAKDOWN_RECEIPT_SCHEMA_A3_V1 = "weft1_upstream_path_breakdown_a3_v4"
PATH_BREAKDOWN_ARTIFACT_SCHEMA_A3_V1 = (
    "weft1_upstream_path_breakdown_artifact_a3_v4"
)
PATH_BREAKDOWN_STATUS_A3_V1 = "OBSERVED_NOT_EXECUTION_AUTHORITY"

PRODUCTION_OBSERVATION_MODE_A3 = "PINNED_HUGGINGFACE_HUB_1_24_0"
FIXTURE_OBSERVATION_MODE_A3 = "NONAUTHORITATIVE_FIXTURE"
OBSERVATION_MODES_A3 = (
    PRODUCTION_OBSERVATION_MODE_A3,
    FIXTURE_OBSERVATION_MODE_A3,
)

DOLMA_SOURCE_FAMILY = "dolma_web"
FINEWEB_SOURCE_FAMILY = "fineweb_edu"
BREAKDOWN_SOURCE_FAMILIES = (DOLMA_SOURCE_FAMILY, FINEWEB_SOURCE_FAMILY)

DOLMA_DEFINITION_A3_V1 = "dolma3_web_top_quality_bucket_a3_v1"
FINEWEB_DEFINITION_A3_V1 = "fineweb_edu_all_main_cc_dumps_a3_v1"
DOLMA_CANDIDATE_SCOPE_A3_V1 = (
    "data/common_crawl-<topic>-<4digits>/*.jsonl.zst"
)
FINEWEB_CANDIDATE_SCOPE_A3_V1 = "data/<group>/*.parquet"
DOLMA_TOP_QUALITY_ASSERTION_A3_V1 = "dolma3_bucket_0019_is_top_quality"
FINEWEB_MAIN_DATA_ASSERTION_A3_V1 = (
    "fineweb_edu_main_data_is_all_configured_cc_main_dumps"
)

DOLMA_SELECTED_CLASSIFICATION = "IN_FAMILY_TOP_QUALITY_BUCKET"
DOLMA_EXCLUDED_CLASSIFICATION = "OUT_OF_FAMILY_LOWER_QUALITY_BUCKET"
DOLMA_NON_COMMON_CRAWL_CLASSIFICATION = (
    "OUT_OF_FAMILY_NON_COMMON_CRAWL_DATA"
)
REPOSITORY_METADATA_CLASSIFICATION = "OUT_OF_FAMILY_REPOSITORY_METADATA"
FINEWEB_SELECTED_CLASSIFICATION = "IN_FAMILY_MAIN_CC_DUMP"
FINEWEB_SAMPLE_CLASSIFICATION = "OUT_OF_FAMILY_SAMPLE"
FINEWEB_SCORE_CLASSIFICATION = "OUT_OF_FAMILY_SCORE_VARIANT"

DOLMA_EFFECTIVE_SELECTOR_A3_V1 = "data/common_crawl-*-0019/*.jsonl.zst"
FINEWEB_EFFECTIVE_SELECTOR_A3_V1 = (
    "semantic:fineweb_edu_configured_110_cc_main_dumps_all_parquet_a3_v1"
)
DOLMA_CONFIRM_RESOLUTION_A3_V1 = (
    "CONFIRM_TOP_BUCKET_SELECTOR_REMINT_DECLARATION"
)
DOLMA_NARROW_RESOLUTION_A3_V1 = "NARROW_TO_TOP_QUALITY_BUCKET"
FINEWEB_ACCEPT_RESOLUTION_A3_V1 = "ACCEPT_OBSERVED_MAIN_DATA_ALL_CC_DUMPS"
FINEWEB_WIDEN_RESOLUTION_A3_V1 = "WIDEN_TO_ALL_MAIN_DATA_CC_DUMPS"

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOLMA_ASSET = re.compile(
    r"^data/common_crawl-(?P<topic>[^/]+)-(?P<bucket>[0-9]{4})/"
    r"(?P<filename>[^/]+[.]jsonl[.]zst)$"
)
_DOLMA_GROUP_ID = re.compile(
    r"^common_crawl-(?P<topic>[^/]+)-(?P<bucket>[0-9]{4})$"
)
_FINEWEB_PARQUET = re.compile(
    r"^data/(?P<group>[^/]+)/(?P<filename>[^/]+[.]parquet)$"
)
_FINEWEB_ROOT_SAMPLE = re.compile(
    r"^sample/(?P<config>[^/]+)/(?P<relative>.+)$"
)
_FINEWEB_MAIN_DUMP = re.compile(r"^CC-MAIN-[0-9]{4}-[0-9]{2}$")
_FINEWEB_SCORE_TOKEN = re.compile(r"(?:^|[-_])score(?:[-_]|$)", re.IGNORECASE)

_DOLMA_CLASSIFICATIONS = frozenset(
    {
        DOLMA_SELECTED_CLASSIFICATION,
        DOLMA_EXCLUDED_CLASSIFICATION,
        DOLMA_NON_COMMON_CRAWL_CLASSIFICATION,
        REPOSITORY_METADATA_CLASSIFICATION,
    }
)
_FINEWEB_CLASSIFICATIONS = frozenset(
    {
        FINEWEB_SELECTED_CLASSIFICATION,
        FINEWEB_SAMPLE_CLASSIFICATION,
        FINEWEB_SCORE_CLASSIFICATION,
        REPOSITORY_METADATA_CLASSIFICATION,
    }
)
_IDENTITY_LENGTHS = {
    "content_sha256": 64,
    "git_sha1": 40,
    "git_sha256": 64,
}


class PathBreakdownError(ValueError):
    """A3 path evidence is incomplete, ambiguous, or noncanonical."""


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PathBreakdownError(f"{name} must be a nonempty exact string")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PathBreakdownError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise PathBreakdownError(f"{name} must be a positive exact integer")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PathBreakdownError(f"{name} must be a nonnegative exact integer")
    return value


def _canonical_repository_path(value: str) -> str:
    _require_nonempty(value, "repository path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PathBreakdownError(
            "repository path must be canonical relative POSIX"
        )
    return value


def _domain_sha256(schema: str, value: object) -> str:
    _require_nonempty(schema, "hash schema")
    return canonical_sha256({"payload": value, "schema": schema})


def _field(value: object, name: str, *, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class PathObservationClientIdentityA3:
    """Exact implementation and invocation surface used to observe a tree."""

    client_package: str
    client_version: str
    client_api: str
    endpoint: str
    repo_type: str
    recursive: bool
    expand: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.client_package, "observation client package"),
            (self.client_version, "observation client version"),
            (self.client_api, "observation client API"),
            (self.endpoint, "observation client endpoint"),
            (self.repo_type, "observation repository type"),
        ):
            _require_nonempty(value, name)
        if type(self.recursive) is not bool or type(self.expand) is not bool:
            raise TypeError("observation client flags must be exact booleans")

    @property
    def receipt_sha256(self) -> str:
        return _domain_sha256(
            PATH_BREAKDOWN_CLIENT_IDENTITY_SCHEMA_A3_V1,
            self,
        )


PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3 = PathObservationClientIdentityA3(
    client_package="huggingface_hub",
    client_version="1.24.0",
    client_api="HfApi.list_repo_tree",
    endpoint="https://huggingface.co",
    repo_type="dataset",
    recursive=True,
    expand=False,
)
FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3 = PathObservationClientIdentityA3(
    client_package="weft1_test_fixture",
    client_version="1",
    client_api="typed_path_member_sequence",
    endpoint="non_network_fixture",
    repo_type="dataset",
    recursive=True,
    expand=False,
)
PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3 = (
    PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
)
FIXTURE_OBSERVATION_CLIENT_IDENTITY_SHA256_A3 = (
    FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3.receipt_sha256
)


@dataclass(frozen=True)
class PathMemberReceiptA3:
    """One exact file observation from a revision-pinned repository tree."""

    path: str
    upstream_bytes: int
    blob_identity_kind: str
    blob_identity: str

    def __post_init__(self) -> None:
        _canonical_repository_path(self.path)
        _require_nonnegative_int(self.upstream_bytes, "member upstream bytes")
        expected_length = _IDENTITY_LENGTHS.get(self.blob_identity_kind)
        if expected_length is None:
            raise PathBreakdownError("member blob identity kind is unknown")
        expression = _SHA1 if expected_length == 40 else _SHA256
        if (
            not isinstance(self.blob_identity, str)
            or expression.fullmatch(self.blob_identity) is None
        ):
            raise PathBreakdownError(
                "member blob identity does not match its declared kind"
            )

    @property
    def receipt_sha256(self) -> str:
        return _domain_sha256(PATH_BREAKDOWN_MEMBER_SCHEMA_A3_V1, self)


def canonical_member_order_a3(
    members: Sequence[PathMemberReceiptA3],
) -> tuple[PathMemberReceiptA3, ...]:
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise TypeError("path members must be a typed sequence")
    if any(not isinstance(member, PathMemberReceiptA3) for member in members):
        raise TypeError("path members contain an untyped value")
    paths = tuple(member.path for member in members)
    if len(paths) != len(set(paths)):
        raise PathBreakdownError("path observation repeats a repository file")
    return tuple(
        sorted(
            members,
            key=lambda member: (
                member.path.encode("utf-8"),
                member.blob_identity_kind,
                member.blob_identity,
            ),
        )
    )


def observe_hf_tree_files_a3(
    raw_tree: Iterable[object],
) -> tuple[PathMemberReceiptA3, ...]:
    """Normalize one pinned Hugging Face tree without applying a selector."""

    if isinstance(raw_tree, (str, bytes, Mapping)):
        raise TypeError("raw repository tree must be an iterable of entries")
    members: list[PathMemberReceiptA3] = []
    for raw in raw_tree:
        entry_type = _field(raw, "type")
        if entry_type is None:
            has_file_shape = (
                _field(raw, "size") is not None
                and _field(raw, "blob_id") is not None
            )
            has_folder_shape = _field(raw, "tree_id") is not None
            if has_file_shape == has_folder_shape:
                raise PathBreakdownError(
                    "Hugging Face tree entry has an ambiguous shape"
                )
            entry_type = "file" if has_file_shape else "directory"
        if entry_type in {"directory", "dir", "tree"}:
            continue
        if entry_type not in {"file", "blob"}:
            raise PathBreakdownError(
                "Hugging Face tree entry lacks an exact file type"
            )
        path = _field(raw, "path", default=_field(raw, "rfilename"))
        size = _field(raw, "size")
        blob_identity = _field(raw, "blob_id")
        if not isinstance(path, str):
            raise PathBreakdownError("Hugging Face file entry lacks a path")
        _require_nonnegative_int(size, "Hugging Face file size")  # type: ignore[arg-type]
        if not isinstance(blob_identity, str):
            raise PathBreakdownError("Hugging Face file entry lacks a blob_id")
        if _SHA1.fullmatch(blob_identity):
            identity_kind = "git_sha1"
        elif _SHA256.fullmatch(blob_identity):
            identity_kind = "git_sha256"
        else:
            raise PathBreakdownError(
                "Hugging Face blob_id is not a lowercase hash"
            )
        members.append(
            PathMemberReceiptA3(
                path=path,
                upstream_bytes=size,  # type: ignore[arg-type]
                blob_identity_kind=identity_kind,
                blob_identity=blob_identity,
            )
        )
    return canonical_member_order_a3(tuple(members))


def _member_set_sha256(members: Sequence[PathMemberReceiptA3]) -> str:
    ordered = canonical_member_order_a3(members)
    return _domain_sha256(
        "weft1_path_member_set_a3_v1",
        tuple(member.receipt_sha256 for member in ordered),
    )


@dataclass(frozen=True)
class PathGroupReceiptA3:
    """One exact parent-pattern group and its deterministic member inventory."""

    group_id: str
    path_pattern: str
    classification: str
    selected: bool
    asset_count: int
    upstream_bytes: int
    member_set_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.group_id, "path group id")
        _require_nonempty(self.path_pattern, "path group pattern")
        if "\\" in self.path_pattern or "\x00" in self.path_pattern:
            raise PathBreakdownError("path group pattern is noncanonical")
        _require_nonempty(self.classification, "path group classification")
        if type(self.selected) is not bool:
            raise TypeError("path group selected flag must be boolean")
        _require_positive_int(self.asset_count, "path group asset count")
        _require_nonnegative_int(self.upstream_bytes, "path group bytes")
        _require_sha256(self.member_set_sha256, "path group member-set SHA-256")

    @classmethod
    def from_members(
        cls,
        *,
        group_id: str,
        path_pattern: str,
        classification: str,
        selected: bool,
        members: Sequence[PathMemberReceiptA3],
    ) -> "PathGroupReceiptA3":
        ordered = canonical_member_order_a3(members)
        return cls(
            group_id=group_id,
            path_pattern=path_pattern,
            classification=classification,
            selected=selected,
            asset_count=len(ordered),
            upstream_bytes=sum(member.upstream_bytes for member in ordered),
            member_set_sha256=_member_set_sha256(ordered),
        )

    @property
    def receipt_sha256(self) -> str:
        return _domain_sha256(PATH_BREAKDOWN_GROUP_SCHEMA_A3_V1, self)


@dataclass(frozen=True)
class PinnedSemanticEvidenceA3:
    """Content-hash-pinned metadata supporting one semantic classification."""

    evidence_id: str
    locator: str
    pin: str
    content_sha256: str
    assertion: str

    def __post_init__(self) -> None:
        _require_nonempty(self.evidence_id, "semantic evidence id")
        _require_nonempty(self.locator, "semantic evidence locator")
        _require_nonempty(self.pin, "semantic evidence pin")
        _require_sha256(self.content_sha256, "semantic evidence content SHA-256")
        _require_nonempty(self.assertion, "semantic evidence assertion")

    @property
    def receipt_sha256(self) -> str:
        return _domain_sha256(PATH_BREAKDOWN_EVIDENCE_SCHEMA_A3_V1, self)


@dataclass(frozen=True)
class PriorRouteDeclarationA3:
    """The superseded numeric declaration retained as descriptive provenance."""

    source_family: str
    asset_selector: str
    asset_count: int
    available_bytes: int
    declaration_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in BREAKDOWN_SOURCE_FAMILIES:
            raise PathBreakdownError("prior declaration source family is unknown")
        _require_nonempty(self.asset_selector, "prior asset selector")
        _require_positive_int(self.asset_count, "prior asset count")
        _require_positive_int(self.available_bytes, "prior available bytes")
        _require_sha256(
            self.declaration_receipt_sha256,
            "prior declaration receipt SHA-256",
        )


def _canonical_group_order(
    groups: Sequence[PathGroupReceiptA3],
) -> tuple[PathGroupReceiptA3, ...]:
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise TypeError("path groups must be a typed sequence")
    if any(not isinstance(group, PathGroupReceiptA3) for group in groups):
        raise TypeError("path groups contain an untyped value")
    keys = tuple((group.group_id, group.path_pattern) for group in groups)
    if len(keys) != len(set(keys)):
        raise PathBreakdownError("path breakdown repeats a group")
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                group.path_pattern.encode("utf-8"),
                group.group_id.encode("utf-8"),
            ),
        )
    )


def _group_member_sets_sha256(
    groups: Sequence[PathGroupReceiptA3],
) -> str:
    ordered = _canonical_group_order(groups)
    return _domain_sha256(
        "weft1_path_group_member_sets_a3_v1",
        tuple(
            (group.group_id, group.path_pattern, group.member_set_sha256)
            for group in ordered
        ),
    )


@dataclass(frozen=True)
class FamilyPathBreakdownA3:
    """One family definition resolved against exact observed path groups."""

    source_family: str
    repository: str
    revision: str
    family_definition_id: str
    candidate_scope: str
    semantic_evidence: PinnedSemanticEvidenceA3
    semantic_evidence_sha256: str
    prior_declaration: PriorRouteDeclarationA3
    configured_group_ids: tuple[str, ...]
    groups: tuple[PathGroupReceiptA3, ...]
    repository_asset_count: int
    repository_upstream_bytes: int
    repository_member_set_sha256: str
    selected_asset_count: int
    selected_upstream_bytes: int
    selected_member_set_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in BREAKDOWN_SOURCE_FAMILIES:
            raise PathBreakdownError("family path breakdown source is unknown")
        if not isinstance(self.repository, str) or _REPOSITORY.fullmatch(
            self.repository
        ) is None:
            raise PathBreakdownError("family repository must be an owner/name id")
        if not isinstance(self.revision, str) or _SHA1.fullmatch(self.revision) is None:
            raise PathBreakdownError("family revision must be a commit SHA-1")
        _require_nonempty(self.family_definition_id, "family definition id")
        _require_nonempty(self.candidate_scope, "family candidate scope")
        if not isinstance(self.semantic_evidence, PinnedSemanticEvidenceA3):
            raise TypeError("family breakdown requires typed semantic evidence")
        if self.semantic_evidence_sha256 != self.semantic_evidence.receipt_sha256:
            raise PathBreakdownError("family semantic evidence SHA-256 drifted")
        if not isinstance(self.prior_declaration, PriorRouteDeclarationA3):
            raise TypeError("family breakdown requires a prior declaration")
        if self.prior_declaration.source_family != self.source_family:
            raise PathBreakdownError("family prior declaration is misrouted")
        if not isinstance(self.configured_group_ids, tuple):
            raise TypeError("configured group ids must be a tuple")
        if self.configured_group_ids != tuple(
            sorted(self.configured_group_ids, key=lambda value: value.encode("utf-8"))
        ) or len(self.configured_group_ids) != len(set(self.configured_group_ids)):
            raise PathBreakdownError("configured group ids are not canonical and unique")
        if not isinstance(self.groups, tuple) or not self.groups:
            raise PathBreakdownError("family path breakdown requires groups")
        if self.groups != _canonical_group_order(self.groups):
            raise PathBreakdownError("family path groups are not canonical")
        _require_positive_int(
            self.repository_asset_count, "repository observation asset count"
        )
        _require_nonnegative_int(
            self.repository_upstream_bytes, "repository observation bytes"
        )
        _require_sha256(
            self.repository_member_set_sha256,
            "repository observation member-set SHA-256",
        )
        if self.repository_asset_count != sum(
            group.asset_count for group in self.groups
        ):
            raise PathBreakdownError(
                "repository observation asset count is not fully explained"
            )
        if self.repository_upstream_bytes != sum(
            group.upstream_bytes for group in self.groups
        ):
            raise PathBreakdownError(
                "repository observation bytes are not fully explained"
            )
        if self.repository_member_set_sha256 != _group_member_sets_sha256(
            self.groups
        ):
            raise PathBreakdownError(
                "repository observation member-set SHA-256 is not fully explained"
            )
        selected_groups = tuple(group for group in self.groups if group.selected)
        if not selected_groups:
            raise PathBreakdownError("family path breakdown selects no groups")
        if self.selected_asset_count != sum(
            group.asset_count for group in selected_groups
        ):
            raise PathBreakdownError("family selected asset count is not derived")
        if self.selected_upstream_bytes != sum(
            group.upstream_bytes for group in selected_groups
        ):
            raise PathBreakdownError("family selected byte total is not derived")
        if self.selected_member_set_sha256 != _group_member_sets_sha256(
            selected_groups
        ):
            raise PathBreakdownError("family selected member-set SHA-256 drifted")
        _validate_family_semantics(self)

    @property
    def reminted_asset_count(self) -> int:
        return self.selected_asset_count

    @property
    def reminted_available_bytes(self) -> int:
        return self.selected_upstream_bytes


def _validate_family_semantics(family: FamilyPathBreakdownA3) -> None:
    if family.source_family == DOLMA_SOURCE_FAMILY:
        if family.family_definition_id != DOLMA_DEFINITION_A3_V1:
            raise PathBreakdownError("Dolma family definition drifted")
        if family.candidate_scope != DOLMA_CANDIDATE_SCOPE_A3_V1:
            raise PathBreakdownError("Dolma candidate scope drifted")
        if family.semantic_evidence.assertion != DOLMA_TOP_QUALITY_ASSERTION_A3_V1:
            raise PathBreakdownError(
                "Dolma top-quality claim lacks its pinned semantic assertion"
            )
        if family.configured_group_ids:
            raise PathBreakdownError("Dolma does not use a configured group-id set")
        observed_buckets: set[str] = set()
        for group in family.groups:
            if group.classification not in _DOLMA_CLASSIFICATIONS:
                raise PathBreakdownError("Dolma path classification is unknown")
            expected_selected = group.classification == DOLMA_SELECTED_CLASSIFICATION
            if group.selected != expected_selected:
                raise PathBreakdownError("Dolma selector/classification mismatch")
            if group.classification in {
                DOLMA_SELECTED_CLASSIFICATION,
                DOLMA_EXCLUDED_CLASSIFICATION,
            }:
                match = _DOLMA_GROUP_ID.fullmatch(group.group_id)
                if match is None:
                    raise PathBreakdownError("Dolma Common Crawl group id drifted")
                bucket = match.group("bucket")
                observed_buckets.add(bucket)
                if group.path_pattern != f"data/{group.group_id}/*.jsonl.zst":
                    raise PathBreakdownError("Dolma Common Crawl path pattern drifted")
                classification = (
                    DOLMA_SELECTED_CLASSIFICATION
                    if bucket == "0019"
                    else DOLMA_EXCLUDED_CLASSIFICATION
                )
                if group.classification != classification:
                    raise PathBreakdownError("Dolma bucket classification drifted")
            elif group.classification == DOLMA_NON_COMMON_CRAWL_CLASSIFICATION:
                if (
                    group.group_id != "data_non_common_crawl"
                    or group.path_pattern != "data/<non-common-crawl>/**"
                ):
                    raise PathBreakdownError("Dolma non-Common-Crawl group drifted")
            elif (
                group.group_id != "repository_metadata"
                or group.path_pattern != "<repository-metadata>/**"
            ):
                raise PathBreakdownError("Dolma repository-metadata group drifted")
        maximum = (
            max(observed_buckets, key=lambda value: int(value))
            if observed_buckets
            else "none"
        )
        if maximum != "0019":
            raise PathBreakdownError(
                "Dolma observed selectable maximum bucket is "
                f"{maximum}, not pinned top bucket 0019; strategy escalation required"
            )
        return

    if family.family_definition_id != FINEWEB_DEFINITION_A3_V1:
        raise PathBreakdownError("FineWeb family definition drifted")
    if family.candidate_scope != FINEWEB_CANDIDATE_SCOPE_A3_V1:
        raise PathBreakdownError("FineWeb candidate scope drifted")
    if family.semantic_evidence.assertion != FINEWEB_MAIN_DATA_ASSERTION_A3_V1:
        raise PathBreakdownError(
            "FineWeb main-data claim lacks its pinned semantic assertion"
        )
    if len(family.configured_group_ids) != 110 or any(
        _FINEWEB_MAIN_DUMP.fullmatch(group_id) is None
        for group_id in family.configured_group_ids
    ):
        raise PathBreakdownError("FineWeb requires exactly 110 canonical CC-MAIN ids")
    selected_ids: list[str] = []
    for group in family.groups:
        if group.classification not in _FINEWEB_CLASSIFICATIONS:
            raise PathBreakdownError("FineWeb path classification is unknown")
        expected_selected = group.classification == FINEWEB_SELECTED_CLASSIFICATION
        if group.selected != expected_selected:
            raise PathBreakdownError("FineWeb selector/classification mismatch")
        if expected_selected:
            selected_ids.append(group.group_id)
            if group.path_pattern != f"data/{group.group_id}/*.parquet":
                raise PathBreakdownError("FineWeb main-data path pattern drifted")
        elif group.classification == FINEWEB_SAMPLE_CLASSIFICATION:
            if not (
                group.group_id.startswith("data/sample")
                and group.path_pattern == f"data/{group.group_id[5:]}/**"
                or group.group_id.startswith("sample/")
                and group.path_pattern == f"{group.group_id}/**"
            ):
                raise PathBreakdownError("FineWeb sample path group drifted")
        elif group.classification == FINEWEB_SCORE_CLASSIFICATION:
            if not (
                group.group_id.startswith("data/")
                and group.path_pattern == f"{group.group_id}/**"
            ):
                raise PathBreakdownError("FineWeb score path group drifted")
        elif (
            group.group_id != "repository_metadata"
            or group.path_pattern != "<repository-metadata>/**"
        ):
            raise PathBreakdownError("FineWeb repository-metadata group drifted")
    if tuple(sorted(selected_ids, key=lambda value: value.encode("utf-8"))) != (
        family.configured_group_ids
    ):
        raise PathBreakdownError(
            "FineWeb selected groups do not exactly cover configured main dumps"
        )


def _build_group(
    *,
    group_id: str,
    path_pattern: str,
    classification: str,
    selected: bool,
    members: Sequence[PathMemberReceiptA3],
) -> PathGroupReceiptA3:
    return PathGroupReceiptA3.from_members(
        group_id=group_id,
        path_pattern=path_pattern,
        classification=classification,
        selected=selected,
        members=members,
    )


def _build_family(
    *,
    source_family: str,
    repository: str,
    revision: str,
    family_definition_id: str,
    candidate_scope: str,
    semantic_evidence: PinnedSemanticEvidenceA3,
    prior_declaration: PriorRouteDeclarationA3,
    configured_group_ids: tuple[str, ...],
    groups: Sequence[PathGroupReceiptA3],
    repository_members: Sequence[PathMemberReceiptA3],
) -> FamilyPathBreakdownA3:
    ordered_groups = _canonical_group_order(groups)
    ordered_repository = canonical_member_order_a3(repository_members)
    selected_groups = tuple(group for group in ordered_groups if group.selected)
    return FamilyPathBreakdownA3(
        source_family=source_family,
        repository=repository,
        revision=revision,
        family_definition_id=family_definition_id,
        candidate_scope=candidate_scope,
        semantic_evidence=semantic_evidence,
        semantic_evidence_sha256=semantic_evidence.receipt_sha256,
        prior_declaration=prior_declaration,
        configured_group_ids=configured_group_ids,
        groups=ordered_groups,
        repository_asset_count=len(ordered_repository),
        repository_upstream_bytes=sum(
            member.upstream_bytes for member in ordered_repository
        ),
        repository_member_set_sha256=_group_member_sets_sha256(ordered_groups),
        selected_asset_count=sum(group.asset_count for group in selected_groups),
        selected_upstream_bytes=sum(
            group.upstream_bytes for group in selected_groups
        ),
        selected_member_set_sha256=_group_member_sets_sha256(selected_groups),
    )


def build_dolma_path_breakdown_a3(
    members: Sequence[PathMemberReceiptA3],
    *,
    repository: str,
    revision: str,
    prior_declaration: PriorRouteDeclarationA3,
    semantic_evidence: PinnedSemanticEvidenceA3,
) -> FamilyPathBreakdownA3:
    """Classify all Dolma Common Crawl groups; select only proven bucket 0019."""

    if not isinstance(semantic_evidence, PinnedSemanticEvidenceA3):
        raise TypeError("Dolma requires pinned semantic evidence metadata")
    if semantic_evidence.assertion != DOLMA_TOP_QUALITY_ASSERTION_A3_V1:
        raise PathBreakdownError(
            "Dolma top-quality selection requires pinned 0019 ordering evidence"
        )
    ordered = canonical_member_order_a3(members)
    grouped: dict[
        tuple[str, str, str, bool],
        list[PathMemberReceiptA3],
    ] = {}
    observed_buckets: set[str] = set()
    for member in ordered:
        match = _DOLMA_ASSET.fullmatch(member.path)
        if match is None:
            if member.path.startswith("data/common_crawl-"):
                raise PathBreakdownError(
                    f"unclassified Dolma Common Crawl path: {member.path}"
                )
            if member.path.startswith("data/"):
                key = (
                    "data_non_common_crawl",
                    "data/<non-common-crawl>/**",
                    DOLMA_NON_COMMON_CRAWL_CLASSIFICATION,
                    False,
                )
            else:
                key = (
                    "repository_metadata",
                    "<repository-metadata>/**",
                    REPOSITORY_METADATA_CLASSIFICATION,
                    False,
                )
            grouped.setdefault(key, []).append(member)
            continue
        topic = match.group("topic")
        bucket = match.group("bucket")
        observed_buckets.add(bucket)
        group_id = f"common_crawl-{topic}-{bucket}"
        classification = (
            DOLMA_SELECTED_CLASSIFICATION
            if bucket == "0019"
            else DOLMA_EXCLUDED_CLASSIFICATION
        )
        key = (
            group_id,
            f"data/{group_id}/*.jsonl.zst",
            classification,
            bucket == "0019",
        )
        grouped.setdefault(key, []).append(member)
    maximum = (
        max(observed_buckets, key=lambda value: int(value))
        if observed_buckets
        else "none"
    )
    if maximum != "0019":
        raise PathBreakdownError(
            "Dolma observed selectable maximum bucket is "
            f"{maximum}, not pinned top bucket 0019; strategy escalation required"
        )
    groups = tuple(
        _build_group(
            group_id=group_id,
            path_pattern=path_pattern,
            classification=classification,
            selected=selected,
            members=rows,
        )
        for (
            group_id,
            path_pattern,
            classification,
            selected,
        ), rows in grouped.items()
    )
    return _build_family(
        source_family=DOLMA_SOURCE_FAMILY,
        repository=repository,
        revision=revision,
        family_definition_id=DOLMA_DEFINITION_A3_V1,
        candidate_scope=DOLMA_CANDIDATE_SCOPE_A3_V1,
        semantic_evidence=semantic_evidence,
        prior_declaration=prior_declaration,
        configured_group_ids=(),
        groups=groups,
        repository_members=ordered,
    )


def _canonical_fineweb_dump_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("FineWeb dump ids must be a typed sequence")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("FineWeb dump ids must be exact strings")
    ordered = tuple(sorted(values, key=lambda value: value.encode("utf-8")))
    if len(ordered) != 110 or len(set(ordered)) != 110:
        raise PathBreakdownError("FineWeb requires exactly 110 unique dump ids")
    if any(_FINEWEB_MAIN_DUMP.fullmatch(value) is None for value in ordered):
        raise PathBreakdownError(
            "FineWeb configured dump id is not canonical CC-MAIN-YYYY-NN"
        )
    return ordered


def _fineweb_exclusion(group_id: str) -> str | None:
    if _FINEWEB_SCORE_TOKEN.search(group_id) is not None:
        return FINEWEB_SCORE_CLASSIFICATION
    if group_id == "sample" or group_id.startswith("sample-"):
        return FINEWEB_SAMPLE_CLASSIFICATION
    return None


def build_fineweb_path_breakdown_a3(
    members: Sequence[PathMemberReceiptA3],
    *,
    repository: str,
    revision: str,
    prior_declaration: PriorRouteDeclarationA3,
    semantic_evidence: PinnedSemanticEvidenceA3,
    configured_main_dump_ids: Sequence[str],
) -> FamilyPathBreakdownA3:
    """Select every Parquet asset in exactly the configured 110 main dumps."""

    if not isinstance(semantic_evidence, PinnedSemanticEvidenceA3):
        raise TypeError("FineWeb requires pinned semantic evidence metadata")
    if semantic_evidence.assertion != FINEWEB_MAIN_DATA_ASSERTION_A3_V1:
        raise PathBreakdownError(
            "FineWeb main-data selection requires pinned card evidence"
        )
    expected_ids = _canonical_fineweb_dump_ids(configured_main_dump_ids)
    expected = frozenset(expected_ids)
    ordered = canonical_member_order_a3(members)
    grouped: dict[
        tuple[str, str, str, bool],
        list[PathMemberReceiptA3],
    ] = {}
    for member in ordered:
        root_sample = _FINEWEB_ROOT_SAMPLE.fullmatch(member.path)
        if root_sample is not None:
            config = root_sample.group("config")
            key = (
                f"sample/{config}",
                f"sample/{config}/**",
                FINEWEB_SAMPLE_CLASSIFICATION,
                False,
            )
            grouped.setdefault(key, []).append(member)
            continue
        if member.path.startswith("sample/"):
            raise PathBreakdownError(
                f"unclassified FineWeb root sample path: {member.path}"
            )
        match = _FINEWEB_PARQUET.fullmatch(member.path)
        if match is None:
            if member.path.startswith("data/"):
                parts = PurePosixPath(member.path).parts
                if len(parts) < 2:
                    raise PathBreakdownError(
                        f"unclassified FineWeb data path: {member.path}"
                    )
                group_id = parts[1]
                if group_id in expected or _FINEWEB_MAIN_DUMP.fullmatch(
                    group_id
                ) is not None:
                    raise PathBreakdownError(
                        f"unclassified FineWeb in-family data path: {member.path}"
                    )
                classification = _fineweb_exclusion(group_id)
                if classification is None:
                    raise PathBreakdownError(
                        "FineWeb observation contains an unknown data group: "
                        f"{group_id}"
                    )
                key = (
                    f"data/{group_id}",
                    f"data/{group_id}/**",
                    classification,
                    False,
                )
                grouped.setdefault(key, []).append(member)
                continue
            key = (
                "repository_metadata",
                "<repository-metadata>/**",
                REPOSITORY_METADATA_CLASSIFICATION,
                False,
            )
            grouped.setdefault(key, []).append(member)
            continue
        group_id = match.group("group")
        if group_id in expected:
            classification = FINEWEB_SELECTED_CLASSIFICATION
            key = (
                group_id,
                f"data/{group_id}/*.parquet",
                classification,
                True,
            )
        elif _FINEWEB_MAIN_DUMP.fullmatch(group_id) is not None:
            raise PathBreakdownError(
                f"FineWeb observation contains an extra main dump: {group_id}"
            )
        elif (classification := _fineweb_exclusion(group_id)) is None:
            raise PathBreakdownError(
                f"FineWeb observation contains an unknown data group: {group_id}"
            )
        else:
            key = (
                f"data/{group_id}",
                f"data/{group_id}/**",
                classification,
                False,
            )
        grouped.setdefault(key, []).append(member)
    observed_main = frozenset(
        group_id
        for group_id, _path_pattern, classification, _selected in grouped
        if classification == FINEWEB_SELECTED_CLASSIFICATION
    )
    if observed_main != expected:
        missing = tuple(sorted(expected - observed_main))
        raise PathBreakdownError(
            f"FineWeb observation omits {len(missing)} configured main dumps"
        )
    groups = tuple(
        _build_group(
            group_id=group_id,
            path_pattern=path_pattern,
            classification=classification,
            selected=selected,
            members=rows,
        )
        for (
            group_id,
            path_pattern,
            classification,
            selected,
        ), rows in grouped.items()
    )
    return _build_family(
        source_family=FINEWEB_SOURCE_FAMILY,
        repository=repository,
        revision=revision,
        family_definition_id=FINEWEB_DEFINITION_A3_V1,
        candidate_scope=FINEWEB_CANDIDATE_SCOPE_A3_V1,
        semantic_evidence=semantic_evidence,
        prior_declaration=prior_declaration,
        configured_group_ids=expected_ids,
        groups=groups,
        repository_members=ordered,
    )


def _configuration_payload(
    families: Sequence[FamilyPathBreakdownA3],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "candidate_scope": family.candidate_scope,
            "configured_group_ids": family.configured_group_ids,
            "family_definition_id": family.family_definition_id,
            "prior_declaration": family.prior_declaration,
            "repository": family.repository,
            "revision": family.revision,
            "semantic_evidence_sha256": family.semantic_evidence_sha256,
            "source_family": family.source_family,
        }
        for family in families
    )


@dataclass(frozen=True)
class UpstreamPathBreakdownReceiptA3:
    """Durable A3 observation; explicitly not source-execution authority."""

    schema: str
    status: str
    authorizes_downloads: bool
    authority_sha256: str
    observation_mode: str
    observation_client_identity: PathObservationClientIdentityA3
    observation_client_identity_sha256: str
    configuration_sha256: str
    families: tuple[FamilyPathBreakdownA3, ...]

    def __post_init__(self) -> None:
        if self.schema != PATH_BREAKDOWN_RECEIPT_SCHEMA_A3_V1:
            raise PathBreakdownError("unexpected A3 path-breakdown schema")
        if self.status != PATH_BREAKDOWN_STATUS_A3_V1:
            raise PathBreakdownError("A3 path-breakdown status drifted")
        if self.authorizes_downloads is not False:
            raise PathBreakdownError("path observation may not authorize downloads")
        if self.authority_sha256 != A3_AUTHORITY_SHA256:
            raise PathBreakdownError(
                "path breakdown is not bound to the verified A3 authority"
            )
        if self.observation_mode not in OBSERVATION_MODES_A3:
            raise PathBreakdownError("path observation mode is unknown")
        if not isinstance(
            self.observation_client_identity,
            PathObservationClientIdentityA3,
        ):
            raise TypeError("path observation requires a typed client identity")
        expected_client = (
            PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3
            if self.observation_mode == PRODUCTION_OBSERVATION_MODE_A3
            else FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3
        )
        if self.observation_client_identity != expected_client:
            raise PathBreakdownError(
                "path observation client identity does not match its mode"
            )
        if (
            self.observation_client_identity_sha256
            != self.observation_client_identity.receipt_sha256
        ):
            raise PathBreakdownError(
                "path observation client identity SHA-256 drifted"
            )
        _require_sha256(self.configuration_sha256, "A3 configuration SHA-256")
        if not isinstance(self.families, tuple):
            raise TypeError("A3 path-breakdown families must be a tuple")
        if tuple(family.source_family for family in self.families) != (
            BREAKDOWN_SOURCE_FAMILIES
        ):
            raise PathBreakdownError(
                "A3 path breakdown requires Dolma then FineWeb exactly"
            )
        expected_configuration = _domain_sha256(
            PATH_BREAKDOWN_CONFIGURATION_SCHEMA_A3_V1,
            _configuration_payload(self.families),
        )
        if self.configuration_sha256 != expected_configuration:
            raise PathBreakdownError("A3 path-breakdown configuration drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v4_bound_sha256(self.schema, self)


def build_upstream_path_breakdown_a3(
    *,
    authority_sha256: str,
    observation_mode: str,
    observation_client_identity: PathObservationClientIdentityA3,
    dolma: FamilyPathBreakdownA3,
    fineweb: FamilyPathBreakdownA3,
) -> UpstreamPathBreakdownReceiptA3:
    if not isinstance(dolma, FamilyPathBreakdownA3) or not isinstance(
        fineweb, FamilyPathBreakdownA3
    ):
        raise TypeError("A3 breakdown requires typed family observations")
    if not isinstance(
        observation_client_identity,
        PathObservationClientIdentityA3,
    ):
        raise TypeError("A3 breakdown requires a typed observation client identity")
    if authority_sha256 != A3_AUTHORITY_SHA256:
        raise PathBreakdownError("foreign A3 authority cannot mint a breakdown")
    families = (dolma, fineweb)
    configuration_sha256 = _domain_sha256(
        PATH_BREAKDOWN_CONFIGURATION_SCHEMA_A3_V1,
        _configuration_payload(families),
    )
    return UpstreamPathBreakdownReceiptA3(
        schema=PATH_BREAKDOWN_RECEIPT_SCHEMA_A3_V1,
        status=PATH_BREAKDOWN_STATUS_A3_V1,
        authorizes_downloads=False,
        authority_sha256=authority_sha256,
        observation_mode=observation_mode,
        observation_client_identity=observation_client_identity,
        observation_client_identity_sha256=(
            observation_client_identity.receipt_sha256
        ),
        configuration_sha256=configuration_sha256,
        families=families,
    )


@dataclass(frozen=True)
class FamilyResolutionProjectionA3:
    """Compact deterministic edge from observer evidence to an A3 overlay row."""

    source_family: str
    repository: str
    revision: str
    family_definition_id: str
    semantic_evidence_sha256: str
    configured_group_ids_sha256: str
    selected_path_patterns: tuple[str, ...]
    effective_asset_selector: str
    selected_asset_count: int
    selected_upstream_bytes: int
    available_bytes_basis: str
    resolution: str
    repository_member_set_sha256: str
    selected_member_set_sha256: str
    breakdown_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.source_family not in BREAKDOWN_SOURCE_FAMILIES:
            raise PathBreakdownError("resolution projection source is unknown")
        if not isinstance(self.repository, str) or _REPOSITORY.fullmatch(
            self.repository
        ) is None:
            raise PathBreakdownError("projection repository is invalid")
        if not isinstance(self.revision, str) or _SHA1.fullmatch(self.revision) is None:
            raise PathBreakdownError("projection revision is invalid")
        _require_nonempty(self.family_definition_id, "projection family definition")
        for value, name in (
            (self.semantic_evidence_sha256, "projection semantic evidence"),
            (self.configured_group_ids_sha256, "projection group-id set"),
            (self.repository_member_set_sha256, "projection repository universe"),
            (self.selected_member_set_sha256, "projection selected members"),
            (self.breakdown_receipt_sha256, "projection breakdown receipt"),
        ):
            _require_sha256(value, name)
        if not isinstance(self.selected_path_patterns, tuple) or not (
            self.selected_path_patterns
        ):
            raise PathBreakdownError("projection requires selected path patterns")
        if self.selected_path_patterns != tuple(
            sorted(self.selected_path_patterns, key=lambda value: value.encode("utf-8"))
        ) or len(self.selected_path_patterns) != len(set(self.selected_path_patterns)):
            raise PathBreakdownError("projection path patterns are not canonical")
        _require_nonempty(self.effective_asset_selector, "effective asset selector")
        _require_positive_int(self.selected_asset_count, "projected asset count")
        _require_positive_int(self.selected_upstream_bytes, "projected bytes")
        _require_nonempty(self.available_bytes_basis, "projected byte basis")
        _require_nonempty(self.resolution, "projected resolution")


def project_family_resolution_a3(
    receipt: UpstreamPathBreakdownReceiptA3,
    source_family: str,
) -> FamilyResolutionProjectionA3:
    """Derive the only local overlay projection permitted by this observation."""

    if not isinstance(receipt, UpstreamPathBreakdownReceiptA3):
        raise TypeError("resolution projection requires a typed breakdown")
    if source_family not in BREAKDOWN_SOURCE_FAMILIES:
        raise PathBreakdownError("resolution projection source is unknown")
    family = next(
        row for row in receipt.families if row.source_family == source_family
    )
    selected_patterns = tuple(
        group.path_pattern for group in family.groups if group.selected
    )
    if source_family == DOLMA_SOURCE_FAMILY:
        selector = DOLMA_EFFECTIVE_SELECTOR_A3_V1
        selector_already_exact = family.prior_declaration.asset_selector == selector
        resolution = (
            DOLMA_CONFIRM_RESOLUTION_A3_V1
            if selector_already_exact
            else DOLMA_NARROW_RESOLUTION_A3_V1
        )
        byte_basis = "pinned repository compressed asset bytes"
    else:
        selector = FINEWEB_EFFECTIVE_SELECTOR_A3_V1
        selector_already_exact = family.prior_declaration.asset_selector == selector
        resolution = (
            FINEWEB_ACCEPT_RESOLUTION_A3_V1
            if selector_already_exact
            else FINEWEB_WIDEN_RESOLUTION_A3_V1
        )
        byte_basis = "pinned repository parquet bytes"
    return FamilyResolutionProjectionA3(
        source_family=source_family,
        repository=family.repository,
        revision=family.revision,
        family_definition_id=family.family_definition_id,
        semantic_evidence_sha256=family.semantic_evidence_sha256,
        configured_group_ids_sha256=_domain_sha256(
            "weft1_configured_path_group_ids_a3_v1",
            family.configured_group_ids,
        ),
        selected_path_patterns=selected_patterns,
        effective_asset_selector=selector,
        selected_asset_count=family.selected_asset_count,
        selected_upstream_bytes=family.selected_upstream_bytes,
        available_bytes_basis=byte_basis,
        resolution=resolution,
        repository_member_set_sha256=family.repository_member_set_sha256,
        selected_member_set_sha256=family.selected_member_set_sha256,
        breakdown_receipt_sha256=receipt.receipt_sha256,
    )


def replay_upstream_path_breakdown_a3(
    expected: UpstreamPathBreakdownReceiptA3,
    *,
    dolma_members: Sequence[PathMemberReceiptA3],
    fineweb_members: Sequence[PathMemberReceiptA3],
) -> UpstreamPathBreakdownReceiptA3:
    """Re-enumerate transient members and require the compact receipt exactly."""

    if not isinstance(expected, UpstreamPathBreakdownReceiptA3):
        raise TypeError("breakdown replay requires a typed expected receipt")
    dolma_expected, fineweb_expected = expected.families
    dolma = build_dolma_path_breakdown_a3(
        dolma_members,
        repository=dolma_expected.repository,
        revision=dolma_expected.revision,
        prior_declaration=dolma_expected.prior_declaration,
        semantic_evidence=dolma_expected.semantic_evidence,
    )
    fineweb = build_fineweb_path_breakdown_a3(
        fineweb_members,
        repository=fineweb_expected.repository,
        revision=fineweb_expected.revision,
        prior_declaration=fineweb_expected.prior_declaration,
        semantic_evidence=fineweb_expected.semantic_evidence,
        configured_main_dump_ids=fineweb_expected.configured_group_ids,
    )
    replayed = build_upstream_path_breakdown_a3(
        authority_sha256=expected.authority_sha256,
        observation_mode=expected.observation_mode,
        observation_client_identity=expected.observation_client_identity,
        dolma=dolma,
        fineweb=fineweb,
    )
    if replayed != expected:
        raise PathBreakdownError("live path-breakdown replay differs from receipt")
    return replayed


def write_upstream_path_breakdown_a3(
    receipt: UpstreamPathBreakdownReceiptA3,
    path: Path,
) -> str:
    """Atomically persist one canonical A3 observation envelope."""

    if not isinstance(receipt, UpstreamPathBreakdownReceiptA3):
        raise TypeError("path-breakdown artifact requires a typed receipt")
    if not isinstance(path, Path):
        raise TypeError("path-breakdown artifact path must be pathlib.Path")
    assert_no_symlink_ancestors(path)
    if path.exists():
        raise PathBreakdownError("refusing to overwrite a path-breakdown artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise PathBreakdownError("stale path-breakdown partial exists")
    envelope = {
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "schema": PATH_BREAKDOWN_ARTIFACT_SCHEMA_A3_V1,
    }
    raw = canonical_json_bytes(envelope) + b"\n"
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def _require_mapping_keys(
    value: object,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PathBreakdownError(f"{name} shape drifted")
    return value


def _load_group(value: object) -> PathGroupReceiptA3:
    row = _require_mapping_keys(
        value,
        {
            "asset_count",
            "classification",
            "group_id",
            "member_set_sha256",
            "path_pattern",
            "selected",
            "upstream_bytes",
        },
        "path group",
    )
    return PathGroupReceiptA3(
        group_id=row["group_id"],  # type: ignore[arg-type]
        path_pattern=row["path_pattern"],  # type: ignore[arg-type]
        classification=row["classification"],  # type: ignore[arg-type]
        selected=row["selected"],  # type: ignore[arg-type]
        asset_count=row["asset_count"],  # type: ignore[arg-type]
        upstream_bytes=row["upstream_bytes"],  # type: ignore[arg-type]
        member_set_sha256=row["member_set_sha256"],  # type: ignore[arg-type]
    )


def _load_evidence(value: object) -> PinnedSemanticEvidenceA3:
    row = _require_mapping_keys(
        value,
        {"assertion", "content_sha256", "evidence_id", "locator", "pin"},
        "semantic evidence",
    )
    return PinnedSemanticEvidenceA3(**dict(row))


def _load_observation_client_identity(
    value: object,
) -> PathObservationClientIdentityA3:
    row = _require_mapping_keys(
        value,
        {
            "client_api",
            "client_package",
            "client_version",
            "endpoint",
            "expand",
            "recursive",
            "repo_type",
        },
        "path observation client identity",
    )
    return PathObservationClientIdentityA3(**dict(row))


def _load_prior(value: object) -> PriorRouteDeclarationA3:
    row = _require_mapping_keys(
        value,
        {
            "asset_count",
            "asset_selector",
            "available_bytes",
            "declaration_receipt_sha256",
            "source_family",
        },
        "prior declaration",
    )
    return PriorRouteDeclarationA3(**dict(row))


def _load_family(value: object) -> FamilyPathBreakdownA3:
    row = _require_mapping_keys(
        value,
        {
            "candidate_scope",
            "configured_group_ids",
            "family_definition_id",
            "groups",
            "prior_declaration",
            "repository",
            "repository_asset_count",
            "repository_member_set_sha256",
            "repository_upstream_bytes",
            "revision",
            "selected_asset_count",
            "selected_member_set_sha256",
            "selected_upstream_bytes",
            "semantic_evidence",
            "semantic_evidence_sha256",
            "source_family",
        },
        "family path breakdown",
    )
    raw_groups = row["groups"]
    raw_group_ids = row["configured_group_ids"]
    if not isinstance(raw_groups, list) or not isinstance(raw_group_ids, list):
        raise PathBreakdownError("family group values must be lists")
    return FamilyPathBreakdownA3(
        source_family=row["source_family"],  # type: ignore[arg-type]
        repository=row["repository"],  # type: ignore[arg-type]
        revision=row["revision"],  # type: ignore[arg-type]
        family_definition_id=row["family_definition_id"],  # type: ignore[arg-type]
        candidate_scope=row["candidate_scope"],  # type: ignore[arg-type]
        semantic_evidence=_load_evidence(row["semantic_evidence"]),
        semantic_evidence_sha256=row["semantic_evidence_sha256"],  # type: ignore[arg-type]
        prior_declaration=_load_prior(row["prior_declaration"]),
        configured_group_ids=tuple(raw_group_ids),  # type: ignore[arg-type]
        groups=tuple(_load_group(group) for group in raw_groups),
        repository_asset_count=row["repository_asset_count"],  # type: ignore[arg-type]
        repository_upstream_bytes=row["repository_upstream_bytes"],  # type: ignore[arg-type]
        repository_member_set_sha256=row["repository_member_set_sha256"],  # type: ignore[arg-type]
        selected_asset_count=row["selected_asset_count"],  # type: ignore[arg-type]
        selected_upstream_bytes=row["selected_upstream_bytes"],  # type: ignore[arg-type]
        selected_member_set_sha256=row["selected_member_set_sha256"],  # type: ignore[arg-type]
    )


def _parse_upstream_path_breakdown_a3(
    raw: bytes,
    envelope: Mapping[str, Any],
    *,
    expected_receipt_sha256: str | None,
) -> UpstreamPathBreakdownReceiptA3:
    if expected_receipt_sha256 is not None:
        _require_sha256(expected_receipt_sha256, "expected breakdown receipt")
    if raw != canonical_json_bytes(envelope) + b"\n":
        raise PathBreakdownError("path-breakdown artifact is not canonical JSON")
    row = _require_mapping_keys(
        envelope,
        {"receipt", "receipt_sha256", "schema"},
        "path-breakdown artifact",
    )
    if row["schema"] != PATH_BREAKDOWN_ARTIFACT_SCHEMA_A3_V1:
        raise PathBreakdownError("path-breakdown artifact schema drifted")
    receipt_row = _require_mapping_keys(
        row["receipt"],
        {
            "authority_sha256",
            "authorizes_downloads",
            "configuration_sha256",
            "families",
            "observation_client_identity",
            "observation_client_identity_sha256",
            "observation_mode",
            "schema",
            "status",
        },
        "path-breakdown receipt",
    )
    raw_families = receipt_row["families"]
    if not isinstance(raw_families, list):
        raise PathBreakdownError("path-breakdown families must be a list")
    receipt = UpstreamPathBreakdownReceiptA3(
        schema=receipt_row["schema"],  # type: ignore[arg-type]
        status=receipt_row["status"],  # type: ignore[arg-type]
        authorizes_downloads=receipt_row["authorizes_downloads"],  # type: ignore[arg-type]
        authority_sha256=receipt_row["authority_sha256"],  # type: ignore[arg-type]
        observation_mode=receipt_row["observation_mode"],  # type: ignore[arg-type]
        observation_client_identity=_load_observation_client_identity(
            receipt_row["observation_client_identity"]
        ),
        observation_client_identity_sha256=receipt_row[
            "observation_client_identity_sha256"
        ],  # type: ignore[arg-type]
        configuration_sha256=receipt_row["configuration_sha256"],  # type: ignore[arg-type]
        families=tuple(_load_family(family) for family in raw_families),
    )
    if row["receipt_sha256"] != receipt.receipt_sha256:
        raise PathBreakdownError("path-breakdown receipt SHA-256 drifted")
    if (
        expected_receipt_sha256 is not None
        and receipt.receipt_sha256 != expected_receipt_sha256
    ):
        raise PathBreakdownError("path-breakdown receipt differs from expected")
    return receipt


def load_upstream_path_breakdown_snapshot_a3(
    path: Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> tuple[bytes, UpstreamPathBreakdownReceiptA3]:
    """Read once and return both exact artifact bytes and its typed receipt."""

    raw, envelope = load_canonical_json_snapshot(path)
    receipt = _parse_upstream_path_breakdown_a3(
        raw,
        envelope,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return raw, receipt


def load_upstream_path_breakdown_a3(
    path: Path,
    *,
    expected_authority_sha256: str = A3_AUTHORITY_SHA256,
    expected_receipt_sha256: str | None = None,
) -> UpstreamPathBreakdownReceiptA3:
    """Replay a canonical observation and re-derive every nested total/hash."""

    if expected_authority_sha256 != A3_AUTHORITY_SHA256:
        raise PathBreakdownError("path-breakdown authority differs from expected")
    _, receipt = load_upstream_path_breakdown_snapshot_a3(
        path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return receipt


__all__ = [
    "BREAKDOWN_SOURCE_FAMILIES",
    "DOLMA_EXCLUDED_CLASSIFICATION",
    "DOLMA_NON_COMMON_CRAWL_CLASSIFICATION",
    "DOLMA_SOURCE_FAMILY",
    "DOLMA_TOP_QUALITY_ASSERTION_A3_V1",
    "DOLMA_SELECTED_CLASSIFICATION",
    "FINEWEB_MAIN_DATA_ASSERTION_A3_V1",
    "FINEWEB_SAMPLE_CLASSIFICATION",
    "FINEWEB_SCORE_CLASSIFICATION",
    "FINEWEB_SELECTED_CLASSIFICATION",
    "FINEWEB_SOURCE_FAMILY",
    "FIXTURE_OBSERVATION_CLIENT_IDENTITY_A3",
    "FIXTURE_OBSERVATION_CLIENT_IDENTITY_SHA256_A3",
    "FIXTURE_OBSERVATION_MODE_A3",
    "FamilyPathBreakdownA3",
    "FamilyResolutionProjectionA3",
    "OBSERVATION_MODES_A3",
    "PATH_BREAKDOWN_ARTIFACT_SCHEMA_A3_V1",
    "PATH_BREAKDOWN_CLIENT_IDENTITY_SCHEMA_A3_V1",
    "PATH_BREAKDOWN_RECEIPT_SCHEMA_A3_V1",
    "PATH_BREAKDOWN_STATUS_A3_V1",
    "PRODUCTION_OBSERVATION_CLIENT_IDENTITY_A3",
    "PRODUCTION_OBSERVATION_CLIENT_IDENTITY_SHA256_A3",
    "PRODUCTION_OBSERVATION_MODE_A3",
    "REPOSITORY_METADATA_CLASSIFICATION",
    "PathBreakdownError",
    "PathGroupReceiptA3",
    "PathMemberReceiptA3",
    "PathObservationClientIdentityA3",
    "PinnedSemanticEvidenceA3",
    "PriorRouteDeclarationA3",
    "UpstreamPathBreakdownReceiptA3",
    "build_dolma_path_breakdown_a3",
    "build_fineweb_path_breakdown_a3",
    "build_upstream_path_breakdown_a3",
    "canonical_member_order_a3",
    "load_upstream_path_breakdown_a3",
    "load_upstream_path_breakdown_snapshot_a3",
    "observe_hf_tree_files_a3",
    "project_family_resolution_a3",
    "replay_upstream_path_breakdown_a3",
    "write_upstream_path_breakdown_a3",
]
