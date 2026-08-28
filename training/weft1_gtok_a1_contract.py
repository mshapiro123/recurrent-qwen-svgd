"""Amendment-A1 v2 bindings that preserve every banked v1 receipt hash.

This module records what A1 settles and makes its remaining literal gaps
machine-visible.  It intentionally contains no downloader, corpus writer,
tokenizer fit, optimizer constructor, gate minter, or training launcher.
Those actions remain fail-closed until the open bindings at the bottom of this
file are ratified.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from training.weft1_gtok_contract import (
    GTOK_AMENDMENT_A1_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_PIPELINE_RNG_NAMES,
    GTOK_PRETOKENIZER_REGEX,
    GTOK_RUN_RNG_NAME_TEMPLATES,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_STRATUM_TOLERANCE,
    GTOK_TOKENIZER_FAMILY,
    GTOK_TOKENIZER_LIBRARY,
    GTOK_TOKENIZER_MIN_FREQUENCY,
    GTOK_TRAINING_BYTE_BUDGET,
    a1_flat_adamw_recipe,
    execution_authority_v2_bound_sha256,
)


SOURCE_ROUTE_MANIFEST_PATH = Path(__file__).with_name(
    "weft1_gtok_source_routes_20260828.json"
)
SOURCE_FAMILIES = (
    "dolma_web",
    "wikipedia_wikibooks",
    "stackedu",
    "finemath_3plus",
    "arxiv",
    "olmocr",
    "fineweb_edu",
)
SOURCE_FAMILY_TARGET_BYTES = {
    "dolma_web": 6_669_000_000,
    "wikipedia_wikibooks": 3_762_000_000,
    "stackedu": 9_500_000_000,
    "finemath_3plus": 5_700_000_000,
    "arxiv": 5_700_000_000,
    "olmocr": 5_700_000_000,
    "fineweb_edu": 6_669_000_000,
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SourceRouteBindingV2:
    """One A1-R2 resolver binding plus its mechanical admissibility evidence."""

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
    external_locator_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError(f"unknown A1 source family: {self.source_family!r}")
        if self.stratum not in GTOK_STRATA:
            raise ValueError(f"unknown G-TOK stratum: {self.stratum!r}")
        for name in (
            "role",
            "repository",
            "config",
            "split",
            "asset_selector",
            "selection_rule",
            "available_bytes_basis",
            "lineage_evidence",
            "parse_policy",
        ):
            _nonempty(getattr(self, name), name)
        if _SHA1.fullmatch(self.revision) is None:
            raise ValueError("source revision must be an exact commit SHA")
        if self.declared_license != "odc-by":
            raise ValueError("A1 route ledger accepts only its mechanically verified ODC-By rows")
        _sha256(self.card_sha256, "card_sha256")
        parsed = urlsplit(self.card_url)
        expected_path = (
            f"/datasets/{self.repository}/blob/{self.revision}/README.md"
        )
        if (
            parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("card URL must bind the exact repository and revision")
        if type(self.asset_count) is not int or self.asset_count < 1:
            raise ValueError("route requires at least one enumerated source asset")
        if type(self.available_bytes) is not int or self.available_bytes < 1:
            raise ValueError("available_bytes must be a positive exact integer")
        expected_target = SOURCE_FAMILY_TARGET_BYTES[self.source_family]
        if self.required_bytes != expected_target:
            raise ValueError("route required_bytes disagrees with the curriculum target")
        if self.available_bytes <= self.required_bytes:
            raise ValueError("route has no positive byte margin above its target")
        if self.external_locator_manifest_sha256 is not None:
            _sha256(
                self.external_locator_manifest_sha256,
                "external_locator_manifest_sha256",
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceRouteBindingV2":
        if not isinstance(value, Mapping):
            raise TypeError("source route must be a mapping")
        return cls(**dict(value))

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v2_bound_sha256(
            "weft1_gtok_source_route_binding_v2",
            self,
        )


@dataclass(frozen=True)
class SourceRouteManifestV2:
    """Complete ordered A1-R2 route ledger; not the P-B human license approval."""

    schema: str
    authority_sha256: str
    resolved_on: str
    approved_host: str
    license_check_scope: str
    routes: tuple[SourceRouteBindingV2, ...]
    known_route_findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != "weft1_gtok_source_route_manifest_v2":
            raise ValueError("unexpected source-route schema")
        if self.authority_sha256 != GTOK_AMENDMENT_A1_SHA256:
            raise ValueError("source routes are not bound to verified Amendment A1")
        if self.resolved_on != "2026-08-28":
            raise ValueError("source-route resolution date drifted")
        if self.approved_host != "huggingface.co":
            raise ValueError("source-route host allowlist drifted")
        _nonempty(self.license_check_scope, "license_check_scope")
        if not isinstance(self.routes, tuple):
            raise TypeError("routes must be a tuple")
        if any(not isinstance(route, SourceRouteBindingV2) for route in self.routes):
            raise TypeError("routes must contain SourceRouteBindingV2 values")
        if tuple(route.source_family for route in self.routes) != SOURCE_FAMILIES:
            raise ValueError("route ledger must contain every family in canonical order")
        if not isinstance(self.known_route_findings, tuple) or not self.known_route_findings:
            raise ValueError("route ledger must retain its known findings")
        if any(not isinstance(item, str) or not item for item in self.known_route_findings):
            raise TypeError("route findings must be nonempty strings")

    @property
    def manifest_sha256(self) -> str:
        return execution_authority_v2_bound_sha256(self.schema, self)


def load_source_route_manifest(
    path: Path = SOURCE_ROUTE_MANIFEST_PATH,
) -> SourceRouteManifestV2:
    """Load and validate the checked-in exact route ledger."""

    if not isinstance(path, Path):
        raise TypeError("route manifest path must be a pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("route manifest root must be a JSON object")
    routes = tuple(
        SourceRouteBindingV2.from_mapping(item) for item in payload.pop("routes")
    )
    findings = tuple(payload.pop("known_route_findings"))
    return SourceRouteManifestV2(
        routes=routes,
        known_route_findings=findings,
        **payload,
    )


@dataclass(frozen=True)
class StratumFloorReceiptV2:
    """Evidence for one independent document-aligned T or H prefix floor."""

    stream: str
    stratum: str
    target_bytes: int
    realized_bytes: int
    ordered_document_ids_sha256: str
    boundary_document_id_sha256: str
    next_document_byte_count: int | None
    source_exhausted: bool = False

    def __post_init__(self) -> None:
        if self.stream not in {"T", "H"}:
            raise ValueError("screen stream must be T or H")
        expected_rows = dict(
            GTOK_SCREEN_TRAIN_STRATUM_TARGETS
            if self.stream == "T"
            else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
        )
        if self.stratum not in expected_rows:
            raise ValueError("unknown screen stratum")
        if self.target_bytes != expected_rows[self.stratum]:
            raise ValueError("stratum target disagrees with A1-R3")
        if type(self.realized_bytes) is not int or not 0 < self.realized_bytes <= self.target_bytes:
            raise ValueError("realized bytes must be a positive document floor")
        _sha256(self.ordered_document_ids_sha256, "ordered_document_ids_sha256")
        _sha256(self.boundary_document_id_sha256, "boundary_document_id_sha256")
        if type(self.source_exhausted) is not bool:
            raise TypeError("source_exhausted must be boolean")
        shortfall = self.target_bytes - self.realized_bytes
        if Fraction(shortfall, self.target_bytes) > GTOK_STRATUM_TOLERANCE:
            raise ValueError("stratum document-floor shortfall exceeds 0.5 percent")
        if shortfall:
            if self.source_exhausted:
                if self.next_document_byte_count is not None:
                    raise ValueError("an exhausted source cannot name a next document")
            else:
                if (
                    type(self.next_document_byte_count) is not int
                    or self.next_document_byte_count < 1
                ):
                    raise ValueError("a non-exhausted floor must record the next document")
                if self.realized_bytes + self.next_document_byte_count <= self.target_bytes:
                    raise ValueError("recorded floor is not maximal in its fixed document order")
        elif self.next_document_byte_count is not None:
            raise ValueError("an exact target must not claim a rejected boundary document")

    @property
    def shortfall_bytes(self) -> int:
        return self.target_bytes - self.realized_bytes


@dataclass(frozen=True)
class ScreenCorpusReceiptV2:
    """A1's independent T/H floors and disjointness evidence."""

    training: tuple[StratumFloorReceiptV2, ...]
    heldout: tuple[StratumFloorReceiptV2, ...]
    training_stream_sha256: str
    heldout_stream_sha256: str
    document_overlap_count: int
    cluster_overlap_count: int

    def __post_init__(self) -> None:
        for rows, stream in ((self.training, "T"), (self.heldout, "H")):
            if not isinstance(rows, tuple) or len(rows) != len(GTOK_STRATA):
                raise ValueError(f"{stream} requires one row per stratum")
            if any(not isinstance(row, StratumFloorReceiptV2) for row in rows):
                raise TypeError("screen floors must be StratumFloorReceiptV2 values")
            if tuple(row.stream for row in rows) != (stream,) * len(GTOK_STRATA):
                raise ValueError("screen floor appears in the wrong stream")
            if tuple(row.stratum for row in rows) != GTOK_STRATA:
                raise ValueError("screen floors require canonical stratum order")
        _sha256(self.training_stream_sha256, "training_stream_sha256")
        _sha256(self.heldout_stream_sha256, "heldout_stream_sha256")
        if self.training_stream_sha256 == self.heldout_stream_sha256:
            raise ValueError("T and H stream hashes must differ")
        for name in ("document_overlap_count", "cluster_overlap_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) != 0:
                raise ValueError("T and H must be document- and cluster-disjoint")
        if self.training_target_bytes != GTOK_TRAINING_BYTE_BUDGET:
            raise RuntimeError("training target no longer equals four billion bytes")
        if self.heldout_target_bytes != GTOK_HELDOUT_BYTE_TARGET:
            raise RuntimeError("held-out target no longer equals eighty million bytes")

    @property
    def training_target_bytes(self) -> int:
        return sum(row.target_bytes for row in self.training)

    @property
    def training_realized_bytes(self) -> int:
        return sum(row.realized_bytes for row in self.training)

    @property
    def heldout_target_bytes(self) -> int:
        return sum(row.target_bytes for row in self.heldout)

    @property
    def heldout_realized_bytes(self) -> int:
        return sum(row.realized_bytes for row in self.heldout)

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v2_bound_sha256(
            "weft1_gtok_screen_corpus_receipt_v2",
            self,
        )


@dataclass(frozen=True)
class DedupBindingV2:
    """Settled A1 dimensions plus explicit unresolved production details."""

    match_normalization: str = "NFC_plus_whitespace_collapse_match_only"
    exact_digest: str = "sha1"
    shingle_width_bytes: int = 13
    minhash_components: int = 128
    lsh_bands: int = 16
    lsh_rows_per_band: int = 8
    jaccard_threshold: Fraction = Fraction(4, 5)
    keep_source: str = "dolma_web"
    drop_source: str = "fineweb_edu"

    def __post_init__(self) -> None:
        if (
            self.match_normalization != "NFC_plus_whitespace_collapse_match_only"
            or self.exact_digest != "sha1"
            or self.shingle_width_bytes != 13
            or self.minhash_components != 128
            or self.lsh_bands != 16
            or self.lsh_rows_per_band != 8
            or self.lsh_bands * self.lsh_rows_per_band != self.minhash_components
            or self.jaccard_threshold != Fraction(4, 5)
            or self.keep_source != "dolma_web"
            or self.drop_source != "fineweb_edu"
        ):
            raise ValueError("dedup binding must equal Amendment A1's settled dimensions")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v2_bound_sha256(
            "weft1_gtok_dedup_binding_v2",
            self,
        )


@dataclass(frozen=True)
class TokenizerBindingV2:
    """A1's tokenizer headline recipe; literal serialization remains blocked."""

    library: str = GTOK_TOKENIZER_LIBRARY
    family: str = GTOK_TOKENIZER_FAMILY
    pretokenizer_regex: str = GTOK_PRETOKENIZER_REGEX
    min_frequency: int = GTOK_TOKENIZER_MIN_FREQUENCY
    training_scope: str = "full_realized_T"
    lowercase: bool = False
    nfkc: bool = False
    bpe_dropout: float = 0.0
    byte_atom_count: int = 256
    numeric_split: str = "single_digit"

    def __post_init__(self) -> None:
        expected = (
            GTOK_TOKENIZER_LIBRARY,
            GTOK_TOKENIZER_FAMILY,
            GTOK_PRETOKENIZER_REGEX,
            GTOK_TOKENIZER_MIN_FREQUENCY,
            "full_realized_T",
            False,
            False,
            0.0,
            256,
            "single_digit",
        )
        actual = (
            self.library,
            self.family,
            self.pretokenizer_regex,
            self.min_frequency,
            self.training_scope,
            self.lowercase,
            self.nfkc,
            self.bpe_dropout,
            self.byte_atom_count,
            self.numeric_split,
        )
        if actual != expected:
            raise ValueError("tokenizer binding drifted from Amendment A1")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v2_bound_sha256(
            "weft1_gtok_tokenizer_binding_v2",
            self,
        )


A1_DEDUP_BINDING = DedupBindingV2()
A1_TOKENIZER_BINDING = TokenizerBindingV2()
A1_OPTIMIZER_RECIPE = a1_flat_adamw_recipe()
A1_OPTIMIZER_RECIPE_SHA256 = execution_authority_v2_bound_sha256(
    "weft1_gtok_flat_adamw_recipe_v2",
    A1_OPTIMIZER_RECIPE,
)

A1_EXECUTION_DEFECTS = (
    "run termination and 0.25/0.5/1.0 milestones when document-floor T is below four billion bytes",
    "language-ID package, model, version, exact threshold, and boundary tie behavior",
    "literal NFC whitespace set/collapse/newline/trim policy and Unicode version",
    "MinHash hash/permutation family, root seed, framing, short-document rule, candidate ordering, and recall audit",
    "shard serializer, compression, record framing, timestamp policy, and manifest self-hash exclusions",
    "tokenizers version and dependency lock; Split/ByteLevel flags; decoder/post-processor; initial alphabet; ordered special tokens, IDs, roles, and AddedToken flags",
    "campaign/corpus root seed, two numeric training seeds, canonical arm formatting, and the non-existent BpeTrainer seed hook",
    "quality-ranking tie semantics where the bound source lacks an explicit scalar score and re-deduplication of FineWeb top-ups",
    "D1/D2 independent-process receipts and authoritative D3-D6 production evidence",
    "undertrained-row threshold; packing/final-batch/scheduler-step semantics; runtime, FLOP, memory, throughput, and latency receipt schemas",
    "pre-dispatch and in-flight enforcement of the cumulative 12 A100-hour tripwire",
)


class A1ExecutionBlocked(RuntimeError):
    """Raised when a still-unbound A1 execution action is attempted."""


def require_a1_execution_ready(action: str) -> None:
    """Fail closed while any literal A1 execution defect remains."""

    _nonempty(action, "action")
    raise A1ExecutionBlocked(
        f"{action} is not reproducibly bound: " + "; ".join(A1_EXECUTION_DEFECTS)
    )


def a1_contract_snapshot() -> dict[str, Any]:
    """Return the hash-addressable, non-executable A1 contract surface."""

    routes = load_source_route_manifest()
    return {
        "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
        "dedup_binding_sha256": A1_DEDUP_BINDING.receipt_sha256,
        "execution_defects": A1_EXECUTION_DEFECTS,
        "optimizer_recipe_sha256": A1_OPTIMIZER_RECIPE_SHA256,
        "pipeline_rng_names": GTOK_PIPELINE_RNG_NAMES,
        "route_manifest_sha256": routes.manifest_sha256,
        "run_rng_name_templates": GTOK_RUN_RNG_NAME_TEMPLATES,
        "tokenizer_binding_sha256": A1_TOKENIZER_BINDING.receipt_sha256,
    }


def a1_contract_snapshot_sha256() -> str:
    return execution_authority_v2_bound_sha256(
        "weft1_gtok_a1_contract_snapshot_v2",
        a1_contract_snapshot(),
    )


__all__ = [
    "A1ExecutionBlocked",
    "A1_DEDUP_BINDING",
    "A1_EXECUTION_DEFECTS",
    "A1_OPTIMIZER_RECIPE",
    "A1_OPTIMIZER_RECIPE_SHA256",
    "A1_TOKENIZER_BINDING",
    "DedupBindingV2",
    "SOURCE_FAMILIES",
    "SOURCE_FAMILY_TARGET_BYTES",
    "SOURCE_ROUTE_MANIFEST_PATH",
    "ScreenCorpusReceiptV2",
    "SourceRouteBindingV2",
    "SourceRouteManifestV2",
    "StratumFloorReceiptV2",
    "TokenizerBindingV2",
    "a1_contract_snapshot",
    "a1_contract_snapshot_sha256",
    "load_source_route_manifest",
    "require_a1_execution_ready",
]
