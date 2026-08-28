"""Fail-closed WEFT-1 corpus preflight and draft D1-D6 diagnostics.

The 2026-08-28 handoff authorizes a bounded corpus/tokenizer run axis, but it
does not yet provide enough literal bindings to execute that authority.  This
module therefore implements only choice-independent schemas and explicitly
non-authoritative diagnostics.  It cannot mint a passing D1-D6 gate receipt.

In particular, this file performs no network access, corpus materialization,
tokenizer fitting, optimizer construction, model training, sealed-data read,
or checkpoint write.  Reference normalization, MinHash, and framing helpers
are fixtures for implementation tests; their outputs are not production
evidence until strategy binds exact algorithms and an execution-envelope v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import ipaddress
import re
from typing import Callable, Mapping, NoReturn, Sequence
from urllib.parse import urlsplit

from training.weft1_gtok_contract import (
    GTOK_EXECUTION_AUTHORITY_CHAIN,
    GTOK_ROUND_TRIP_CATEGORIES,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_VOCABULARY_ARMS,
    canonical_json_bytes,
    execution_authority_bound_sha256,
    require_gtok_execution_authority,
)


# These are handoff target centers.  They do not settle the still-open
# whole-document rounding, tolerance denominator, or train/held-out join.
CORPUS_TOTAL_TARGET_BYTES = 38_000_000_000
CORPUS_STRATUM_TARGETS = (
    ("general", 17_100_000_000),
    ("code", 9_500_000_000),
    ("mathematics", 5_700_000_000),
    ("science_technical", 5_700_000_000),
)
GENERAL_SOURCE_TARGETS = (
    ("wikipedia_wikibooks", 3_762_000_000),
    ("dolma_web", 6_669_000_000),
    ("fineweb_edu", 6_669_000_000),
)
GTOK_SCREEN_TARGET_BYTES = 4_000_000_000
BYTE_SHINGLE_WIDTH = 13
NEAR_DUPLICATE_JACCARD_THRESHOLD = Fraction(4, 5)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHARD_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REPO_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_DEVICES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_MAX_REFERENCE_MINHASH_COMPONENTS = 4096


def _require_sha1(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character SHA-1")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an exact nonempty string")
    return value


def _require_run_id(value: str) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must use canonical lowercase ASCII identifier syntax")
    return value


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    return hashlib.sha256(value).hexdigest()


def _is_windows_device_segment(value: str) -> bool:
    return value.rstrip(". ").split(".", 1)[0].casefold() in _WINDOWS_DEVICES


def _canonical_shard_path(value: str) -> str:
    _nonempty(value, "relative shard path")
    if "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("shard paths must be canonical relative POSIX paths")
    parts = value.split("/")
    if any(
        _SHARD_SEGMENT.fullmatch(part) is None or part.endswith(".")
        for part in parts
    ):
        raise ValueError("shard paths must use lowercase ASCII safe segments")
    if any(_is_windows_device_segment(part) for part in parts):
        raise ValueError("shard paths may not use Windows device aliases")
    return value


def _safe_repo_path(value: str) -> str:
    _nonempty(value, "repository path")
    if "\\" in value or value.startswith("/") or "\x00" in value:
        raise ValueError("repository path must be a relative POSIX path")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."}
        or _REPO_PATH_SEGMENT.fullmatch(part) is None
        or part.endswith(".")
        or _is_windows_device_segment(part)
        for part in parts
    ):
        raise ValueError("repository path contains a noncanonical segment")
    return value


def _canonical_draft_https_uri(value: str) -> str:
    _nonempty(value, "HTTPS URI")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("source URI may not contain control characters")
    parsed = urlsplit(value)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("source URI has an invalid host or port") from error
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError("source URI must be canonical HTTPS without credentials/port/query")
    if hostname != hostname.casefold() or hostname.endswith("."):
        raise ValueError("source URI host must use canonical lowercase DNS form")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("source URI must use an authority-bound DNS host, not an IP literal")
    labels = hostname.split(".")
    if (
        len(labels) < 2
        or hostname == "localhost"
        or any(
            not label
            or len(label) > 63
            or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) is None
            for label in labels
        )
    ):
        raise ValueError("source URI host must use canonical lowercase DNS form")
    if (
        "\\" in parsed.path
        or "%" in parsed.path
        or "//" in parsed.path
    ):
        raise ValueError("source URI path contains an unsafe segment")
    _safe_repo_path(parsed.path[1:])
    if value != f"https://{hostname}{parsed.path}":
        raise ValueError("source URI is not in exact canonical form")
    return value


@dataclass(frozen=True)
class AlgorithmSpec:
    """Draft identity of an algorithm; not evidence that the code executed it."""

    name: str
    schema_version: str
    exact_spec_sha256: str
    implementation_tree_sha256: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.name, "algorithm name")
        _nonempty(self.schema_version, "algorithm schema_version")
        for field_name in (
            "exact_spec_sha256",
            "implementation_tree_sha256",
            "dependency_lock_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256("weft1_draft_algorithm_spec_v1", self)


@dataclass(frozen=True)
class SourceAsset:
    """Draft immutable source asset; route authority is intentionally not inferred."""

    source_family: str
    requested_repository: str
    resolved_repository: str
    revision: str
    config: str
    split: str
    locator_kind: str
    locator: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_family",
            "requested_repository",
            "resolved_repository",
            "config",
            "split",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _require_sha1(self.revision, "revision")
        if self.locator_kind == "repo_path":
            _safe_repo_path(self.locator)
        elif self.locator_kind == "https_uri":
            _canonical_draft_https_uri(self.locator)
        else:
            raise ValueError("locator_kind must be repo_path or https_uri")
        if type(self.byte_size) is not int or self.byte_size < 1:
            raise ValueError("source asset byte_size must be a positive exact integer")
        _require_sha256(self.sha256, "source asset sha256")

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256("weft1_draft_source_asset_v1", self)


@dataclass(frozen=True)
class DocumentRecord:
    """Retained source bytes with an asset-provenanced stable identity."""

    source_asset: SourceAsset
    stratum: str
    stable_source_record_id: str
    retained_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source_asset, SourceAsset):
            raise TypeError("source_asset must be a typed SourceAsset")
        if self.stratum not in GTOK_STRATA:
            raise ValueError(f"unknown corpus stratum: {self.stratum!r}")
        _nonempty(self.stable_source_record_id, "stable_source_record_id")
        if not isinstance(self.retained_bytes, bytes):
            raise TypeError("retained_bytes must be bytes")
        if not self.retained_bytes:
            raise ValueError("retained_bytes must contain at least one byte")

    @property
    def retained_sha256(self) -> str:
        return sha256_bytes(self.retained_bytes)

    @property
    def source(self) -> str:
        return self.source_asset.source_family

    @property
    def source_asset_draft_sha256(self) -> str:
        return self.source_asset.draft_sha256

    @property
    def doc_id(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_document_identity_v1",
            {
                "retained_sha256": self.retained_sha256,
                "source": self.source,
                "source_asset_draft_sha256": self.source_asset_draft_sha256,
                "stable_source_record_id": self.stable_source_record_id,
            },
        )


@dataclass(frozen=True)
class NormalizedDocumentDiagnostic:
    """Caller-supplied reference transform output, explicitly not production proof."""

    document: DocumentRecord
    normalization_spec: AlgorithmSpec
    normalized_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentRecord):
            raise TypeError("document must be a DocumentRecord")
        if not isinstance(self.normalization_spec, AlgorithmSpec):
            raise TypeError("normalization_spec must be an AlgorithmSpec")
        if not isinstance(self.normalized_bytes, bytes):
            raise TypeError("normalized_bytes must be bytes")

    @property
    def normalized_sha1(self) -> str:
        return hashlib.sha1(self.normalized_bytes).hexdigest()  # noqa: S324 - candidate only

    @property
    def normalized_sha256(self) -> str:
        return sha256_bytes(self.normalized_bytes)

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_normalization_diagnostic_v1",
            {
                "doc_id": self.document.doc_id,
                "normalization_spec_draft_sha256": self.normalization_spec.draft_sha256,
                "normalized_byte_count": len(self.normalized_bytes),
                "normalized_sha1": self.normalized_sha1,
                "normalized_sha256": self.normalized_sha256,
                "retained_sha256": self.document.retained_sha256,
            },
        )


@dataclass(frozen=True)
class ReferenceDedupBinding:
    """Parameters for quarantined fixture code, not the production binding."""

    normalization: AlgorithmSpec
    minhash: AlgorithmSpec
    shingle_width: int
    minhash_components: int
    minhash_seed: int
    lsh_bands: int
    lsh_rows_per_band: int
    jaccard_threshold: Fraction
    uint64_endianness: str = "little"

    def __post_init__(self) -> None:
        if not isinstance(self.normalization, AlgorithmSpec):
            raise TypeError("normalization must be an AlgorithmSpec")
        if not isinstance(self.minhash, AlgorithmSpec):
            raise TypeError("minhash must be an AlgorithmSpec")
        if self.shingle_width != BYTE_SHINGLE_WIDTH:
            raise ValueError("the handoff binds byte-level 13-gram shingles")
        if (
            type(self.minhash_components) is not int
            or not 1 <= self.minhash_components <= _MAX_REFERENCE_MINHASH_COMPONENTS
        ):
            raise ValueError("reference MinHash component count is outside its safe bound")
        for name in ("lsh_bands", "lsh_rows_per_band"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.lsh_bands * self.lsh_rows_per_band != self.minhash_components:
            raise ValueError("LSH bands times rows must equal MinHash components")
        if type(self.minhash_seed) is not int or not 0 <= self.minhash_seed < 1 << 64:
            raise ValueError("reference MinHash seed must be an unsigned 64-bit integer")
        if self.jaccard_threshold != NEAR_DUPLICATE_JACCARD_THRESHOLD:
            raise ValueError("the handoff binds exact Jaccard >= 4/5")
        if self.uint64_endianness != "little":
            raise ValueError("the reference implementation binds little-endian uint64")

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256("weft1_draft_dedup_binding_v1", self)

    @property
    def nominal_ideal_minhash_candidate_probability(self) -> float:
        """Ideal-family formula only; not measured recall for the reference hash."""

        similarity = float(self.jaccard_threshold)
        return 1.0 - (
            1.0 - similarity**self.lsh_rows_per_band
        ) ** self.lsh_bands


def byte_shingles(value: bytes, width: int = BYTE_SHINGLE_WIDTH) -> frozenset[bytes]:
    """Reference overlapping byte shingles with explicit short-document behavior."""

    if not isinstance(value, bytes):
        raise TypeError("byte shingles require bytes")
    if type(width) is not int or width < 1:
        raise ValueError("shingle width must be a positive exact integer")
    if len(value) <= width:
        return frozenset((value,))
    return frozenset(value[index : index + width] for index in range(len(value) - width + 1))


def exact_set_jaccard(left: frozenset[bytes], right: frozenset[bytes]) -> Fraction:
    if not isinstance(left, frozenset) or not isinstance(right, frozenset):
        raise TypeError("exact_set_jaccard requires frozensets")
    union = left | right
    if not union:
        return Fraction(1, 1)
    return Fraction(len(left & right), len(union))


def reference_minhash_signature(
    shingles: frozenset[bytes], *, components: int, seed: int
) -> tuple[int, ...]:
    """Stable fixture signature; not asserted to be a production MinHash family."""

    if not isinstance(shingles, frozenset) or not shingles:
        raise ValueError("reference MinHash requires a nonempty frozenset")
    if type(components) is not int or not 1 <= components <= _MAX_REFERENCE_MINHASH_COMPONENTS:
        raise ValueError("reference MinHash component count is outside its safe bound")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("reference MinHash seed must be an unsigned 64-bit integer")
    seed_bytes = seed.to_bytes(8, "little")
    signature: list[int] = []
    for component in range(components):
        component_bytes = component.to_bytes(4, "little")
        signature.append(
            min(
                int.from_bytes(
                    hashlib.sha256(seed_bytes + component_bytes + shingle).digest()[:8],
                    "little",
                )
                for shingle in shingles
            )
        )
    return tuple(signature)


def _reference_lsh_keys(
    signature: tuple[int, ...], *, bands: int, rows_per_band: int
) -> tuple[bytes, ...]:
    if len(signature) != bands * rows_per_band:
        raise ValueError("signature length disagrees with LSH shape")
    return tuple(
        b"".join(
            value.to_bytes(8, "little", signed=False)
            for value in signature[
                band * rows_per_band : (band + 1) * rows_per_band
            ]
        )
        for band in range(bands)
    )


@dataclass(frozen=True)
class DedupDecisionDiagnostic:
    fineweb_doc_id: str
    canonical_dolma_doc_id: str
    route: str
    exact_jaccard_numerator: int
    exact_jaccard_denominator: int
    disposition: str = "drop_fineweb_keep_dolma"

    def __post_init__(self) -> None:
        _require_sha256(self.fineweb_doc_id, "fineweb_doc_id")
        _require_sha256(self.canonical_dolma_doc_id, "canonical_dolma_doc_id")
        if self.route not in {"exact", "near"}:
            raise ValueError("dedup route must be exact or near")
        if type(self.exact_jaccard_numerator) is not int or self.exact_jaccard_numerator < 0:
            raise ValueError("Jaccard numerator must be a non-negative exact integer")
        if type(self.exact_jaccard_denominator) is not int or self.exact_jaccard_denominator < 1:
            raise ValueError("Jaccard denominator must be a positive exact integer")
        score = Fraction(self.exact_jaccard_numerator, self.exact_jaccard_denominator)
        if (
            self.exact_jaccard_numerator != score.numerator
            or self.exact_jaccard_denominator != score.denominator
        ):
            raise ValueError("Jaccard fraction must use its canonical reduced form")
        if score > 1:
            raise ValueError("Jaccard score may not exceed one")
        if self.route == "exact" and (
            self.exact_jaccard_numerator,
            self.exact_jaccard_denominator,
        ) != (1, 1):
            raise ValueError("exact-route decisions must carry Jaccard 1/1")
        if self.route == "near" and score < NEAR_DUPLICATE_JACCARD_THRESHOLD:
            raise ValueError("near-route decisions must satisfy exact Jaccard >= 4/5")
        if self.disposition != "drop_fineweb_keep_dolma":
            raise ValueError("the handoff fixes Dolma as the canonical copy")


@dataclass(frozen=True)
class DedupDiagnosticLedger:
    run_id: str
    binding_draft_sha256: str
    dolma_input_manifest_sha256: str
    fineweb_input_manifest_sha256: str
    dolma_document_count: int
    fineweb_document_count: int
    candidate_pair_count: int
    decisions: tuple[DedupDecisionDiagnostic, ...]

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        for name in (
            "binding_draft_sha256",
            "dolma_input_manifest_sha256",
            "fineweb_input_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("dolma_document_count", "fineweb_document_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.candidate_pair_count) is not int or self.candidate_pair_count < 0:
            raise ValueError("candidate_pair_count must be a non-negative exact integer")
        if not isinstance(self.decisions, tuple):
            raise TypeError("dedup decisions must be a tuple")
        if any(not isinstance(item, DedupDecisionDiagnostic) for item in self.decisions):
            raise TypeError("dedup ledger contains a non-decision value")
        if len(self.decisions) > self.fineweb_document_count:
            raise ValueError("dedup decisions exceed the FineWeb input count")
        near_decisions = sum(item.route == "near" for item in self.decisions)
        if self.candidate_pair_count < near_decisions:
            raise ValueError("candidate pair count is smaller than near-duplicate decisions")
        if tuple(sorted(self.decisions, key=lambda item: item.fineweb_doc_id)) != self.decisions:
            raise ValueError("dedup decisions must use canonical FineWeb order")
        if len({item.fineweb_doc_id for item in self.decisions}) != len(self.decisions):
            raise ValueError("a FineWeb document may be dropped at most once")

    @property
    def output_identity_sha256(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_dedup_output_v1",
            {
                "binding_draft_sha256": self.binding_draft_sha256,
                "candidate_pair_count": self.candidate_pair_count,
                "decisions": self.decisions,
                "dolma_document_count": self.dolma_document_count,
                "dolma_input_manifest_sha256": self.dolma_input_manifest_sha256,
                "fineweb_document_count": self.fineweb_document_count,
                "fineweb_input_manifest_sha256": self.fineweb_input_manifest_sha256,
            },
        )


def build_reference_cross_source_dedup_diagnostic(
    dolma_documents: Sequence[NormalizedDocumentDiagnostic],
    fineweb_documents: Sequence[NormalizedDocumentDiagnostic],
    binding: ReferenceDedupBinding,
    *,
    run_id: str,
) -> DedupDiagnosticLedger:
    """Exercise reference exact/near-dedup logic without minting gate evidence."""

    _require_run_id(run_id)
    if not isinstance(binding, ReferenceDedupBinding):
        raise TypeError("binding must be a ReferenceDedupBinding")
    if not dolma_documents or not fineweb_documents:
        raise ValueError("reference cross-source dedup requires both source sides")
    if any(item.document.source != "dolma_web" for item in dolma_documents):
        raise ValueError("canonical side must contain only dolma_web documents")
    if any(item.document.source != "fineweb_edu" for item in fineweb_documents):
        raise ValueError("drop side must contain only fineweb_edu documents")
    if any(
        item.normalization_spec.draft_sha256 != binding.normalization.draft_sha256
        for item in (*dolma_documents, *fineweb_documents)
    ):
        raise ValueError("normalized diagnostics do not share the bound reference normalizer")
    all_ids = [item.document.doc_id for item in (*dolma_documents, *fineweb_documents)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dedup input repeats a stable document identity")

    dolma = tuple(sorted(dolma_documents, key=lambda item: item.document.doc_id))
    fineweb = tuple(sorted(fineweb_documents, key=lambda item: item.document.doc_id))
    exact_index: dict[tuple[str, int, str], list[NormalizedDocumentDiagnostic]] = {}
    for item in dolma:
        key = (item.normalized_sha1, len(item.normalized_bytes), item.normalized_sha256)
        exact_index.setdefault(key, []).append(item)

    dolma_shingles = {
        item.document.doc_id: byte_shingles(item.normalized_bytes, binding.shingle_width)
        for item in dolma
    }
    lsh: list[dict[bytes, list[str]]] = [dict() for _ in range(binding.lsh_bands)]
    for item in dolma:
        signature = reference_minhash_signature(
            dolma_shingles[item.document.doc_id],
            components=binding.minhash_components,
            seed=binding.minhash_seed,
        )
        for band, key in enumerate(
            _reference_lsh_keys(
                signature,
                bands=binding.lsh_bands,
                rows_per_band=binding.lsh_rows_per_band,
            )
        ):
            lsh[band].setdefault(key, []).append(item.document.doc_id)

    decisions: list[DedupDecisionDiagnostic] = []
    candidate_pairs = 0
    for item in fineweb:
        exact_key = (item.normalized_sha1, len(item.normalized_bytes), item.normalized_sha256)
        exact_matches = tuple(
            candidate
            for candidate in exact_index.get(exact_key, ())
            if candidate.normalized_bytes == item.normalized_bytes
        )
        if exact_matches:
            winner = min(exact_matches, key=lambda candidate: candidate.document.doc_id)
            decisions.append(
                DedupDecisionDiagnostic(
                    fineweb_doc_id=item.document.doc_id,
                    canonical_dolma_doc_id=winner.document.doc_id,
                    route="exact",
                    exact_jaccard_numerator=1,
                    exact_jaccard_denominator=1,
                )
            )
            continue

        shingles = byte_shingles(item.normalized_bytes, binding.shingle_width)
        signature = reference_minhash_signature(
            shingles,
            components=binding.minhash_components,
            seed=binding.minhash_seed,
        )
        candidates: set[str] = set()
        for band, key in enumerate(
            _reference_lsh_keys(
                signature,
                bands=binding.lsh_bands,
                rows_per_band=binding.lsh_rows_per_band,
            )
        ):
            candidates.update(lsh[band].get(key, ()))
        candidate_pairs += len(candidates)
        accepted: list[tuple[Fraction, str]] = []
        for candidate_id in sorted(candidates):
            score = exact_set_jaccard(shingles, dolma_shingles[candidate_id])
            if score >= binding.jaccard_threshold:
                accepted.append((score, candidate_id))
        if accepted:
            score, winner_id = min(accepted, key=lambda row: (-row[0], row[1]))
            decisions.append(
                DedupDecisionDiagnostic(
                    fineweb_doc_id=item.document.doc_id,
                    canonical_dolma_doc_id=winner_id,
                    route="near",
                    exact_jaccard_numerator=score.numerator,
                    exact_jaccard_denominator=score.denominator,
                )
            )

    return DedupDiagnosticLedger(
        run_id=run_id,
        binding_draft_sha256=binding.draft_sha256,
        dolma_input_manifest_sha256=sha256_bytes(
            canonical_json_bytes(tuple(item.draft_sha256 for item in dolma))
        ),
        fineweb_input_manifest_sha256=sha256_bytes(
            canonical_json_bytes(tuple(item.draft_sha256 for item in fineweb))
        ),
        dolma_document_count=len(dolma),
        fineweb_document_count=len(fineweb),
        candidate_pair_count=candidate_pairs,
        decisions=tuple(sorted(decisions, key=lambda item: item.fineweb_doc_id)),
    )


@dataclass(frozen=True)
class ShardDiagnostic:
    relative_path: str
    serializer_spec_draft_sha256: str
    logical_stream_sha256: str
    retained_byte_count: int
    record_count: int
    file_sha256: str
    file_size: int
    compressed_file_sha256: str | None = None
    compressed_file_size: int | None = None

    def __post_init__(self) -> None:
        _canonical_shard_path(self.relative_path)
        for name in ("serializer_spec_draft_sha256", "logical_stream_sha256", "file_sha256"):
            _require_sha256(getattr(self, name), name)
        for name in ("retained_byte_count", "record_count", "file_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        compressed = (self.compressed_file_sha256, self.compressed_file_size)
        if (compressed[0] is None) != (compressed[1] is None):
            raise ValueError("compressed file hash and size must be both present or absent")
        if compressed[0] is not None:
            _require_sha256(compressed[0], "compressed_file_sha256")
            if type(compressed[1]) is not int or compressed[1] < 1:
                raise ValueError("compressed_file_size must be a positive exact integer")


@dataclass(frozen=True)
class BuildRunDiagnostic:
    run_id: str
    source_asset_manifest_sha256: str
    algorithm_binding_manifest_sha256: str
    selection_spec_sha256: str
    shards: tuple[ShardDiagnostic, ...]

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        for name in (
            "source_asset_manifest_sha256",
            "algorithm_binding_manifest_sha256",
            "selection_spec_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.shards, tuple) or not self.shards:
            raise ValueError("build run requires a nonempty shard tuple")
        if any(not isinstance(item, ShardDiagnostic) for item in self.shards):
            raise TypeError("build run contains a non-shard value")
        paths = tuple(item.relative_path for item in self.shards)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("shards must use unique canonical path order")

    @property
    def input_identity_sha256(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_build_inputs_v1",
            {
                "algorithm_binding_manifest_sha256": self.algorithm_binding_manifest_sha256,
                "selection_spec_sha256": self.selection_spec_sha256,
                "source_asset_manifest_sha256": self.source_asset_manifest_sha256,
            },
        )

    @property
    def output_identity_sha256(self) -> str:
        return execution_authority_bound_sha256("weft1_draft_build_outputs_v1", self.shards)


@dataclass(frozen=True)
class DraftDiagnosticReceipt:
    """A draft comparison result that is structurally unable to assert gate PASS."""

    gate: str
    status: str
    named_input_sha256s: tuple[tuple[str, str], ...]
    evidence: tuple[tuple[str, object], ...]
    authoritative: bool = False

    def __post_init__(self) -> None:
        if self.gate not in {"D1", "D2", "D3", "D4", "D5", "D6"}:
            raise ValueError("unknown corpus/G-TOK diagnostic")
        if self.status not in {"DRAFT_CONSISTENT", "DESCRIPTIVE_ONLY"}:
            raise ValueError("draft diagnostic status cannot represent an authoritative pass")
        if self.authoritative is not False:
            raise ValueError("draft diagnostic may never be authoritative")
        for rows, label in (
            (self.named_input_sha256s, "named inputs"),
            (self.evidence, "evidence"),
        ):
            if not isinstance(rows, tuple) or not rows:
                raise ValueError(f"draft diagnostic {label} must be nonempty")
            if any(not isinstance(row, tuple) or len(row) != 2 for row in rows):
                raise TypeError(f"draft diagnostic {label} rows must be pairs")
            keys = tuple(row[0] for row in rows)
            if any(not isinstance(key, str) or not key for key in keys):
                raise TypeError(f"draft diagnostic {label} keys must be nonempty strings")
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise ValueError(f"draft diagnostic {label} keys must be unique and sorted")
        for unused_name, digest in self.named_input_sha256s:
            _require_sha256(digest, "draft diagnostic input sha256")
        canonical_json_bytes(self.evidence)

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256(
            f"weft1_draft_{self.gate.lower()}_diagnostic_v1", self
        )


def diagnose_d1_reproduction(
    first: BuildRunDiagnostic, second: BuildRunDiagnostic
) -> DraftDiagnosticReceipt:
    if first.run_id == second.run_id:
        raise ValueError("D1 requires two distinct build run IDs")
    if first.input_identity_sha256 != second.input_identity_sha256:
        raise ValueError("D1 failed: independent runs used different inputs")
    if first.output_identity_sha256 != second.output_identity_sha256:
        raise ValueError("D1 failed: independent runs produced different shards")
    return DraftDiagnosticReceipt(
        gate="D1",
        status="DRAFT_CONSISTENT",
        named_input_sha256s=(
            ("first_run_inputs", first.input_identity_sha256),
            ("second_run_inputs", second.input_identity_sha256),
        ),
        evidence=(
            ("first_run_id", first.run_id),
            ("output_identity_sha256", first.output_identity_sha256),
            ("second_run_id", second.run_id),
            ("shard_count", len(first.shards)),
        ),
    )


def diagnose_d2_dedup_reproduction(
    first: DedupDiagnosticLedger, second: DedupDiagnosticLedger
) -> DraftDiagnosticReceipt:
    if first.run_id == second.run_id:
        raise ValueError("D2 requires two distinct dedup run IDs")
    if first.output_identity_sha256 != second.output_identity_sha256:
        raise ValueError("D2 failed: independent dedup diagnostics differ")
    return DraftDiagnosticReceipt(
        gate="D2",
        status="DRAFT_CONSISTENT",
        named_input_sha256s=(
            ("first_dedup_output", first.output_identity_sha256),
            ("second_dedup_output", second.output_identity_sha256),
        ),
        evidence=(
            ("decision_count", len(first.decisions)),
            ("first_run_id", first.run_id),
            ("second_run_id", second.run_id),
        ),
    )


def _relative_error(observed: int, target: int) -> Fraction:
    if type(observed) is not int or observed < 0:
        raise ValueError("observed bytes must be a non-negative exact integer")
    return Fraction(abs(observed - target), target)


def describe_d3_composition(
    observed_strata: Mapping[str, int],
    observed_general_sources: Mapping[str, int],
    *,
    composition_input_sha256: str,
) -> DraftDiagnosticReceipt:
    """Report exact deviations without choosing the unresolved tolerance semantics."""

    _require_sha256(composition_input_sha256, "composition input sha256")
    expected_strata = dict(CORPUS_STRATUM_TARGETS)
    expected_general = dict(GENERAL_SOURCE_TARGETS)
    if set(observed_strata) != set(expected_strata):
        raise ValueError("D3 requires all and only the four corpus strata")
    if set(observed_general_sources) != set(expected_general):
        raise ValueError("D3 requires the exact three-way general-source split")
    if any(
        type(value) is not int or value < 0
        for value in (*observed_strata.values(), *observed_general_sources.values())
    ):
        raise ValueError("D3 observed byte counts must be non-negative exact integers")
    if sum(observed_general_sources.values()) != observed_strata["general"]:
        raise ValueError("D3 failed: general-source bytes do not reconcile to general bytes")
    stratum_rows = tuple(
        (name, observed_strata[name], target, _relative_error(observed_strata[name], target))
        for name, target in CORPUS_STRATUM_TARGETS
    )
    general_rows = tuple(
        (
            name,
            observed_general_sources[name],
            target,
            _relative_error(observed_general_sources[name], target),
        )
        for name, target in GENERAL_SOURCE_TARGETS
    )
    return DraftDiagnosticReceipt(
        gate="D3",
        status="DESCRIPTIVE_ONLY",
        named_input_sha256s=(("composition_input", composition_input_sha256),),
        evidence=(
            ("general_source_rows", general_rows),
            ("observed_total_bytes", sum(observed_strata.values())),
            ("stratum_rows", stratum_rows),
            ("target_total_bytes", CORPUS_TOTAL_TARGET_BYTES),
            ("tolerance_semantics", "UNBOUND"),
        ),
    )


def diagnose_d4_language_filter_scope(
    invocation_counts: Mapping[str, int],
    rejection_counts: Mapping[str, int],
    *,
    language_filter_input_sha256: str,
) -> DraftDiagnosticReceipt:
    _require_sha256(language_filter_input_sha256, "language filter input sha256")
    if set(invocation_counts) != set(GTOK_STRATA) or set(rejection_counts) != set(GTOK_STRATA):
        raise ValueError("D4 requires invocation and rejection counts for every stratum")
    for mapping in (invocation_counts, rejection_counts):
        if any(type(value) is not int or value < 0 for value in mapping.values()):
            raise ValueError("D4 counts must be non-negative exact integers")
    if any(rejection_counts[name] > invocation_counts[name] for name in GTOK_STRATA):
        raise ValueError("D4 failed: rejection count exceeds classifier invocation count")
    for stratum in GTOK_STRATA[1:]:
        if invocation_counts[stratum] != 0 or rejection_counts[stratum] != 0:
            raise ValueError("D4 failed: language-ID was called outside the general stratum")
    return DraftDiagnosticReceipt(
        gate="D4",
        status="DRAFT_CONSISTENT",
        named_input_sha256s=(("language_filter_input", language_filter_input_sha256),),
        evidence=(
            ("invocation_counts", tuple(sorted(invocation_counts.items()))),
            ("rejection_counts", tuple(sorted(rejection_counts.items()))),
        ),
    )


@dataclass(frozen=True)
class RoundTripFixture:
    fixture_id: str
    category: str
    payload: bytes

    def __post_init__(self) -> None:
        _require_run_id(self.fixture_id)
        if self.category not in GTOK_ROUND_TRIP_CATEGORIES:
            raise ValueError("unknown registered round-trip category")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("round-trip fixture must contain at least one byte")

    @property
    def identity_sha256(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_round_trip_fixture_v1",
            {
                "category": self.category,
                "fixture_id": self.fixture_id,
                "original_byte_count": len(self.payload),
                "original_sha256": sha256_bytes(self.payload),
            },
        )


@dataclass(frozen=True)
class RoundTripFixtureManifest:
    fixtures: tuple[RoundTripFixture, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixtures, tuple):
            raise TypeError("round-trip fixtures must be a tuple")
        if any(not isinstance(item, RoundTripFixture) for item in self.fixtures):
            raise TypeError("round-trip manifest contains a non-fixture value")
        fixture_ids = tuple(item.fixture_id for item in self.fixtures)
        if fixture_ids != tuple(sorted(fixture_ids)) or len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("round-trip fixture IDs must be unique and sorted")
        categories = tuple(item.category for item in self.fixtures)
        if tuple(sorted(categories)) != tuple(sorted(GTOK_ROUND_TRIP_CATEGORIES)):
            raise ValueError("round-trip manifest requires exactly one registered category")

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256(
            "weft1_draft_round_trip_fixture_manifest_v1",
            tuple(item.identity_sha256 for item in self.fixtures),
        )


@dataclass(frozen=True)
class RoundTripCaseDiagnostic:
    fixture_identity_sha256: str
    fixture_id: str
    category: str
    original_byte_count: int
    restored_byte_count: int
    original_sha256: str
    restored_sha256: str

    @classmethod
    def exercise(
        cls,
        fixture: RoundTripFixture,
        round_trip: Callable[[bytes], bytes],
    ) -> "RoundTripCaseDiagnostic":
        if not isinstance(fixture, RoundTripFixture):
            raise TypeError("fixture must be a RoundTripFixture")
        if not callable(round_trip):
            raise TypeError("round_trip must be callable")
        restored = round_trip(fixture.payload)
        if not isinstance(restored, bytes):
            raise TypeError("round-trip callback must return bytes")
        return cls(
            fixture_identity_sha256=fixture.identity_sha256,
            fixture_id=fixture.fixture_id,
            category=fixture.category,
            original_byte_count=len(fixture.payload),
            restored_byte_count=len(restored),
            original_sha256=sha256_bytes(fixture.payload),
            restored_sha256=sha256_bytes(restored),
        )

    def __post_init__(self) -> None:
        _require_sha256(self.fixture_identity_sha256, "fixture identity sha256")
        _require_run_id(self.fixture_id)
        if self.category not in GTOK_ROUND_TRIP_CATEGORIES:
            raise ValueError("unknown registered round-trip category")
        for name in ("original_byte_count", "restored_byte_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("round-trip byte counts must be positive exact integers")
        _require_sha256(self.original_sha256, "original_sha256")
        _require_sha256(self.restored_sha256, "restored_sha256")

    @property
    def byte_exact(self) -> bool:
        return (
            self.original_byte_count == self.restored_byte_count
            and self.original_sha256 == self.restored_sha256
        )


def diagnose_d5_round_trip_suite(
    cases: tuple[RoundTripCaseDiagnostic, ...],
    *,
    codec_spec: AlgorithmSpec,
    fixture_manifest: RoundTripFixtureManifest,
) -> DraftDiagnosticReceipt:
    if not isinstance(codec_spec, AlgorithmSpec):
        raise TypeError("codec_spec must be an AlgorithmSpec")
    if not isinstance(fixture_manifest, RoundTripFixtureManifest):
        raise TypeError("fixture_manifest must be a RoundTripFixtureManifest")
    if not isinstance(cases, tuple):
        raise TypeError("round-trip cases must be a tuple")
    ordered = tuple(sorted(cases, key=lambda item: item.fixture_id))
    expected_identities = tuple(item.identity_sha256 for item in fixture_manifest.fixtures)
    observed_identities = tuple(item.fixture_identity_sha256 for item in ordered)
    if observed_identities != expected_identities:
        raise ValueError("D5 cases do not exactly match the typed fixture manifest")
    for case, fixture in zip(ordered, fixture_manifest.fixtures, strict=True):
        if (
            case.fixture_id != fixture.fixture_id
            or case.category != fixture.category
            or case.original_byte_count != len(fixture.payload)
            or case.original_sha256 != sha256_bytes(fixture.payload)
        ):
            raise ValueError("D5 case metadata disagrees with its typed fixture")
    if any(not item.byte_exact for item in cases):
        raise ValueError("D5 failed: reference round trip changed bytes")
    return DraftDiagnosticReceipt(
        gate="D5",
        status="DRAFT_CONSISTENT",
        named_input_sha256s=(
            ("codec_spec_draft", codec_spec.draft_sha256),
            ("fixture_manifest_draft", fixture_manifest.draft_sha256),
        ),
        evidence=(
            ("case_count", len(ordered)),
            ("cases", ordered),
        ),
    )


def reference_frame_payload(payload: bytes) -> bytes:
    """Unambiguous fixture framing; the production serialization remains unbound."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) >= 1 << 64:
        raise ValueError("payload is too large for reference uint64 framing")
    return len(payload).to_bytes(8, "big") + payload


@dataclass(frozen=True)
class SplitEntryDiagnostic:
    doc_id: str
    cluster_id: str
    stratum: str
    retained_byte_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.doc_id, "split doc_id")
        _require_sha256(self.cluster_id, "split cluster_id")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("split entry uses an unknown stratum")
        if type(self.retained_byte_count) is not int or self.retained_byte_count < 1:
            raise ValueError("split retained_byte_count must be positive")


@dataclass(frozen=True)
class SplitManifestDiagnostic:
    split_spec_draft_sha256: str
    cluster_spec_draft_sha256: str
    training: tuple[SplitEntryDiagnostic, ...]
    heldout: tuple[SplitEntryDiagnostic, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.split_spec_draft_sha256, "split spec draft sha256")
        _require_sha256(self.cluster_spec_draft_sha256, "cluster spec draft sha256")
        for entries, name in ((self.training, "training"), (self.heldout, "heldout")):
            if not isinstance(entries, tuple) or not entries:
                raise ValueError(f"split manifest {name} entries must be nonempty")
            if any(not isinstance(item, SplitEntryDiagnostic) for item in entries):
                raise TypeError(f"split manifest {name} contains a non-entry")
            ids = tuple(item.doc_id for item in entries)
            if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
                raise ValueError(f"split manifest {name} IDs must be unique and sorted")
        train_docs = {item.doc_id for item in self.training}
        heldout_docs = {item.doc_id for item in self.heldout}
        if train_docs & heldout_docs:
            raise ValueError("split manifest crosses a document between train and held-out")
        train_clusters = {item.cluster_id for item in self.training}
        heldout_clusters = {item.cluster_id for item in self.heldout}
        if train_clusters & heldout_clusters:
            raise ValueError("split manifest crosses a dedup cluster between train and held-out")

    @staticmethod
    def _document_set_sha256(entries: tuple[SplitEntryDiagnostic, ...]) -> str:
        return sha256_bytes(canonical_json_bytes(tuple(item.doc_id for item in entries)))

    @property
    def training_document_set_sha256(self) -> str:
        return self._document_set_sha256(self.training)

    @property
    def heldout_document_set_sha256(self) -> str:
        return self._document_set_sha256(self.heldout)

    @property
    def draft_sha256(self) -> str:
        return execution_authority_bound_sha256("weft1_draft_split_manifest_v1", self)


@dataclass(frozen=True)
class StreamDiagnostic:
    permutation_seed: int | None
    codec_spec_draft_sha256: str
    ordered_document_ids_sha256: str
    document_set_sha256: str
    framed_payload_stream_sha256: str
    retained_byte_count: int
    document_count: int
    stratum_byte_counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[DocumentRecord],
        *,
        permutation_seed: int | None,
        codec_spec: AlgorithmSpec,
    ) -> "StreamDiagnostic":
        if permutation_seed is not None and type(permutation_seed) is not int:
            raise TypeError("permutation_seed must be an exact integer or None")
        if not isinstance(codec_spec, AlgorithmSpec):
            raise TypeError("codec_spec must be an AlgorithmSpec")
        if not documents:
            raise ValueError("stream diagnostic requires at least one document")
        ordered_ids = tuple(item.doc_id for item in documents)
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("stream diagnostic contains a repeated document")
        stream_hash = hashlib.sha256()
        stratum_bytes = {name: 0 for name in GTOK_STRATA}
        for item in documents:
            stream_hash.update(reference_frame_payload(item.retained_bytes))
            stratum_bytes[item.stratum] += len(item.retained_bytes)
        return cls(
            permutation_seed=permutation_seed,
            codec_spec_draft_sha256=codec_spec.draft_sha256,
            ordered_document_ids_sha256=sha256_bytes(canonical_json_bytes(ordered_ids)),
            document_set_sha256=sha256_bytes(canonical_json_bytes(tuple(sorted(ordered_ids)))),
            framed_payload_stream_sha256=stream_hash.hexdigest(),
            retained_byte_count=sum(len(item.retained_bytes) for item in documents),
            document_count=len(documents),
            stratum_byte_counts=tuple(sorted(stratum_bytes.items())),
        )

    def __post_init__(self) -> None:
        if self.permutation_seed is not None and type(self.permutation_seed) is not int:
            raise TypeError("permutation_seed must be an exact integer or None")
        for name in (
            "codec_spec_draft_sha256",
            "ordered_document_ids_sha256",
            "document_set_sha256",
            "framed_payload_stream_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("retained_byte_count", "document_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if tuple(name for name, unused in self.stratum_byte_counts) != tuple(sorted(GTOK_STRATA)):
            raise ValueError("stream diagnostic must report every stratum in canonical order")
        if any(type(value) is not int or value < 0 for unused, value in self.stratum_byte_counts):
            raise ValueError("stream stratum byte counts must be non-negative integers")
        if sum(value for unused, value in self.stratum_byte_counts) != self.retained_byte_count:
            raise ValueError("stream stratum bytes do not reconcile to retained bytes")


@dataclass(frozen=True)
class RunStreamDiagnostic:
    vocabulary_size: int
    seed: int
    training: StreamDiagnostic
    heldout: StreamDiagnostic

    def __post_init__(self) -> None:
        if (
            type(self.vocabulary_size) is not int
            or self.vocabulary_size not in GTOK_VOCABULARY_ARMS
        ):
            raise ValueError("run stream uses an unregistered vocabulary arm")
        if type(self.seed) is not int:
            raise TypeError("run stream seed must be an exact integer")
        if not isinstance(self.training, StreamDiagnostic) or not isinstance(
            self.heldout, StreamDiagnostic
        ):
            raise TypeError("run stream requires typed training and held-out diagnostics")
        if self.training.permutation_seed != self.seed:
            raise ValueError("training stream seed disagrees with run seed")
        if self.heldout.permutation_seed is not None:
            raise ValueError("the fixed held-out stream must not vary by run seed")


def diagnose_d6_streams(
    runs: tuple[RunStreamDiagnostic, ...],
    split_manifest: SplitManifestDiagnostic,
) -> DraftDiagnosticReceipt:
    """Check paired stream topology without choosing the unresolved seed/byte rules."""

    if not isinstance(runs, tuple) or not runs:
        raise ValueError("D6 requires typed run-stream diagnostics")
    if not isinstance(split_manifest, SplitManifestDiagnostic):
        raise TypeError("D6 requires a typed split manifest diagnostic")
    if any(not isinstance(item, RunStreamDiagnostic) for item in runs):
        raise TypeError("D6 run matrix contains a non-run value")
    keys = {(item.vocabulary_size, item.seed) for item in runs}
    seeds = tuple(sorted({item.seed for item in runs}))
    expected_keys = {
        (vocabulary_size, seed)
        for vocabulary_size in GTOK_VOCABULARY_ARMS
        for seed in seeds
    }
    if len(runs) != len(keys) or len(seeds) != GTOK_SEED_COUNT or keys != expected_keys:
        raise ValueError("D6 requires the complete four-arm, two-seed matrix")
    if any(
        item.training.document_set_sha256 != split_manifest.training_document_set_sha256
        or item.heldout.document_set_sha256 != split_manifest.heldout_document_set_sha256
        or item.training.document_count != len(split_manifest.training)
        or item.heldout.document_count != len(split_manifest.heldout)
        for item in runs
    ):
        raise ValueError("D6 failed: stream receipts disagree with the split manifest")
    for seed in seeds:
        within_seed = tuple(item.training for item in runs if item.seed == seed)
        if len({item.ordered_document_ids_sha256 for item in within_seed}) != 1:
            raise ValueError("D6 failed: vocabulary arms use different order within a seed")
        if len({item.framed_payload_stream_sha256 for item in within_seed}) != 1:
            raise ValueError("D6 failed: vocabulary arms use different payload streams")
        if len({item.codec_spec_draft_sha256 for item in within_seed}) != 1:
            raise ValueError("D6 failed: vocabulary arms use different stream codecs")
    order_by_seed = {
        seed: next(item.training.ordered_document_ids_sha256 for item in runs if item.seed == seed)
        for seed in seeds
    }
    if len(set(order_by_seed.values())) != len(seeds):
        raise ValueError("D6 failed: independent seeds require distinct training orders")
    heldout_identities = {
        (
            item.heldout.codec_spec_draft_sha256,
            item.heldout.ordered_document_ids_sha256,
            item.heldout.framed_payload_stream_sha256,
            item.heldout.retained_byte_count,
            item.heldout.stratum_byte_counts,
        )
        for item in runs
    }
    if len(heldout_identities) != 1:
        raise ValueError("D6 failed: held-out stream differs across runs")
    training_codec_hashes = {item.training.codec_spec_draft_sha256 for item in runs}
    heldout_codec_hashes = {item.heldout.codec_spec_draft_sha256 for item in runs}
    if len(training_codec_hashes) != 1 or training_codec_hashes != heldout_codec_hashes:
        raise ValueError("D6 failed: one stream codec must cover train and held-out")
    training_bytes = {item.training.retained_byte_count for item in runs}
    heldout_bytes = {item.heldout.retained_byte_count for item in runs}
    expected_training_bytes = sum(item.retained_byte_count for item in split_manifest.training)
    expected_heldout_bytes = sum(item.retained_byte_count for item in split_manifest.heldout)
    expected_training_strata = tuple(
        sorted(
            (
                stratum,
                sum(
                    item.retained_byte_count
                    for item in split_manifest.training
                    if item.stratum == stratum
                ),
            )
            for stratum in GTOK_STRATA
        )
    )
    expected_heldout_strata = tuple(
        sorted(
            (
                stratum,
                sum(
                    item.retained_byte_count
                    for item in split_manifest.heldout
                    if item.stratum == stratum
                ),
            )
            for stratum in GTOK_STRATA
        )
    )
    if training_bytes != {expected_training_bytes} or heldout_bytes != {expected_heldout_bytes}:
        raise ValueError("D6 failed: stream byte counts disagree with the split manifest")
    if any(
        item.training.stratum_byte_counts != expected_training_strata
        or item.heldout.stratum_byte_counts != expected_heldout_strata
        for item in runs
    ):
        raise ValueError("D6 failed: stream strata disagree with the split manifest")
    return DraftDiagnosticReceipt(
        gate="D6",
        status="DESCRIPTIVE_ONLY",
        named_input_sha256s=(("split_manifest_draft", split_manifest.draft_sha256),),
        evidence=(
            ("arm_seed_count", len(runs)),
            ("heldout_retained_bytes", next(iter(heldout_bytes))),
            ("heldout_semantics", "UNBOUND"),
            ("ordered_stream_sha256_by_seed", tuple(sorted(order_by_seed.items()))),
            ("registered_screen_target_bytes", GTOK_SCREEN_TARGET_BYTES),
            ("seed_identities", seeds),
            ("seed_identities_status", "UNBOUND"),
            ("training_retained_bytes", next(iter(training_bytes))),
        ),
    )


def mint_authoritative_gate_receipt(
    gate: str, diagnostic: DraftDiagnosticReceipt
) -> NoReturn:
    """Fail closed until strategy supplies the literal v2 execution bindings."""

    if gate not in {"D1", "D2", "D3", "D4", "D5", "D6"}:
        raise ValueError("unknown corpus/G-TOK gate")
    if not isinstance(diagnostic, DraftDiagnosticReceipt) or diagnostic.gate != gate:
        raise TypeError("authoritative mint requires a matching draft diagnostic")
    require_gtok_execution_authority(f"mint authoritative {gate} corpus gate receipt")


__all__ = [
    "AlgorithmSpec",
    "BuildRunDiagnostic",
    "BYTE_SHINGLE_WIDTH",
    "CORPUS_STRATUM_TARGETS",
    "CORPUS_TOTAL_TARGET_BYTES",
    "DedupDecisionDiagnostic",
    "DedupDiagnosticLedger",
    "DocumentRecord",
    "DraftDiagnosticReceipt",
    "GENERAL_SOURCE_TARGETS",
    "GTOK_EXECUTION_AUTHORITY_CHAIN",
    "GTOK_SCREEN_TARGET_BYTES",
    "NEAR_DUPLICATE_JACCARD_THRESHOLD",
    "NormalizedDocumentDiagnostic",
    "ReferenceDedupBinding",
    "RoundTripCaseDiagnostic",
    "RoundTripFixture",
    "RoundTripFixtureManifest",
    "RunStreamDiagnostic",
    "ShardDiagnostic",
    "SourceAsset",
    "SplitEntryDiagnostic",
    "SplitManifestDiagnostic",
    "StreamDiagnostic",
    "build_reference_cross_source_dedup_diagnostic",
    "byte_shingles",
    "describe_d3_composition",
    "diagnose_d1_reproduction",
    "diagnose_d2_dedup_reproduction",
    "diagnose_d4_language_filter_scope",
    "diagnose_d5_round_trip_suite",
    "diagnose_d6_streams",
    "exact_set_jaccard",
    "mint_authoritative_gate_receipt",
    "reference_frame_payload",
    "reference_minhash_signature",
    "sha256_bytes",
]
