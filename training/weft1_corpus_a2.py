"""Pure Amendment-A2 contracts for deterministic WEFT-1 corpus P-A.

This module appends a V3 authority domain to the banked V2 chain.  It contains
only typed values and deterministic pure functions: no network access, source
downloads, filesystem reads or writes, compression, subprocess launch, model
execution, or gate-side effects occur here.

The production I/O layer may consume these contracts, but it must supply the
independent-process, source-asset, shard-byte, and tripwire evidence that the
types below validate.  Run-local metadata never participates in a corpus
content identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Sequence
import unicodedata

import numpy as np

from training.weft1_gtok_a1_contract import SOURCE_FAMILIES
from training.weft1_gtok_contract import (
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_STRATA,
    canonical_json_bytes,
    canonical_sha256,
)
from training.weft1_seed import derive_module_seed


GTOK_AMENDMENT_A2_SHA256 = (
    "f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02"
)
GTOK_EXECUTION_AUTHORITY_CHAIN_V3 = (
    *GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_AMENDMENT_A2_SHA256,
)
A2_CAMPAIGN_ROOT_SEED = int(GTOK_AMENDMENT_A2_SHA256[:16], 16)
A2_PIPELINE_SEED_NAMES = (
    "corpus.dedup",
    "corpus.shuffle",
    "corpus.split",
    "corpus.topup",
    "gtok.bpe",
)
A2_PIPELINE_SEEDS = tuple(
    (name, derive_module_seed(A2_CAMPAIGN_ROOT_SEED, name))
    for name in A2_PIPELINE_SEED_NAMES
)
A2_DEDUP_SEED = dict(A2_PIPELINE_SEEDS)["corpus.dedup"]

PYTHON_UNICODE_DATA_VERSION = "14.0.0"
NUMPY_VERSION = "2.4.6"
MINHASH_COMPONENTS = 128
MINHASH_BANDS = 16
MINHASH_ROWS_PER_BAND = 8
MINHASH_SHINGLE_WIDTH = 13
MINHASH_MODULUS = 1 << 64
NEAR_DUPLICATE_THRESHOLD = Fraction(4, 5)
MINHASH_RECALL_JACCARD_LEVELS = (
    Fraction(3, 4),
    Fraction(79, 100),
    Fraction(4, 5),
    Fraction(81, 100),
    Fraction(17, 20),
    Fraction(9, 10),
)
FIRST_FIT_TOLERANCE = Fraction(5, 1_000)
A2_TRIPWIRE_A100_SECONDS = 12 * 60 * 60
A2_CALIBRATION_WARMUP_STEPS = 20
A2_CALIBRATION_MEASURED_STEPS = 80
A2_CALIBRATION_STEPS_MAXIMUM = 100
A2_CHARGED_ATTEMPT_STATUSES = (
    "calibration",
    "completed",
    "failed",
    "aborted",
    "preempted",
    "retried",
)
A2_STREAM_PRECEDENCE = ("T", "H")
EMPTY_MATCH_NORMALIZATION_DISPOSITION = "DROP_EMPTY_AFTER_MATCH_NORMALIZATION"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_WINDOWS_DEVICES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


def _require_run_id(value: str) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must use canonical lowercase ASCII syntax")
    return value


def _require_uint64(value: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < MINHASH_MODULUS:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _canonical_relative_path(value: str) -> str:
    _require_nonempty(value, "relative path")
    if "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("artifact paths must be canonical relative POSIX paths")
    parts = value.split("/")
    if any(
        _PATH_SEGMENT.fullmatch(part) is None
        or part.endswith(".")
        or part.rstrip(". ").split(".", 1)[0].casefold() in _WINDOWS_DEVICES
        for part in parts
    ):
        raise ValueError("artifact path contains a noncanonical segment")
    return value


def execution_authority_v3_bound_sha256(schema: str, value: object) -> str:
    """Hash a new A2 receipt without re-keying any V1/V2 receipt."""

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("receipt schema must be a nonempty exact string")
    if not schema.endswith("_v3"):
        raise ValueError("A2 execution receipts require an explicit v3 schema")
    return canonical_sha256(
        {
            "authority_chain": GTOK_EXECUTION_AUTHORITY_CHAIN_V3,
            "payload": value,
            "schema": schema,
        }
    )


def pipeline_seed(name: str) -> int:
    """Return one A2 seed through the existing namespaced RNG derivation."""

    if name not in A2_PIPELINE_SEED_NAMES:
        raise ValueError("unknown A2 pipeline RNG name")
    return derive_module_seed(A2_CAMPAIGN_ROOT_SEED, name)


@dataclass(frozen=True)
class StableDocumentV3:
    """One logical document, independent of an upsampled asset occurrence."""

    source: str
    stratum: str
    stable_source_record_id: str
    text: str

    def __post_init__(self) -> None:
        if self.source not in SOURCE_FAMILIES:
            raise ValueError("document uses an unknown source family")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("document uses an unknown corpus stratum")
        _require_sha256(self.stable_source_record_id, "stable_source_record_id")
        if not isinstance(self.text, str):
            raise TypeError("document text must be an exact string")
        try:
            retained = self.text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("document text must be scalar-valid UTF-8") from error
        if not retained:
            raise ValueError("document text must retain at least one UTF-8 byte")

    @property
    def retained_bytes(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def retained_sha256(self) -> str:
        return hashlib.sha256(self.retained_bytes).hexdigest()

    @property
    def retained_sha1(self) -> str:
        return hashlib.sha1(self.retained_bytes).hexdigest()  # noqa: S324 - contract

    @property
    def retained_byte_count(self) -> int:
        return len(self.retained_bytes)

    @property
    def document_id(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_stable_document_v3",
            {
                "retained_sha256": self.retained_sha256,
                "source": self.source,
                "stable_source_record_id": self.stable_source_record_id,
                "stratum": self.stratum,
            },
        )

    @property
    def shard_record_id(self) -> str:
        """Return A2's lowercase SHA-1 identity of the raw retained UTF-8 bytes."""

        return self.retained_sha1


@dataclass(frozen=True)
class DocumentOccurrenceV3:
    """Provenance for one physical occurrence of a stable document."""

    document: StableDocumentV3
    source_asset_sha256: str
    source_record_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, StableDocumentV3):
            raise TypeError("occurrence requires a StableDocumentV3")
        _require_sha256(self.source_asset_sha256, "source_asset_sha256")
        if type(self.source_record_ordinal) is not int or self.source_record_ordinal < 0:
            raise ValueError("source_record_ordinal must be a non-negative integer")

    @property
    def occurrence_id(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_document_occurrence_v3",
            {
                "document_id": self.document.document_id,
                "source_asset_sha256": self.source_asset_sha256,
                "source_record_ordinal": self.source_record_ordinal,
            },
        )


@dataclass(frozen=True)
class MatchNormalizationBindingV3:
    transform_order: tuple[str, ...] = (
        "unicode_nfc",
        "collapse_maximal_python_str_isspace_runs_to_ascii_space",
        "strip_collapsed_ascii_space",
        "utf8_encode",
    )
    unicode_data_version: str = PYTHON_UNICODE_DATA_VERSION
    retained_text_is_unchanged: bool = True

    def __post_init__(self) -> None:
        if self.transform_order != (
            "unicode_nfc",
            "collapse_maximal_python_str_isspace_runs_to_ascii_space",
            "strip_collapsed_ascii_space",
            "utf8_encode",
        ):
            raise ValueError("A2 match-normalization order drifted")
        if self.unicode_data_version != PYTHON_UNICODE_DATA_VERSION:
            raise ValueError("A2 Unicode data version drifted")
        if self.retained_text_is_unchanged is not True:
            raise ValueError("match normalization may not mutate retained text")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_match_normalization_binding_v3", self
        )


A2_MATCH_NORMALIZATION_BINDING = MatchNormalizationBindingV3()


def normalize_match_text(text: str) -> str:
    """Apply NFC, maximal ``str.isspace`` collapse, and boundary stripping."""

    if not isinstance(text, str):
        raise TypeError("match normalization requires an exact string")
    if unicodedata.unidata_version != PYTHON_UNICODE_DATA_VERSION:
        raise RuntimeError("runtime Unicode data version differs from A2")
    normalized = unicodedata.normalize("NFC", text)
    output: list[str] = []
    whitespace_pending = False
    for character in normalized:
        if character.isspace():
            whitespace_pending = bool(output)
            continue
        if whitespace_pending:
            output.append(" ")
            whitespace_pending = False
        output.append(character)
    return "".join(output)


def normalized_match_bytes(text: str) -> bytes:
    return normalize_match_text(text).encode("utf-8")


def dedup_match_input_v3(document: StableDocumentV3) -> bytes | None:
    """Return match bytes, or ``None`` for the bound whitespace-only drop."""

    if not isinstance(document, StableDocumentV3):
        raise TypeError("dedup match input requires a StableDocumentV3")
    normalized = normalized_match_bytes(document.text)
    return normalized or None


def is_exact_duplicate_v3(
    left: StableDocumentV3, right: StableDocumentV3
) -> bool:
    """Confirm an SHA-1 candidate by length, SHA-256, then byte equality."""

    if not isinstance(left, StableDocumentV3) or not isinstance(
        right, StableDocumentV3
    ):
        raise TypeError("exact duplicate confirmation requires stable documents")
    if left.retained_sha1 != right.retained_sha1:
        return False
    return (
        left.retained_byte_count == right.retained_byte_count
        and left.retained_sha256 == right.retained_sha256
        and left.retained_bytes == right.retained_bytes
    )


@dataclass(frozen=True)
class LanguageIdBindingV3:
    package: str = "fasttext-wheel"
    package_version: str = "0.9.2"
    adapter: str = "fasttext._FastText.f.predict"
    model_url: str = (
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    )
    model_bytes: int = 131_266_198
    model_sha256: str = (
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    )
    scoring_prefix_bytes: int = 65_536
    partial_codepoint_rule: str = (
        "drop_only_an_incomplete_trailing_utf8_codepoint_from_scoring_prefix"
    )
    single_line_rule: str = "replace_cr_and_lf_with_ascii_space_for_scoring_only"
    keep_label: str = "__label__en"
    keep_probability: Fraction = Fraction(9, 10)
    equality_keeps: bool = True
    equal_probability_tie_rule: str = "lowest_ASCII_label"
    scope: str = "general_only"

    def __post_init__(self) -> None:
        if (
            self.package != "fasttext-wheel"
            or self.package_version != "0.9.2"
            or self.adapter != "fasttext._FastText.f.predict"
            or self.model_bytes != 131_266_198
            or self.scoring_prefix_bytes != 65_536
            or self.keep_label != "__label__en"
            or self.keep_probability != Fraction(9, 10)
            or self.equality_keeps is not True
            or self.equal_probability_tie_rule != "lowest_ASCII_label"
            or self.scope != "general_only"
        ):
            raise ValueError("A2 language-ID binding drifted")
        _require_sha256(self.model_sha256, "language-ID model SHA-256")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_language_id_binding_v3", self
        )


A2_LANGUAGE_ID_BINDING = LanguageIdBindingV3()


def language_scoring_bytes_v3(document: StableDocumentV3) -> bytes:
    """Build the exact 64 KiB, UTF-8-safe, single-line FastText input."""

    if not isinstance(document, StableDocumentV3):
        raise TypeError("language scoring requires a StableDocumentV3")
    if document.stratum != "general":
        raise ValueError("A2 language ID may be invoked only for general")
    raw = document.retained_bytes
    prefix = raw[: A2_LANGUAGE_ID_BINDING.scoring_prefix_bytes]
    try:
        decoded = prefix.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.end != len(prefix) or error.reason != "unexpected end of data":
            raise ValueError("language scoring prefix has invalid interior UTF-8") from error
        decoded = prefix[: error.start].decode("utf-8", errors="strict")
    return decoded.replace("\r", " ").replace("\n", " ").encode("utf-8")


def language_backend_input_bytes_v3(document: StableDocumentV3) -> bytes:
    """Return the exact byte string passed to the registered FastText backend.

    ``fasttext._FastText.f.predict`` consumes a newline-terminated line.  Keep
    that adapter terminator inside the hashed input contract so a receipt can
    never describe different bytes from those actually classified.
    """

    return language_scoring_bytes_v3(document) + b"\n"


@dataclass(frozen=True)
class LanguageIdDecisionV3:
    """One general-only classifier decision; non-general calls are unrepresentable."""

    document_id: str
    stratum: str
    scoring_input_sha256: str
    label: str
    probability: float
    binding_sha256: str = A2_LANGUAGE_ID_BINDING.receipt_sha256

    def __post_init__(self) -> None:
        _require_sha256(self.document_id, "language document_id")
        if self.stratum != "general":
            raise ValueError("A2 language ID may be invoked only for general")
        _require_sha256(self.scoring_input_sha256, "language scoring input SHA-256")
        _require_nonempty(self.label, "language label")
        if isinstance(self.probability, bool) or not isinstance(
            self.probability, (int, float)
        ):
            raise TypeError("language probability must be numeric")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("language probability must lie in [0, 1]")
        object.__setattr__(self, "probability", probability)
        if self.binding_sha256 != A2_LANGUAGE_ID_BINDING.receipt_sha256:
            raise ValueError("language decision uses the wrong A2 binding")

    @property
    def keep(self) -> bool:
        return (
            self.label == A2_LANGUAGE_ID_BINDING.keep_label
            and self.probability >= float(A2_LANGUAGE_ID_BINDING.keep_probability)
        )

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_language_id_decision_v3", self
        )


def language_id_decision_v3(
    document: StableDocumentV3, *, label: str, probability: float
) -> LanguageIdDecisionV3:
    """Bind caller-supplied FastText output to the exact backend input."""

    scoring = language_backend_input_bytes_v3(document)
    return LanguageIdDecisionV3(
        document_id=document.document_id,
        stratum=document.stratum,
        scoring_input_sha256=hashlib.sha256(scoring).hexdigest(),
        label=label,
        probability=probability,
    )


def language_id_decision_from_predictions_v3(
    document: StableDocumentV3,
    predictions: Sequence[tuple[str, float]],
) -> LanguageIdDecisionV3:
    """Select the highest probability, breaking an exact tie by ASCII label."""

    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise TypeError("language predictions must be a typed sequence")
    if not predictions:
        raise ValueError("language classifier returned no predictions")
    validated: list[tuple[str, float]] = []
    for prediction in predictions:
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise TypeError("language prediction rows must be (label, probability)")
        label, probability = prediction
        _require_nonempty(label, "language label")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("language probability must be numeric")
        numeric = float(probability)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("language probability must lie in [0, 1]")
        try:
            label.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("language labels must be ASCII") from error
        validated.append((label, numeric))
    label, probability = min(validated, key=lambda row: (-row[1], row[0]))
    return language_id_decision_v3(
        document,
        label=label,
        probability=probability,
    )


def byte_shingles_v3(
    value: bytes, width: int = MINHASH_SHINGLE_WIDTH
) -> frozenset[bytes]:
    """Return overlapping byte shingles; short inputs form exactly one shingle."""

    if not isinstance(value, bytes):
        raise TypeError("byte shingles require exact bytes")
    if type(width) is not int or width != MINHASH_SHINGLE_WIDTH:
        raise ValueError("A2 binds byte-level 13-gram shingles")
    if not value:
        raise ValueError("A2 dedup requires nonempty normalized bytes")
    if len(value) <= width:
        return frozenset((value,))
    return frozenset(
        value[index : index + width] for index in range(len(value) - width + 1)
    )


def exact_jaccard_v3(
    left: frozenset[bytes], right: frozenset[bytes]
) -> Fraction:
    if not isinstance(left, frozenset) or not isinstance(right, frozenset):
        raise TypeError("exact Jaccard requires two frozensets")
    if any(not isinstance(value, bytes) for value in (*left, *right)):
        raise TypeError("exact Jaccard sets must contain only bytes")
    union = left | right
    if not union:
        return Fraction(1, 1)
    return Fraction(len(left & right), len(union))


@dataclass(frozen=True)
class MinHashBindingV3:
    numpy_version: str = NUMPY_VERSION
    bit_generator: str = "numpy.random.PCG64"
    component_count: int = MINHASH_COMPONENTS
    shingle_width_bytes: int = MINHASH_SHINGLE_WIDTH
    shingle_digest: str = "uint64le_first_8_bytes_sha256_u64be_length_then_shingle"
    permutation: str = "odd_a_times_x_plus_b_mod_2_pow_64"
    coefficient_draw: str = "pcg64_random_raw_256_split_a_then_b"
    seed: int = A2_DEDUP_SEED
    bands: int = MINHASH_BANDS
    rows_per_band: int = MINHASH_ROWS_PER_BAND
    band_uint64_endianness: str = "little"
    exact_jaccard_threshold: Fraction = NEAR_DUPLICATE_THRESHOLD
    short_document_rule: str = "one_complete_nonempty_normalized_byte_shingle"
    winner_rule: str = "maximum_exact_jaccard_then_lowest_dolma_source_record_id"

    def __post_init__(self) -> None:
        if self.numpy_version != NUMPY_VERSION or self.bit_generator != "numpy.random.PCG64":
            raise ValueError("A2 NumPy/PCG64 binding drifted")
        if (
            self.component_count != 128
            or self.shingle_width_bytes != 13
            or self.bands != 16
            or self.rows_per_band != 8
            or self.bands * self.rows_per_band != self.component_count
        ):
            raise ValueError("A2 MinHash/LSH dimensions drifted")
        _require_uint64(self.seed, "MinHash seed")
        if self.band_uint64_endianness != "little":
            raise ValueError("A2 band keys require little-endian uint64 values")
        if self.exact_jaccard_threshold != Fraction(4, 5):
            raise ValueError("A2 exact Jaccard threshold drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_minhash_binding_v3", self
        )


A2_MINHASH_BINDING = MinHashBindingV3()


def ideal_lsh_candidate_probability_v3(jaccard: Fraction) -> Fraction:
    """Return ``1 - (1 - s**8)**16`` exactly for the registered LSH."""

    if not isinstance(jaccard, Fraction) or not Fraction(0, 1) <= jaccard <= 1:
        raise ValueError("LSH Jaccard must be a Fraction in [0, 1]")
    return 1 - (1 - jaccard**MINHASH_ROWS_PER_BAND) ** MINHASH_BANDS


@dataclass(frozen=True)
class MinHashSyntheticRecallCellV3:
    exact_jaccard: Fraction
    pair_count: int
    candidate_count: int

    def __post_init__(self) -> None:
        if self.exact_jaccard not in MINHASH_RECALL_JACCARD_LEVELS:
            raise ValueError("synthetic recall cell uses an unregistered Jaccard level")
        if type(self.pair_count) is not int or self.pair_count < 1:
            raise ValueError("synthetic recall pair_count must be positive")
        if (
            type(self.candidate_count) is not int
            or not 0 <= self.candidate_count <= self.pair_count
        ):
            raise ValueError("synthetic recall candidate_count is invalid")

    @property
    def ideal_candidate_probability(self) -> Fraction:
        return ideal_lsh_candidate_probability_v3(self.exact_jaccard)


@dataclass(frozen=True)
class MinHashRecallAuditV3:
    """Report-only qualification; it intentionally imposes no recall floor."""

    seed: int
    synthetic_cells: tuple[MinHashSyntheticRecallCellV3, ...]
    real_sample_identity_sha256: str
    real_dolma_document_count: int
    real_fineweb_document_count: int
    real_exact_pairs_at_or_above_threshold: int
    real_candidate_pairs_at_or_above_threshold: int
    status: str = "REPORT_ONLY_NO_RECALL_FLOOR"

    def __post_init__(self) -> None:
        _require_uint64(self.seed, "MinHash recall-audit seed")
        if self.seed != A2_DEDUP_SEED:
            raise ValueError("MinHash recall audit must use the registered A2 seed")
        if not isinstance(self.synthetic_cells, tuple) or tuple(
            cell.exact_jaccard for cell in self.synthetic_cells
        ) != MINHASH_RECALL_JACCARD_LEVELS:
            raise ValueError("recall audit must contain every registered synthetic cell")
        if any(
            not isinstance(cell, MinHashSyntheticRecallCellV3)
            for cell in self.synthetic_cells
        ):
            raise TypeError("recall audit contains a non-cell value")
        _require_sha256(self.real_sample_identity_sha256, "real recall sample identity")
        for name in (
            "real_dolma_document_count",
            "real_fineweb_document_count",
            "real_exact_pairs_at_or_above_threshold",
            "real_candidate_pairs_at_or_above_threshold",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.real_dolma_document_count < 1 or self.real_fineweb_document_count < 1:
            raise ValueError("real recall audit must sample both dedup source families")
        if self.real_candidate_pairs_at_or_above_threshold > (
            self.real_exact_pairs_at_or_above_threshold
        ):
            raise ValueError("real recall candidate count exceeds exact qualifying pairs")
        if self.status != "REPORT_ONLY_NO_RECALL_FLOOR":
            raise ValueError("16x8 recall qualification is report-only")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_minhash_recall_audit_v3", self
        )

def _shingle_uint64(shingle: bytes) -> int:
    if not isinstance(shingle, bytes):
        raise TypeError("MinHash shingles must be bytes")
    framed = len(shingle).to_bytes(8, "big", signed=False) + shingle
    return int.from_bytes(
        hashlib.sha256(framed).digest()[:8], "little"
    )


def minhash_coefficients_v3(seed: int = A2_DEDUP_SEED) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Draw the exact 128 odd multipliers and 128 offsets bound by A2."""

    _require_uint64(seed, "MinHash seed")
    if seed != A2_DEDUP_SEED:
        raise ValueError("MinHash coefficients require the registered A2 seed")
    if np.__version__ != NUMPY_VERSION:
        raise RuntimeError("runtime NumPy version differs from A2")
    raw = np.random.PCG64(seed).random_raw(MINHASH_COMPONENTS * 2)
    multipliers = np.bitwise_or(raw[:MINHASH_COMPONENTS], np.uint64(1))
    offsets = raw[MINHASH_COMPONENTS:]
    return (
        tuple(int(value) for value in multipliers),
        tuple(int(value) for value in offsets),
    )


def minhash_signature_v3(
    shingles: frozenset[bytes], *, seed: int = A2_DEDUP_SEED
) -> tuple[int, ...]:
    """Return the 128-component uint64 linear-permutation MinHash."""

    if not isinstance(shingles, frozenset) or not shingles:
        raise ValueError("MinHash requires a nonempty frozenset")
    if any(not isinstance(value, bytes) for value in shingles):
        raise TypeError("MinHash shingles must contain only bytes")
    multipliers, offsets = minhash_coefficients_v3(seed)
    a = np.asarray(multipliers, dtype=np.uint64)
    b = np.asarray(offsets, dtype=np.uint64)
    minima = np.full(MINHASH_COMPONENTS, np.iinfo(np.uint64).max, dtype=np.uint64)
    for shingle in sorted(shingles):
        value = np.uint64(_shingle_uint64(shingle))
        minima = np.minimum(minima, a * value + b)
    return tuple(int(value) for value in minima)


def lsh_band_keys_v3(signature: Sequence[int]) -> tuple[bytes, ...]:
    """Pack 16 consecutive 8-row bands as little-endian uint64 byte keys."""

    if not isinstance(signature, Sequence) or isinstance(signature, (str, bytes)):
        raise TypeError("signature must be an integer sequence")
    if len(signature) != MINHASH_COMPONENTS:
        raise ValueError("A2 signatures require exactly 128 components")
    values = tuple(_require_uint64(value, "signature component") for value in signature)
    return tuple(
        b"".join(
            value.to_bytes(8, "little", signed=False)
            for value in values[
                band * MINHASH_ROWS_PER_BAND : (band + 1) * MINHASH_ROWS_PER_BAND
            ]
        )
        for band in range(MINHASH_BANDS)
    )


@dataclass(frozen=True)
class DedupWinnerV3:
    canonical_document_id: str
    canonical_source_record_id: str
    exact_jaccard: Fraction

    def __post_init__(self) -> None:
        _require_sha256(self.canonical_document_id, "canonical_document_id")
        _require_nonempty(
            self.canonical_source_record_id, "canonical_source_record_id"
        )
        if not isinstance(self.exact_jaccard, Fraction):
            raise TypeError("winner exact_jaccard must be a Fraction")
        if not Fraction(0, 1) <= self.exact_jaccard <= Fraction(1, 1):
            raise ValueError("winner exact Jaccard is outside [0, 1]")


def select_dedup_winner_v3(
    query_document: StableDocumentV3,
    candidates: Sequence[StableDocumentV3],
    *,
    threshold: Fraction = NEAR_DUPLICATE_THRESHOLD,
) -> DedupWinnerV3 | None:
    """Drop FineWeb only against Dolma, tying on the literal Dolma record ID."""

    if not isinstance(query_document, StableDocumentV3):
        raise TypeError("dedup query must be a StableDocumentV3")
    if query_document.source != "fineweb_edu":
        raise ValueError("A2 cross-source dedup query must be FineWeb-Edu")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("dedup candidates must be a typed sequence")
    if any(not isinstance(candidate, StableDocumentV3) for candidate in candidates):
        raise TypeError("dedup candidates contain a non-document")
    if any(candidate.source != "dolma_web" for candidate in candidates):
        raise ValueError("A2 canonical dedup candidates must be Dolma web")
    if not isinstance(threshold, Fraction) or threshold != NEAR_DUPLICATE_THRESHOLD:
        raise ValueError("A2 binds exact Jaccard >= 4/5")
    query_shingles = byte_shingles_v3(normalized_match_bytes(query_document.text))
    accepted: list[tuple[Fraction, str, str]] = []
    for candidate in candidates:
        score = exact_jaccard_v3(
            query_shingles,
            byte_shingles_v3(normalized_match_bytes(candidate.text)),
        )
        if score >= threshold:
            accepted.append(
                (score, candidate.stable_source_record_id, candidate.document_id)
            )
    if not accepted:
        return None
    score, source_record_id, document_id = min(
        accepted, key=lambda row: (-row[0], row[1])
    )
    return DedupWinnerV3(document_id, source_record_id, score)


@dataclass(frozen=True)
class SelectionCandidateV3:
    occurrence: DocumentOccurrenceV3
    cluster_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, DocumentOccurrenceV3):
            raise TypeError("selection candidate requires a DocumentOccurrenceV3")
        _require_sha256(self.cluster_id, "cluster_id")

    @property
    def document_id(self) -> str:
        return self.occurrence.document.document_id

    @property
    def retained_byte_count(self) -> int:
        return self.occurrence.document.retained_byte_count


_FIRST_FIT_DISPOSITIONS = (
    "accepted",
    "excluded_cluster",
    "excluded_document",
    "oversized_remaining_capacity",
)


@dataclass(frozen=True)
class FirstFitDecisionV3:
    occurrence_id: str
    document_id: str
    cluster_id: str
    retained_byte_count: int
    remaining_before: int
    remaining_after: int
    disposition: str

    def __post_init__(self) -> None:
        for name in ("occurrence_id", "document_id", "cluster_id"):
            _require_sha256(getattr(self, name), name)
        for name in ("retained_byte_count", "remaining_before", "remaining_after"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.retained_byte_count < 1:
            raise ValueError("retained_byte_count must be positive")
        if self.disposition not in _FIRST_FIT_DISPOSITIONS:
            raise ValueError("unknown greedy first-fit disposition")
        if self.disposition == "accepted":
            if self.retained_byte_count > self.remaining_before:
                raise ValueError("an accepted document does not fit")
            if self.remaining_after != self.remaining_before - self.retained_byte_count:
                raise ValueError("accepted-document remaining bytes are inconsistent")
        elif self.remaining_after != self.remaining_before:
            raise ValueError("a rejected document may not consume capacity")
        if self.disposition == "oversized_remaining_capacity" and (
            self.retained_byte_count <= self.remaining_before
        ):
            raise ValueError("oversized disposition requires a document that does not fit")


@dataclass(frozen=True)
class GreedyFirstFitReceiptV3:
    stream: str
    stratum: str
    target_bytes: int
    initial_used_document_ids_sha256: str
    initial_used_cluster_ids_sha256: str
    candidate_order_sha256: str
    candidate_count: int
    decisions: tuple[FirstFitDecisionV3, ...]
    considered_document_ids_sha256: str
    accepted_document_ids_sha256: str
    skipped_documents_sha256: str
    unscanned_suffix_start: int | None
    realized_bytes: int
    deficit_bytes: int
    source_exhausted: bool
    termination_reason: str

    def __post_init__(self) -> None:
        if self.stream not in {"T", "H"}:
            raise ValueError("first-fit stream must be T or H")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("first-fit receipt uses an unknown stratum")
        if type(self.target_bytes) is not int or self.target_bytes < 1:
            raise ValueError("first-fit target_bytes must be positive")
        for name in (
            "initial_used_document_ids_sha256",
            "initial_used_cluster_ids_sha256",
            "candidate_order_sha256",
            "considered_document_ids_sha256",
            "accepted_document_ids_sha256",
            "skipped_documents_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative exact integer")
        if not isinstance(self.decisions, tuple):
            raise TypeError("first-fit decisions must be a tuple")
        if any(not isinstance(item, FirstFitDecisionV3) for item in self.decisions):
            raise TypeError("first-fit receipt contains a non-decision")
        if len(self.decisions) > self.candidate_count:
            raise ValueError("first-fit considered more candidates than supplied")
        for name in ("realized_bytes", "deficit_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if type(self.source_exhausted) is not bool:
            raise TypeError("source_exhausted must be boolean")
        if self.termination_reason not in {"exact", "within_tolerance"}:
            raise ValueError("successful first-fit termination reason is invalid")
        remaining = self.target_bytes
        for decision in self.decisions:
            if decision.remaining_before != remaining:
                raise ValueError("first-fit decision chain is discontinuous")
            remaining = decision.remaining_after
        realized = sum(item.retained_byte_count for item in self.accepted_decisions)
        if self.realized_bytes != realized or self.deficit_bytes != remaining:
            raise ValueError("first-fit realized/deficit byte arithmetic is inconsistent")
        if self.realized_bytes + self.deficit_bytes != self.target_bytes:
            raise ValueError("first-fit realized bytes plus deficit must equal target")
        if Fraction(self.deficit_bytes, self.target_bytes) > FIRST_FIT_TOLERANCE:
            raise ValueError("first-fit receipt exceeds the A2 0.5% deficit tolerance")
        expected_reason = "exact" if self.deficit_bytes == 0 else "within_tolerance"
        if self.termination_reason != expected_reason:
            raise ValueError("first-fit termination reason differs from its deficit")
        exhausted = len(self.decisions) == self.candidate_count
        if self.source_exhausted is not exhausted:
            raise ValueError("source_exhausted differs from the considered prefix")
        expected_suffix = None if exhausted else len(self.decisions)
        if self.unscanned_suffix_start != expected_suffix:
            raise ValueError("unscanned suffix start differs from the considered prefix")
        considered_hash = _ordered_rows_sha256(
            tuple(item.document_id for item in self.decisions)
        )
        accepted_hash = _ordered_rows_sha256(self.accepted_document_ids)
        skipped_hash = _ordered_rows_sha256(
            tuple(
                (item.document_id, item.retained_byte_count, item.disposition)
                for item in self.decisions
                if item.disposition != "accepted"
            )
        )
        if self.considered_document_ids_sha256 != considered_hash:
            raise ValueError("considered-document identity does not match decisions")
        if self.accepted_document_ids_sha256 != accepted_hash:
            raise ValueError("accepted-document identity does not match decisions")
        if self.skipped_documents_sha256 != skipped_hash:
            raise ValueError("skipped-document identity does not match decisions")
        accepted = self.accepted_document_ids
        if len(accepted) != len(set(accepted)):
            raise ValueError("first-fit receipt accepts a document more than once")
        clusters = self.accepted_cluster_ids
        if len(clusters) != len(set(clusters)):
            raise ValueError("first-fit receipt accepts a cluster more than once")

    @property
    def accepted_decisions(self) -> tuple[FirstFitDecisionV3, ...]:
        return tuple(item for item in self.decisions if item.disposition == "accepted")

    @property
    def accepted_document_ids(self) -> tuple[str, ...]:
        return tuple(item.document_id for item in self.accepted_decisions)

    @property
    def accepted_cluster_ids(self) -> tuple[str, ...]:
        return tuple(item.cluster_id for item in self.accepted_decisions)

    @property
    def shortfall_bytes(self) -> int:
        return self.deficit_bytes

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_greedy_first_fit_receipt_v3", self
        )


def _identity_set_sha256(values: frozenset[str], name: str) -> str:
    if not isinstance(values, frozenset):
        raise TypeError(f"{name} must be a frozenset")
    for value in values:
        _require_sha256(value, name)
    return hashlib.sha256(canonical_json_bytes(tuple(sorted(values)))).hexdigest()


def _ordered_rows_sha256(values: object) -> str:
    return hashlib.sha256(canonical_json_bytes(values)).hexdigest()


def greedy_first_fit_v3(
    candidates: Sequence[SelectionCandidateV3],
    *,
    stream: str,
    stratum: str,
    target_bytes: int,
    used_document_ids: frozenset[str] = frozenset(),
    used_cluster_ids: frozenset[str] = frozenset(),
) -> GreedyFirstFitReceiptV3:
    """Greedily accept fitting documents, continuing after oversized candidates."""

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("first-fit candidates must be a typed sequence")
    if any(not isinstance(item, SelectionCandidateV3) for item in candidates):
        raise TypeError("first-fit candidates contain a non-candidate")
    if stream not in {"T", "H"}:
        raise ValueError("first-fit stream must be T or H")
    if stratum not in GTOK_STRATA:
        raise ValueError("first-fit uses an unknown stratum")
    if type(target_bytes) is not int or target_bytes < 1:
        raise ValueError("target_bytes must be a positive exact integer")
    document_ids = set(used_document_ids)
    cluster_ids = set(used_cluster_ids)
    initial_document_hash = _identity_set_sha256(used_document_ids, "used document ID")
    initial_cluster_hash = _identity_set_sha256(used_cluster_ids, "used cluster ID")
    order_hash = _ordered_rows_sha256(
        tuple(item.occurrence.occurrence_id for item in candidates)
    )
    remaining = target_bytes
    decisions: list[FirstFitDecisionV3] = []
    for candidate in candidates:
        if Fraction(remaining, target_bytes) <= FIRST_FIT_TOLERANCE:
            break
        if candidate.occurrence.document.stratum != stratum:
            raise ValueError("first-fit candidate appears in the wrong stratum")
        before = remaining
        if candidate.document_id in document_ids:
            disposition = "excluded_document"
        elif candidate.cluster_id in cluster_ids:
            disposition = "excluded_cluster"
        elif candidate.retained_byte_count > remaining:
            disposition = "oversized_remaining_capacity"
        else:
            disposition = "accepted"
            remaining -= candidate.retained_byte_count
            document_ids.add(candidate.document_id)
            cluster_ids.add(candidate.cluster_id)
        decisions.append(
            FirstFitDecisionV3(
                occurrence_id=candidate.occurrence.occurrence_id,
                document_id=candidate.document_id,
                cluster_id=candidate.cluster_id,
                retained_byte_count=candidate.retained_byte_count,
                remaining_before=before,
                remaining_after=remaining,
                disposition=disposition,
            )
        )
    if Fraction(remaining, target_bytes) > FIRST_FIT_TOLERANCE:
        raise RuntimeError(
            f"{stream}/{stratum} source exhausted with {remaining} bytes "
            "remaining, above the A2 0.5% tolerance"
        )
    considered = tuple(item.document_id for item in decisions)
    accepted = tuple(
        item.document_id for item in decisions if item.disposition == "accepted"
    )
    skipped = tuple(
        (item.document_id, item.retained_byte_count, item.disposition)
        for item in decisions
        if item.disposition != "accepted"
    )
    exhausted = len(decisions) == len(candidates)
    return GreedyFirstFitReceiptV3(
        stream=stream,
        stratum=stratum,
        target_bytes=target_bytes,
        initial_used_document_ids_sha256=initial_document_hash,
        initial_used_cluster_ids_sha256=initial_cluster_hash,
        candidate_order_sha256=order_hash,
        candidate_count=len(candidates),
        decisions=tuple(decisions),
        considered_document_ids_sha256=_ordered_rows_sha256(considered),
        accepted_document_ids_sha256=_ordered_rows_sha256(accepted),
        skipped_documents_sha256=_ordered_rows_sha256(skipped),
        unscanned_suffix_start=None if exhausted else len(decisions),
        realized_bytes=target_bytes - remaining,
        deficit_bytes=remaining,
        source_exhausted=exhausted,
        termination_reason="exact" if remaining == 0 else "within_tolerance",
    )


@dataclass(frozen=True)
class TrainHeldoutFirstFitReceiptV3:
    training: GreedyFirstFitReceiptV3
    heldout: GreedyFirstFitReceiptV3

    def __post_init__(self) -> None:
        if not isinstance(self.training, GreedyFirstFitReceiptV3) or not isinstance(
            self.heldout, GreedyFirstFitReceiptV3
        ):
            raise TypeError("T/H receipt requires typed first-fit receipts")
        if self.training.stream != "T" or self.heldout.stream != "H":
            raise ValueError("A2 T/H receipt uses T-then-H stream precedence")
        if self.heldout.stratum != self.training.stratum:
            raise ValueError("T/H receipts must cover the same stratum")
        if set(self.heldout.accepted_document_ids) & set(
            self.training.accepted_document_ids
        ):
            raise ValueError("T/H accepted document identities overlap")
        if set(self.heldout.accepted_cluster_ids) & set(
            self.training.accepted_cluster_ids
        ):
            raise ValueError("T/H accepted cluster identities overlap")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_train_heldout_first_fit_receipt_v3", self
        )


def greedy_training_then_heldout_v3(
    ordered_candidates: Sequence[SelectionCandidateV3],
    *,
    stratum: str,
    heldout_target_bytes: int,
    training_target_bytes: int,
    used_document_ids: frozenset[str] = frozenset(),
    used_cluster_ids: frozenset[str] = frozenset(),
) -> TrainHeldoutFirstFitReceiptV3:
    """Select T, then H from T oversize skips followed by T's suffix."""

    if not isinstance(ordered_candidates, Sequence) or isinstance(
        ordered_candidates, (str, bytes)
    ):
        raise TypeError("T/H candidates must be a typed sequence")
    candidates = tuple(ordered_candidates)
    training = greedy_first_fit_v3(
        candidates,
        stream="T",
        stratum=stratum,
        target_bytes=training_target_bytes,
        used_document_ids=used_document_ids,
        used_cluster_ids=used_cluster_ids,
    )
    considered_candidates = candidates[: len(training.decisions)]
    oversized = tuple(
        candidate
        for candidate, decision in zip(
            considered_candidates, training.decisions, strict=True
        )
        if decision.disposition == "oversized_remaining_capacity"
    )
    suffix_start = training.unscanned_suffix_start
    suffix = () if suffix_start is None else candidates[suffix_start:]
    heldout_candidates = (*oversized, *suffix)
    heldout = greedy_first_fit_v3(
        heldout_candidates,
        stream="H",
        stratum=stratum,
        target_bytes=heldout_target_bytes,
        used_document_ids=frozenset(
            (*used_document_ids, *training.accepted_document_ids)
        ),
        used_cluster_ids=frozenset(
            (*used_cluster_ids, *training.accepted_cluster_ids)
        ),
    )
    return TrainHeldoutFirstFitReceiptV3(training=training, heldout=heldout)


def canonical_jsonl_record_bytes_v3(document: StableDocumentV3) -> bytes:
    """Encode exactly ``id/source/stratum/text`` as UTF-8 JSON followed by LF."""

    if not isinstance(document, StableDocumentV3):
        raise TypeError("JSONL serialization requires a StableDocumentV3")
    payload = {
        "id": document.shard_record_id,
        "source": document.source,
        "stratum": document.stratum,
        "text": document.text,
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class ZstdCodecBindingV3:
    serializer: str = "canonical_jsonl_id_source_stratum_text_utf8_lf_v3"
    python_package: str = "zstandard"
    python_package_version: str = "0.25.0"
    libzstd_version: str = "1.5.7"
    compression_level: int = 3
    threads: int = 0
    write_checksum: bool = True
    write_content_size: bool = False
    write_dict_id: bool = False
    dictionary_sha256: None = None
    frame_policy: str = "one_frame_per_shard"

    def __post_init__(self) -> None:
        expected = (
            "canonical_jsonl_id_source_stratum_text_utf8_lf_v3",
            "zstandard",
            "0.25.0",
            "1.5.7",
            3,
            0,
            True,
            False,
            False,
            None,
            "one_frame_per_shard",
        )
        actual = (
            self.serializer,
            self.python_package,
            self.python_package_version,
            self.libzstd_version,
            self.compression_level,
            self.threads,
            self.write_checksum,
            self.write_content_size,
            self.write_dict_id,
            self.dictionary_sha256,
            self.frame_policy,
        )
        if actual != expected:
            raise ValueError("A2 JSONL/zstd codec binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_zstd_codec_binding_v3", self
        )


A2_ZSTD_CODEC_BINDING = ZstdCodecBindingV3()


@dataclass(frozen=True)
class JsonlZstdShardIdentityV3:
    relative_path: str
    record_count: int
    retained_text_bytes: int
    logical_jsonl_sha256: str
    logical_jsonl_bytes: int
    zstd_sha256: str
    zstd_bytes: int
    codec_binding_sha256: str = A2_ZSTD_CODEC_BINDING.receipt_sha256

    def __post_init__(self) -> None:
        _canonical_relative_path(self.relative_path)
        if not self.relative_path.endswith(".jsonl.zst"):
            raise ValueError("A2 corpus shards must use .jsonl.zst")
        for name in (
            "record_count",
            "retained_text_bytes",
            "logical_jsonl_bytes",
            "zstd_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "logical_jsonl_sha256",
            "zstd_sha256",
            "codec_binding_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.codec_binding_sha256 != A2_ZSTD_CODEC_BINDING.receipt_sha256:
            raise ValueError("shard codec binding differs from A2")

    @property
    def content_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_jsonl_zstd_shard_identity_v3", self
        )


@dataclass(frozen=True)
class CorpusContentManifestV3:
    """Deterministic content plus explicitly excluded run-local metadata."""

    run_id: str
    created_at_utc: str
    host_name: str
    process_id: int
    local_output_root: str
    source_asset_manifest_sha256: str
    language_manifest_sha256: str
    dedup_manifest_sha256: str
    selection_manifest_sha256: str
    algorithm_manifest_sha256: str
    shards: tuple[JsonlZstdShardIdentityV3, ...]

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_nonempty(self.created_at_utc, "created_at_utc")
        _require_nonempty(self.host_name, "host_name")
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("process_id must be a positive exact integer")
        _require_nonempty(self.local_output_root, "local_output_root")
        for name in (
            "source_asset_manifest_sha256",
            "language_manifest_sha256",
            "dedup_manifest_sha256",
            "selection_manifest_sha256",
            "algorithm_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.shards, tuple) or not self.shards:
            raise ValueError("content manifest requires at least one shard")
        if any(not isinstance(item, JsonlZstdShardIdentityV3) for item in self.shards):
            raise TypeError("content manifest contains a non-shard")
        paths = tuple(item.relative_path for item in self.shards)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("content-manifest shards must use unique canonical order")

    @property
    def content_payload(self) -> dict[str, object]:
        """Return identity fields, excluding run, time, host, PID and local paths."""

        return {
            "algorithm_manifest_sha256": self.algorithm_manifest_sha256,
            "dedup_manifest_sha256": self.dedup_manifest_sha256,
            "language_manifest_sha256": self.language_manifest_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "shards": tuple(item.content_identity_sha256 for item in self.shards),
            "source_asset_manifest_sha256": self.source_asset_manifest_sha256,
        }

    @property
    def content_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_content_manifest_identity_v3", self.content_payload
        )

    @property
    def audit_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_content_manifest_audit_v3", self
        )


@dataclass(frozen=True)
class TripwireProjectionV3:
    """Exact calibration projection from measured synchronized token throughput."""

    warmup_steps: int
    measured_steps: int
    warmup_a100_microseconds: int
    measured_nonpad_tokens: int
    measured_a100_microseconds: int
    remaining_step_nonpad_token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.warmup_steps != A2_CALIBRATION_WARMUP_STEPS:
            raise ValueError("calibration requires exactly 20 warmup steps")
        if self.measured_steps != A2_CALIBRATION_MEASURED_STEPS:
            raise ValueError("calibration requires exactly 80 measured steps")
        if self.calibration_steps != A2_CALIBRATION_STEPS_MAXIMUM:
            raise ValueError("calibration requires exactly 100 total steps")
        for name in (
            "warmup_a100_microseconds",
            "measured_nonpad_tokens",
            "measured_a100_microseconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if not isinstance(self.remaining_step_nonpad_token_counts, tuple):
            raise TypeError("remaining step token counts must be a tuple")
        if any(
            type(value) is not int or value < 1
            for value in self.remaining_step_nonpad_token_counts
        ):
            raise ValueError("each remaining optimizer step must have positive tokens")

    @property
    def calibration_steps(self) -> int:
        return self.warmup_steps + self.measured_steps

    @property
    def measured_synchronized_tokens_per_second(self) -> Fraction:
        return Fraction(
            self.measured_nonpad_tokens * 1_000_000,
            self.measured_a100_microseconds,
        )

    @property
    def remaining_exact_step_count(self) -> int:
        return len(self.remaining_step_nonpad_token_counts)

    @property
    def calibration_a100_seconds(self) -> Fraction:
        return Fraction(
            self.warmup_a100_microseconds + self.measured_a100_microseconds,
            1_000_000,
        )

    @property
    def projected_remaining_a100_seconds(self) -> Fraction:
        return Fraction(
            sum(self.remaining_step_nonpad_token_counts)
            * self.measured_a100_microseconds,
            self.measured_nonpad_tokens * 1_000_000,
        )

    @property
    def projected_total_a100_seconds(self) -> Fraction:
        return self.calibration_a100_seconds + self.projected_remaining_a100_seconds


@dataclass(frozen=True)
class TripwireDecisionV3:
    action: str
    aggregate_a100_seconds: Fraction
    threshold_a100_seconds: int = A2_TRIPWIRE_A100_SECONDS

    def __post_init__(self) -> None:
        if self.action not in {
            "ALLOW",
            "REJECT_PREFLIGHT",
            "HARD_ABORT_RUN",
            "HARD_ABORT_ALL",
        }:
            raise ValueError("unknown A2 tripwire action")
        if not isinstance(self.aggregate_a100_seconds, Fraction):
            raise TypeError("tripwire aggregate must be an exact Fraction")
        if self.aggregate_a100_seconds < 0:
            raise ValueError("tripwire aggregate may not be negative")
        if self.threshold_a100_seconds != 43_200:
            raise ValueError("A2 cumulative compute tripwire must remain 12 A100-hours")


def tripwire_preflight_v3(
    *,
    consumed_a100_seconds: Fraction,
    active_reservations_a100_seconds: Fraction,
    projection: TripwireProjectionV3,
    calibration_charged_to_consumed: bool,
) -> TripwireDecisionV3:
    """Fail closed on consumed (including calibration) + reservations + remainder."""

    for name, value in (
        ("consumed_a100_seconds", consumed_a100_seconds),
        ("active_reservations_a100_seconds", active_reservations_a100_seconds),
    ):
        if not isinstance(value, Fraction) or value < 0:
            raise ValueError(f"{name} must be a non-negative exact Fraction")
    if not isinstance(projection, TripwireProjectionV3):
        raise TypeError("tripwire preflight requires a typed projection")
    if calibration_charged_to_consumed is not True:
        raise ValueError("calibration must already be charged to consumed compute")
    aggregate = (
        consumed_a100_seconds
        + active_reservations_a100_seconds
        + projection.projected_remaining_a100_seconds
    )
    action = (
        "REJECT_PREFLIGHT"
        if aggregate > A2_TRIPWIRE_A100_SECONDS
        else "ALLOW"
    )
    return TripwireDecisionV3(action, aggregate)


def tripwire_runtime_v3(
    *,
    cumulative_charged_a100_seconds: Fraction,
    run_charged_a100_seconds: Fraction,
    calibration_projection_a100_seconds: Fraction,
) -> TripwireDecisionV3:
    """Apply the global-at-12h and per-run-above-2x hard-abort rules."""

    for name, value in (
        ("cumulative_charged_a100_seconds", cumulative_charged_a100_seconds),
        ("run_charged_a100_seconds", run_charged_a100_seconds),
        ("calibration_projection_a100_seconds", calibration_projection_a100_seconds),
    ):
        if not isinstance(value, Fraction) or value < 0:
            raise ValueError(f"{name} must be a non-negative exact Fraction")
    if calibration_projection_a100_seconds == 0:
        raise ValueError("calibration projection must be positive")
    if cumulative_charged_a100_seconds >= A2_TRIPWIRE_A100_SECONDS:
        return TripwireDecisionV3(
            "HARD_ABORT_ALL", cumulative_charged_a100_seconds
        )
    if run_charged_a100_seconds > 2 * calibration_projection_a100_seconds:
        return TripwireDecisionV3("HARD_ABORT_RUN", run_charged_a100_seconds)
    return TripwireDecisionV3("ALLOW", cumulative_charged_a100_seconds)


@dataclass(frozen=True)
class ReplayRunReceiptV3:
    run_id: str
    process_attestation: ProcessAttestationV3
    input_identity_sha256: str
    dedup_binding_identity_sha256: str
    dedup_decision_ledger_identity_sha256: str
    dedup_exact_match_rate: Fraction
    dedup_near_match_rate: Fraction
    dedup_dropped_bytes: int
    dedup_topup_bytes: int
    minhash_recall_audit: MinHashRecallAuditV3
    content_manifest: CorpusContentManifestV3

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        if not isinstance(self.process_attestation, ProcessAttestationV3):
            raise TypeError("replay run requires a typed ProcessAttestationV3")
        for name in (
            "input_identity_sha256",
            "dedup_binding_identity_sha256",
            "dedup_decision_ledger_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.dedup_binding_identity_sha256 != A2_MINHASH_BINDING.receipt_sha256:
            raise ValueError("replay run uses the wrong registered dedup binding")
        for name in ("dedup_exact_match_rate", "dedup_near_match_rate"):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"{name} must be an exact Fraction")
            if not Fraction(0, 1) <= value <= Fraction(1, 1):
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("dedup_dropped_bytes", "dedup_topup_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if not isinstance(self.minhash_recall_audit, MinHashRecallAuditV3):
            raise TypeError("replay run requires a typed MinHash recall audit")
        if not isinstance(self.content_manifest, CorpusContentManifestV3):
            raise TypeError("replay run requires a CorpusContentManifestV3")
        if self.run_id != self.content_manifest.run_id:
            raise ValueError("replay run ID differs from its content manifest")
        if self.process_attestation.process_id != self.content_manifest.process_id:
            raise ValueError("replay process ID differs from its content manifest")
        if self.process_attestation.output_root != self.content_manifest.local_output_root:
            raise ValueError("replay output root differs from its content manifest")

    @property
    def dedup_replay_tuple(self) -> tuple[object, ...]:
        return (
            self.dedup_binding_identity_sha256,
            self.dedup_decision_ledger_identity_sha256,
            self.dedup_exact_match_rate,
            self.dedup_near_match_rate,
            self.dedup_dropped_bytes,
            self.dedup_topup_bytes,
            self.minhash_recall_audit.receipt_sha256,
        )


@dataclass(frozen=True)
class ProcessAttestationV3:
    """Typed replay process identity; run-local fields remain out of content IDs."""

    executable_sha256: str
    dependency_lock_sha256: str
    environment_identity_sha256: str
    process_id: int
    output_root: str

    def __post_init__(self) -> None:
        for name in (
            "executable_sha256",
            "dependency_lock_sha256",
            "environment_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.process_id) is not int or self.process_id < 1:
            raise ValueError("process attestation PID must be a positive exact integer")
        _require_nonempty(self.output_root, "process attestation output_root")
        path = Path(self.output_root)
        if not path.is_absolute() or str(path.resolve(strict=False)) != self.output_root:
            raise ValueError("process output_root must be an absolute resolved path")

    @property
    def compatibility_identity_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_process_compatibility_v3",
            {
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "environment_identity_sha256": self.environment_identity_sha256,
                "executable_sha256": self.executable_sha256,
            },
        )

    @property
    def attestation_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            "weft1_corpus_process_attestation_v3", self
        )


@dataclass(frozen=True, init=False)
class CorpusGateReceiptV3:
    gate: str
    status: str
    evidence_sha256: str
    first_run_id: str
    second_run_id: str
    authoritative: bool

    def __new__(cls) -> CorpusGateReceiptV3:
        raise TypeError(
            "CorpusGateReceiptV3 is factory-minted by validated independent replays"
        )

    def __post_init__(self) -> None:
        if self.gate not in {"D1", "D2"}:
            raise ValueError("A2 pure replay receipts cover only D1 or D2")
        if (self.status, self.authoritative) not in {
            ("CHECK_PASS", False),
            ("PASS", True),
        }:
            raise ValueError("A2 replay receipt status/authority pairing is invalid")
        _require_sha256(self.evidence_sha256, "gate evidence_sha256")
        _require_run_id(self.first_run_id)
        _require_run_id(self.second_run_id)
        if self.first_run_id == self.second_run_id:
            raise ValueError("gate receipt requires distinct replay run IDs")

    @property
    def receipt_sha256(self) -> str:
        return execution_authority_v3_bound_sha256(
            f"weft1_corpus_{self.gate.lower()}_gate_receipt_v3", self
        )


def _mint_corpus_gate_receipt_v3(
    *,
    gate: str,
    evidence_sha256: str,
    first_run_id: str,
    second_run_id: str,
    authoritative: bool,
) -> CorpusGateReceiptV3:
    """Mint only after validation; metadata-only checks stay non-authoritative."""

    receipt = object.__new__(CorpusGateReceiptV3)
    object.__setattr__(receipt, "gate", gate)
    object.__setattr__(receipt, "status", "PASS" if authoritative else "CHECK_PASS")
    object.__setattr__(receipt, "evidence_sha256", evidence_sha256)
    object.__setattr__(receipt, "first_run_id", first_run_id)
    object.__setattr__(receipt, "second_run_id", second_run_id)
    object.__setattr__(receipt, "authoritative", authoritative)
    receipt.__post_init__()
    return receipt


def validate_independent_replays_v3(
    first: ReplayRunReceiptV3,
    second: ReplayRunReceiptV3,
) -> tuple[CorpusGateReceiptV3, CorpusGateReceiptV3]:
    """Check two replay claims; this pure layer cannot prove process/filesystem facts."""

    if not isinstance(first, ReplayRunReceiptV3) or not isinstance(
        second, ReplayRunReceiptV3
    ):
        raise TypeError("replay validation requires two ReplayRunReceiptV3 values")
    if first.run_id == second.run_id:
        raise ValueError("independent replay requires distinct run IDs")
    first_attestation = first.process_attestation
    second_attestation = second.process_attestation
    if first_attestation.process_id == second_attestation.process_id:
        raise ValueError("independent replay requires distinct process IDs")
    first_root = Path(first_attestation.output_root)
    second_root = Path(second_attestation.output_root)
    if (
        first_root == second_root
        or first_root in second_root.parents
        or second_root in first_root.parents
    ):
        raise ValueError("independent replay requires distinct non-overlapping roots")
    if first_attestation.compatibility_identity_sha256 != (
        second_attestation.compatibility_identity_sha256
    ):
        raise ValueError(
            "independent replay requires identical executable/dependency/environment"
        )
    if first.input_identity_sha256 != second.input_identity_sha256:
        raise ValueError("D1 failed: replay input identities differ")
    if first.content_manifest.content_identity_sha256 != (
        second.content_manifest.content_identity_sha256
    ):
        raise ValueError("D1 failed: replay corpus content identities differ")
    first_shards = tuple(
        item.content_identity_sha256 for item in first.content_manifest.shards
    )
    second_shards = tuple(
        item.content_identity_sha256 for item in second.content_manifest.shards
    )
    if first_shards != second_shards:
        raise ValueError("D1 failed: replay shard identities differ")
    if first.dedup_replay_tuple != second.dedup_replay_tuple:
        raise ValueError("D2 failed: independent dedup evidence differs")

    d1_evidence = execution_authority_v3_bound_sha256(
        "weft1_corpus_d1_replay_evidence_v3",
        {
            "content_identity_sha256": first.content_manifest.content_identity_sha256,
            "first_process_attestation_sha256": first_attestation.attestation_sha256,
            "first_run_id": first.run_id,
            "input_identity_sha256": first.input_identity_sha256,
            "second_process_attestation_sha256": second_attestation.attestation_sha256,
            "second_run_id": second.run_id,
            "shard_identity_sha256s": first_shards,
        },
    )
    d2_evidence = execution_authority_v3_bound_sha256(
        "weft1_corpus_d2_replay_evidence_v3",
        {
            "dedup_replay_tuple": first.dedup_replay_tuple,
            "first_process_attestation_sha256": first_attestation.attestation_sha256,
            "first_run_id": first.run_id,
            "input_identity_sha256": first.input_identity_sha256,
            "second_process_attestation_sha256": second_attestation.attestation_sha256,
            "second_run_id": second.run_id,
        },
    )
    return (
        _mint_corpus_gate_receipt_v3(
            gate="D1",
            evidence_sha256=d1_evidence,
            first_run_id=first.run_id,
            second_run_id=second.run_id,
            authoritative=False,
        ),
        _mint_corpus_gate_receipt_v3(
            gate="D2",
            evidence_sha256=d2_evidence,
            first_run_id=first.run_id,
            second_run_id=second.run_id,
            authoritative=False,
        ),
    )


__all__ = [
    "A2_CAMPAIGN_ROOT_SEED",
    "A2_CALIBRATION_STEPS_MAXIMUM",
    "A2_CHARGED_ATTEMPT_STATUSES",
    "A2_DEDUP_SEED",
    "A2_LANGUAGE_ID_BINDING",
    "A2_MATCH_NORMALIZATION_BINDING",
    "A2_MINHASH_BINDING",
    "MINHASH_RECALL_JACCARD_LEVELS",
    "A2_PIPELINE_SEEDS",
    "A2_STREAM_PRECEDENCE",
    "A2_TRIPWIRE_A100_SECONDS",
    "A2_ZSTD_CODEC_BINDING",
    "CorpusContentManifestV3",
    "CorpusGateReceiptV3",
    "DedupWinnerV3",
    "DocumentOccurrenceV3",
    "FirstFitDecisionV3",
    "FIRST_FIT_TOLERANCE",
    "GTOK_AMENDMENT_A2_SHA256",
    "GTOK_EXECUTION_AUTHORITY_CHAIN_V3",
    "GreedyFirstFitReceiptV3",
    "JsonlZstdShardIdentityV3",
    "LanguageIdBindingV3",
    "LanguageIdDecisionV3",
    "MatchNormalizationBindingV3",
    "MinHashBindingV3",
    "MinHashRecallAuditV3",
    "MinHashSyntheticRecallCellV3",
    "ProcessAttestationV3",
    "ReplayRunReceiptV3",
    "SelectionCandidateV3",
    "StableDocumentV3",
    "TrainHeldoutFirstFitReceiptV3",
    "TripwireDecisionV3",
    "TripwireProjectionV3",
    "ZstdCodecBindingV3",
    "byte_shingles_v3",
    "canonical_jsonl_record_bytes_v3",
    "exact_jaccard_v3",
    "execution_authority_v3_bound_sha256",
    "greedy_first_fit_v3",
    "greedy_training_then_heldout_v3",
    "ideal_lsh_candidate_probability_v3",
    "is_exact_duplicate_v3",
    "language_id_decision_v3",
    "language_id_decision_from_predictions_v3",
    "language_backend_input_bytes_v3",
    "language_scoring_bytes_v3",
    "lsh_band_keys_v3",
    "minhash_coefficients_v3",
    "minhash_signature_v3",
    "normalize_match_text",
    "normalized_match_bytes",
    "pipeline_seed",
    "select_dedup_winner_v3",
    "tripwire_preflight_v3",
    "tripwire_runtime_v3",
    "validate_independent_replays_v3",
]
