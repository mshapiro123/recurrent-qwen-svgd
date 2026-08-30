"""Literal, non-secret contracts for the WEFT-1 P-B hermetic screen.

The two sealed families do not share an algorithm.  CONFIRM is screened from
its governed P3.1 membership using A2 byte shingles.  EVAL-E is screened from
the already-published anonymous TM-0 index using that index's older character
normalization and MinHash family.  Keeping the profiles separate prevents a
receipt from claiming that either family was checked with the other's rules.

This module contains no filesystem or sealed-data access.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from typing import Mapping, Sequence

from training.weft1_corpus_a2 import (
    A2_MATCH_NORMALIZATION_BINDING,
    A2_MINHASH_BINDING,
    MINHASH_COMPONENTS,
    MINHASH_SHINGLE_WIDTH,
)
from training.weft1_gtok_contract import canonical_json_bytes


CONFIRM_BATTERIES = (
    "CONFIRM:arc_challenge",
    "CONFIRM:arc_easy",
    "CONFIRM:gsm8k",
    "CONFIRM:mbpp",
    "CONFIRM:mmlu",
    "CONFIRM:tier1",
)
EVAL_E_BATTERIES = ("EVAL-E",)
DECON_BATTERIES = (*CONFIRM_BATTERIES, *EVAL_E_BATTERIES)

# Public, score-blind identities reread from the durable P3.1 completion
# receipt.  The earlier d831... identity is explicitly labeled a local dry run
# and is deliberately not promoted to this governed completion domain.
GOVERNED_CONFIRM_COMPLETION_RECEIPT_SHA256 = (
    "1b6e40149034047a35cd669a6f8fd045c26330ac9b67075e0cb39b0271d1802b"
)
GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256 = (
    "503d6a5551f187cb96d80de45ee0d1deff9d3202186aa4995cd80b0bfba7653f"
)
GOVERNED_CONFIRM_SEAL_SET_SHA256 = (
    "6edd229f934477ae978bb70193df90a6b90830408e7cbc1286c5dea32259377b"
)
GOVERNED_CONFIRM_SOURCE_ROWS_SHA256 = (
    "5e32eb1905b05076a59b2c5b315ccf9319c04eda18af450565128fd34c18ffa5"
)
GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256 = (
    "1bcb847e02652881b0161718f73f13faeb30ede93f95f6a50152af900cdedef7"
)

# Existing anonymous sealed index and the lock that generated it.  These are
# non-secret artifact identities, not plaintext or membership identifiers.
GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256 = (
    "c95a5aef53667d486ea6ba0852186efbe7fe44d1419d62ff969c6778bc123253"
)
GOVERNED_EVAL_E_LOCK_SHA256 = (
    "45a241647453771a8ab6a615f895a511ec15cc954853e98faa9d23f1f73da6f5"
)

CONFIRM_PROFILE_ID = "confirm_a2_salted_byte13_safe_prefix_v1"
EVAL_E_PROFILE_ID = "eval_e_tm0_anonymous_index_locked_v1"
CONFIRM_PROMPT_RENDERER = "p31_prompt_only_reader_aware_v1"

LEGACY_EVAL_E_NORMALIZATION = "unicode_nfkc_casefold_whitespace_collapse_v1"
LEGACY_EVAL_E_EXACT_HASH = "sha256_salt_utf8_normalized_text"
LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE = 13
LEGACY_EVAL_E_MINHASH_COMPONENTS = 128
LEGACY_EVAL_E_MINHASH_SEED = 202_608_257
LEGACY_EVAL_E_LSH_BANDS = 16
LEGACY_EVAL_E_LSH_ROWS_PER_BAND = 8
LEGACY_EVAL_E_THRESHOLD = Fraction(4, 5)


def algorithm_profiles() -> list[dict[str, object]]:
    """Return the exact public algorithm-profile projection for the receipt."""

    return [
        {
            "battery_scope": list(CONFIRM_BATTERIES),
            "governed_membership": {
                "complete_ledger_sha256": (
                    GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256
                ),
                "completion_receipt_sha256": (
                    GOVERNED_CONFIRM_COMPLETION_RECEIPT_SHA256
                ),
                "seal_set_sha256": GOVERNED_CONFIRM_SEAL_SET_SHA256,
                "source_manifest_sha256": (
                    GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256
                ),
                "source_rows_sha256": GOVERNED_CONFIRM_SOURCE_ROWS_SHA256,
                "split_seed": 20260809,
            },
            "exact": {
                "digest": "sha256_salt_u64be_length_normalized_bytes_v1",
                "normalization_binding_sha256": (
                    A2_MATCH_NORMALIZATION_BINDING.receipt_sha256
                ),
                "salt_source": "eval_e_locked_salt_private_in_process",
            },
            "near": {
                "decision": "salted_prefix_filter_then_exact_jaccard",
                "exact_jaccard_denominator": 5,
                "exact_jaccard_numerator": 4,
                "minhash_binding_sha256": A2_MINHASH_BINDING.receipt_sha256,
                "minhash_lsh_role": "diagnostic_only_not_clean_gate",
                "prefix_selection_digest": "sha256_salt_u64be_length_then_shingle",
                "prefix_size": "floor(sealed_shingle_count/5)+1",
                "runtime_lookup": "raw_shingle_regex_lookahead_inside_hermetic_child",
                "shingle_width_bytes": MINHASH_SHINGLE_WIDTH,
            },
            "profile_id": CONFIRM_PROFILE_ID,
            "prompt_renderer": CONFIRM_PROMPT_RENDERER,
        },
        {
            "battery_scope": list(EVAL_E_BATTERIES),
            "exact": {
                "digest": LEGACY_EVAL_E_EXACT_HASH,
                "normalization": LEGACY_EVAL_E_NORMALIZATION,
            },
            "near": {
                "comparison": "deterministic_26_component_prefix_then_full_anonymous_signature",
                "components": LEGACY_EVAL_E_MINHASH_COMPONENTS,
                "estimated_jaccard_denominator": 5,
                "estimated_jaccard_numerator": 4,
                "lsh_role": "not_used_for_clean_gate",
                "minhash_seed": LEGACY_EVAL_E_MINHASH_SEED,
                "prefix_components": 26,
                "shingle_unit": "normalized_unicode_character",
                "shingle_width_characters": LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE,
            },
            "profile_id": EVAL_E_PROFILE_ID,
        },
    ]


def algorithm_profiles_commitment_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(algorithm_profiles())).hexdigest()


def safe_prefix_size(sealed_shingle_count: int) -> int:
    """Return a prefix larger than the maximum missing sealed share at J>=4/5."""

    if type(sealed_shingle_count) is not int or sealed_shingle_count < 1:
        raise ValueError("sealed shingle count must be a positive exact integer")
    return sealed_shingle_count // 5 + 1


def jaccard_at_least_four_fifths(
    left: Sequence[object] | set[object] | frozenset[object],
    right: Sequence[object] | set[object] | frozenset[object],
) -> bool:
    left_set = frozenset(left)
    right_set = frozenset(right)
    union_count = len(left_set | right_set)
    if union_count == 0:
        return True
    return 5 * len(left_set & right_set) >= 4 * union_count


def length_can_reach_four_fifths(left_count: int, right_count: int) -> bool:
    """Necessary size-ratio condition for set Jaccard >= 4/5."""

    if (
        type(left_count) is not int
        or type(right_count) is not int
        or left_count < 1
        or right_count < 1
    ):
        raise ValueError("shingle counts must be positive exact integers")
    return 4 * left_count <= 5 * right_count and 4 * right_count <= 5 * left_count


def full_signature_match_count(
    left: Sequence[int], right: Sequence[int]
) -> int:
    """Compare every component; this deliberately performs no LSH filtering."""

    if len(left) != MINHASH_COMPONENTS or len(right) != MINHASH_COMPONENTS:
        raise ValueError("full MinHash comparison requires 128 components")
    return sum(a == b for a, b in zip(left, right, strict=True))


def signature_at_least_four_fifths(
    left: Sequence[int], right: Sequence[int]
) -> bool:
    return 5 * full_signature_match_count(left, right) >= 4 * MINHASH_COMPONENTS


def lsh_shares_band(
    left: Sequence[int],
    right: Sequence[int],
    *,
    bands: int = LEGACY_EVAL_E_LSH_BANDS,
    rows_per_band: int = LEGACY_EVAL_E_LSH_ROWS_PER_BAND,
) -> bool:
    if len(left) != bands * rows_per_band or len(right) != len(left):
        raise ValueError("signature and LSH shape disagree")
    return any(
        tuple(left[start : start + rows_per_band])
        == tuple(right[start : start + rows_per_band])
        for start in range(0, len(left), rows_per_band)
    )


def exact_mapping_keys(value: Mapping[str, object], expected: set[str]) -> bool:
    """Small helper used by both the runner and fail-closed receipt loader."""

    return isinstance(value, Mapping) and set(value) == expected


__all__ = [
    "CONFIRM_BATTERIES",
    "CONFIRM_PROFILE_ID",
    "CONFIRM_PROMPT_RENDERER",
    "DECON_BATTERIES",
    "EVAL_E_BATTERIES",
    "EVAL_E_PROFILE_ID",
    "GOVERNED_EVAL_E_ANONYMOUS_INDEX_SHA256",
    "GOVERNED_EVAL_E_LOCK_SHA256",
    "GOVERNED_CONFIRM_COMPLETION_RECEIPT_SHA256",
    "GOVERNED_CONFIRM_COMPLETE_LEDGER_SHA256",
    "GOVERNED_CONFIRM_SEAL_SET_SHA256",
    "GOVERNED_CONFIRM_SOURCE_MANIFEST_SHA256",
    "GOVERNED_CONFIRM_SOURCE_ROWS_SHA256",
    "LEGACY_EVAL_E_CHARACTER_SHINGLE_SIZE",
    "LEGACY_EVAL_E_EXACT_HASH",
    "LEGACY_EVAL_E_LSH_BANDS",
    "LEGACY_EVAL_E_LSH_ROWS_PER_BAND",
    "LEGACY_EVAL_E_MINHASH_COMPONENTS",
    "LEGACY_EVAL_E_MINHASH_SEED",
    "LEGACY_EVAL_E_NORMALIZATION",
    "LEGACY_EVAL_E_THRESHOLD",
    "algorithm_profiles",
    "algorithm_profiles_commitment_sha256",
    "full_signature_match_count",
    "jaccard_at_least_four_fifths",
    "length_can_reach_four_fifths",
    "lsh_shares_band",
    "safe_prefix_size",
    "signature_at_least_four_fifths",
]
