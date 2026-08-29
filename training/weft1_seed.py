"""Torch-free deterministic seed derivation shared by WEFT-1 pipelines.

The corpus P-A environment intentionally does not install the model-training
stack.  Keep the stable SHA-256 seed contract here so corpus code can consume
it without importing :mod:`models`, while ``models.ablation_lm.rng`` retains
its public compatibility wrapper for training callers.
"""

from __future__ import annotations

import hashlib
import re


_SOURCE_KEY_PATTERN = re.compile(
    r"[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|[0-9]+))*"
)
_SEED_DOMAIN = b"WEFT-1/module-rng/seed/v1\x00"


def _validate_base_seed(base_seed: int) -> None:
    if type(base_seed) is not int:
        raise TypeError("base_seed must be an exact integer")


def _validate_source_key(source_key: str) -> None:
    if type(source_key) is not str:
        raise TypeError("source_key must be an exact string")
    if _SOURCE_KEY_PATTERN.fullmatch(source_key) is None:
        raise ValueError(
            "source_key must be canonical lowercase dotted module coordinates; "
            "segments may be identifiers or non-negative decimal indices"
        )


def _validate_replica(replica: int) -> None:
    if type(replica) is not int:
        raise TypeError("replica must be an exact integer")
    if replica < 0:
        raise ValueError("replica must be non-negative")


def _encode_field(value: int | str) -> bytes:
    raw = str(value).encode("utf-8")
    return len(raw).to_bytes(8, byteorder="big", signed=False) + raw


def derive_module_seed(base_seed: int, source_key: str, replica: int = 0) -> int:
    """Return the stable 64-bit seed for one canonical random source.

    The length-prefixed encoding prevents boundary ambiguities such as a key
    suffix being mistaken for part of the replica.  The exact signed decimal
    spelling of ``base_seed`` participates in the hash; no Python hash or
    process-global salt is involved.
    """

    _validate_base_seed(base_seed)
    _validate_source_key(source_key)
    _validate_replica(replica)
    payload = (
        _SEED_DOMAIN
        + _encode_field(base_seed)
        + _encode_field(source_key)
        + _encode_field(replica)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


__all__ = ["derive_module_seed"]
