"""Shared Stage 5 limit parsing helpers."""

from __future__ import annotations


FULL_LIMIT_TOKENS = {"", "none", "all", "full", "unlimited", "0", "-1"}


def parse_optional_limit(value: str | int | None, *, default: int | None = None) -> int | None:
    """Parse a Colab limit environment variable.

    Positive integers cap the number of examples. Common full-run spellings
    return ``None``, which lets eval scripts load the entire split.
    """

    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in FULL_LIMIT_TOKENS:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        return None
    return parsed


def limit_args(limit: int | None) -> list[str]:
    """Return CLI args for eval scripts that accept an optional ``--limit``."""

    return [] if limit is None else ["--limit", str(limit)]


def limit_label(limit: int | None) -> str:
    """Stable label for reports and child run ids."""

    return "full" if limit is None else str(limit)


def difficulty_args(buckets: str | None, examples_per_difficulty: int | None) -> list[str]:
    """Return eval args for deterministic difficulty-stratified ARC slices."""

    args: list[str] = []
    if buckets:
        args += ["--difficulty_buckets", buckets]
    if examples_per_difficulty is not None:
        args += ["--examples_per_difficulty", str(examples_per_difficulty)]
    return args
