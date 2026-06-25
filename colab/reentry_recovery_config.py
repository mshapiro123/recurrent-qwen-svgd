"""Pure configuration helpers for Stage 4 re-entry recovery runs."""

from __future__ import annotations

from typing import Any


def int_dict_max_key(payload: Any, default: int) -> int:
    """Return the largest positive integer-like key from a count dict."""
    values: list[int] = []
    if isinstance(payload, dict):
        for key in payload:
            try:
                value = int(key)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
    return max(values) if values else default


def mode_rows_from_counts(mode_counts: Any) -> str:
    """Convert mode counts into the gate format: direct=12,deep_narrow=8."""
    if not isinstance(mode_counts, dict):
        return ""
    parts: list[str] = []
    for mode, count in sorted(mode_counts.items()):
        try:
            n = int(count)
        except (TypeError, ValueError):
            continue
        if n > 0:
            parts.append(f"{mode}={n}")
    return ",".join(parts)


def target_loop_rows_from_counts(target_loop_counts: Any) -> str:
    """Convert target-loop counts into strict SFT gate requirements.

    The Stage 4 depth curriculum depends on preserving the actual count per
    target loop. Collapsing every observed loop to a minimum of one row can let a
    fake ladder through the preflight gate and erase the intended depth signal.
    """
    if not isinstance(target_loop_counts, dict):
        return ""
    sortable: list[tuple[int, int]] = []
    for loop, count in target_loop_counts.items():
        try:
            loop_value = int(loop)
            count_value = int(count)
        except (TypeError, ValueError):
            continue
        if loop_value > 0 and count_value > 0:
            sortable.append((loop_value, count_value))
    return ",".join(f"{loop}={count}" for loop, count in sorted(sortable))
