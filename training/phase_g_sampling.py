"""Sampling policies for Phase G training rows."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def build_base_problem_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Index rows by their shared prompt problem for balanced multi-target sampling."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        base_problem_id = str(row.get("base_problem_id", ""))
        if not base_problem_id:
            raise ValueError(
                "base_problem_uniform sampling requires nonempty base_problem_id on every row"
            )
        groups[base_problem_id].append(index)
    if not groups:
        raise ValueError("Cannot sample an empty Phase G row collection")
    return {name: tuple(indices) for name, indices in groups.items()}


def sample_phase_g_row_index(
    sampler: random.Random,
    *,
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    groups: Mapping[str, Sequence[int]] | None = None,
) -> int:
    """Choose a row under a declared, resume-stable Phase G sampling policy."""

    if not rows:
        raise ValueError("Cannot sample an empty Phase G row collection")
    if policy == "row_uniform":
        return sampler.randrange(len(rows))
    if policy != "base_problem_uniform":
        raise ValueError(f"Unknown Phase G sampling policy: {policy}")
    if groups is None:
        groups = build_base_problem_groups(rows)
    base_problem_ids = sorted(groups)
    if not base_problem_ids:
        raise ValueError("base_problem_uniform sampling requires at least one group")
    chosen = base_problem_ids[sampler.randrange(len(base_problem_ids))]
    variants = tuple(groups[chosen])
    if not variants:
        raise ValueError(f"Empty Phase G sampling group: {chosen}")
    return variants[sampler.randrange(len(variants))]
