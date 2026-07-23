"""Locked post-hoc train-row readout for the terminal oracle conditioners."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Iterable


READOUT_SEED = 20260722
MATCHED_VARIANTS_BY_DEPTH = {1: 16, 2: 22, 3: 27, 4: 41}
MATCHED_GROUPS_BY_DEPTH = {1: 8, 2: 8, 3: 8, 4: 8}
MATCHED_ROWS = 106
MATCHED_GROUPS = 32
MATCHED_TRANSITIONS = 305
FIT_THRESHOLD = 0.85
NO_FIT_THRESHOLD = 0.25
HELDOUT_NONDEFAULT = {"additive": 0.14351851851851852, "film": 0.1574074074074074}


def _choose_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    group_count: int,
    variant_count: int,
    rng: random.Random,
) -> list[str]:
    """Choose a seeded group subset with an exact variant-count sum."""

    candidates = sorted(groups)
    rng.shuffle(candidates)
    states: dict[tuple[int, int], list[str]] = {(0, 0): []}
    for group in candidates:
        size = len(groups[group])
        next_states = dict(states)
        for (count, total), selected in states.items():
            key = (count + 1, total + size)
            if key[0] <= group_count and key[1] <= variant_count and key not in next_states:
                next_states[key] = [*selected, group]
        states = next_states
    key = (group_count, variant_count)
    if key not in states:
        raise AssertionError(
            f"Cannot select {group_count} groups totaling {variant_count} variants"
        )
    return states[key]


def select_matched_training_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int = READOUT_SEED,
) -> list[dict[str, Any]]:
    by_depth_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_depth_group[int(row["depth"])][str(row["base_problem_id"])].append(row)

    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for depth in sorted(MATCHED_VARIANTS_BY_DEPTH):
        groups = by_depth_group[depth]
        chosen = _choose_groups(
            groups,
            group_count=MATCHED_GROUPS_BY_DEPTH[depth],
            variant_count=MATCHED_VARIANTS_BY_DEPTH[depth],
            rng=rng,
        )
        for group in sorted(chosen):
            selected.extend(
                sorted(groups[group], key=lambda row: str(row["id"]))
            )
    validate_matched_training_rows(selected)
    return selected


def validate_matched_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    depth_counts: dict[int, int] = defaultdict(int)
    depth_groups: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        depth = int(row["depth"])
        depth_counts[depth] += 1
        depth_groups[depth].add(str(row["base_problem_id"]))
    observed = {
        "rows": len(rows),
        "groups": len({str(row["base_problem_id"]) for row in rows}),
        "transitions": sum(int(row["depth"]) for row in rows),
        "variants_by_depth": dict(sorted(depth_counts.items())),
        "groups_by_depth": {
            depth: len(groups) for depth, groups in sorted(depth_groups.items())
        },
    }
    expected = {
        "rows": MATCHED_ROWS,
        "groups": MATCHED_GROUPS,
        "transitions": MATCHED_TRANSITIONS,
        "variants_by_depth": MATCHED_VARIANTS_BY_DEPTH,
        "groups_by_depth": MATCHED_GROUPS_BY_DEPTH,
    }
    if observed != expected:
        raise AssertionError(f"Matched train readout manifest mismatch: {observed} != {expected}")
    return observed


def row_id_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_nondefault_control(value: float) -> str:
    if float(value) >= FIT_THRESHOLD:
        return "fit_seen_command_mapping"
    if float(value) <= NO_FIT_THRESHOLD:
        return "did_not_fit_command_mapping"
    return "partial_fit"


def preregistration_payload() -> dict[str, Any]:
    return {
        "kind": "phase_g_oracle_train_readout_preregistration",
        "status": "locked_before_gpu_evaluation",
        "posthoc_diagnostic_only": True,
        "registered_heldout_verdict_mutable": False,
        "seed": READOUT_SEED,
        "matched_subset": {
            "rows": MATCHED_ROWS,
            "groups": MATCHED_GROUPS,
            "transitions": MATCHED_TRANSITIONS,
            "variants_by_depth": MATCHED_VARIANTS_BY_DEPTH,
            "groups_by_depth": MATCHED_GROUPS_BY_DEPTH,
        },
        "full_training_readout": {
            "rows": 1899,
            "groups": 512,
            "transitions": 5617,
        },
        "interpretation_bands": {
            "fit_seen_command_mapping_at_or_above": FIT_THRESHOLD,
            "did_not_fit_command_mapping_at_or_below": NO_FIT_THRESHOLD,
            "between": "partial_fit",
        },
        "metrics": [
            "nondefault_transition_control",
            "overall_transition_control",
            "transition_legality",
            "terminal_validity",
            "per_loop_localization",
        ],
        "prohibitions": [
            "training",
            "parameter_mutation",
            "changing_the_registered_BOTH_FAIL_verdict",
            "automatic_successor_authorization",
        ],
    }


def combined_readout(
    *,
    matched_summaries: dict[str, dict[str, Any]],
    full_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "phase_g_oracle_train_readout",
        "status": "finished_posthoc_diagnostic",
        "registered_heldout_verdict": "BOTH_FAIL",
        "registered_heldout_verdict_changed": False,
        "automatic_successor_authorized": False,
        "arms": {},
    }
    for route in ("additive", "film"):
        matched = matched_summaries[route]
        full = full_summaries[route]
        matched_rate = float(matched["transition_control"]["nondefault"]["control_rate"])
        full_rate = float(full["transition_control"]["nondefault"]["control_rate"])
        result["arms"][route] = {
            "matched_nondefault_control": matched_rate,
            "matched_interpretation": classify_nondefault_control(matched_rate),
            "full_nondefault_control": full_rate,
            "full_interpretation": classify_nondefault_control(full_rate),
            "heldout_nondefault_control": HELDOUT_NONDEFAULT[route],
            "matched_summary": matched,
            "full_summary": full,
        }
    return result
