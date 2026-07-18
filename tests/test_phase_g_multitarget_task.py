from __future__ import annotations

from copy import deepcopy

from training.branching_relations_task import BranchingRelationsConfig
from training.phase_g_multitarget_task import (
    build_multitarget_rows,
    validate_multitarget_rows,
)


def build_rows(*, targets_per_prompt: int | None = None) -> list[dict]:
    return build_multitarget_rows(
        BranchingRelationsConfig(rows_per_depth=2, max_depth=2, seed=2718),
        split="multitarget_train",
        rendering="symbolic",
        n_symbols=12,
        targets_per_prompt=targets_per_prompt,
    )


def test_multitarget_rows_preserve_prompt_and_cover_distinct_valid_targets() -> None:
    rows = build_rows()
    validation = validate_multitarget_rows(rows)

    assert validation["status"] == "passed"
    assert validation["base_problem_groups"] == 4
    assert validation["groups_with_multiple_targets"] == 4
    assert validation["all_reachable_targets_covered"] is True

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["base_problem_id"], []).append(row)
    for variants in groups.values():
        assert len(variants) == len(variants[0]["reachable_symbols"])
        assert len({row["target"] for row in variants}) == len(variants)
        assert len({row["question"] for row in variants}) == 1
        assert len({str(row["successors"]) for row in variants}) == 1
        for row in variants:
            chain = row["sampled_chain"]
            assert chain[-1] == row["target"]
            assert row["loop_completions"] == [f" {symbol}" for symbol in chain[1:]]
            assert row["completion"] == f" {row['target']}"


def test_multitarget_rows_are_deterministic_and_support_a_target_cap() -> None:
    full_first = build_rows()
    full_second = build_rows()
    capped = build_rows(targets_per_prompt=2)

    assert full_first == full_second
    capped_validation = validate_multitarget_rows(capped)
    assert capped_validation["status"] == "passed"
    assert capped_validation["all_reachable_targets_covered"] is False
    groups: dict[str, list[dict]] = {}
    for row in capped:
        groups.setdefault(row["base_problem_id"], []).append(row)
    assert all(len(variants) == 2 for variants in groups.values())
    assert all(row["target_variant_count"] == 2 for row in capped)


def test_multitarget_validation_rejects_duplicate_target_within_a_prompt_group() -> None:
    rows = build_rows()
    corrupted = deepcopy(rows)
    group_id = corrupted[0]["base_problem_id"]
    group_indices = [
        index for index, row in enumerate(corrupted) if row["base_problem_id"] == group_id
    ]
    corrupted[group_indices[1]]["target"] = corrupted[group_indices[0]]["target"]

    validation = validate_multitarget_rows(corrupted)

    assert validation["status"] == "failed"
    assert any("duplicate targets" in message for message in validation["errors"])
