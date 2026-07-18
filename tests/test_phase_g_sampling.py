from __future__ import annotations

import random

import pytest

from training.phase_g_sampling import build_base_problem_groups, sample_phase_g_row_index


def test_base_problem_sampler_is_reproducible_and_balances_groups_not_variants() -> None:
    rows = [
        {"id": f"a_{index}", "base_problem_id": "a"}
        for index in range(5)
    ] + [{"id": "b_0", "base_problem_id": "b"}]
    groups = build_base_problem_groups(rows)

    first_rng = random.Random(99)
    second_rng = random.Random(99)
    first = [
        sample_phase_g_row_index(first_rng, rows=rows, policy="base_problem_uniform", groups=groups)
        for _ in range(2_000)
    ]
    second = [
        sample_phase_g_row_index(second_rng, rows=rows, policy="base_problem_uniform", groups=groups)
        for _ in range(2_000)
    ]

    assert first == second
    base_counts = {"a": 0, "b": 0}
    for index in first:
        base_counts[rows[index]["base_problem_id"]] += 1
    assert abs(base_counts["a"] - base_counts["b"]) < 120


def test_row_uniform_sampler_and_group_validation_have_explicit_contracts() -> None:
    rows = [
        {"id": "a_0", "base_problem_id": "a"},
        {"id": "a_1", "base_problem_id": "a"},
        {"id": "b_0", "base_problem_id": "b"},
    ]
    groups = build_base_problem_groups(rows)
    sampler = random.Random(7)

    assert 0 <= sample_phase_g_row_index(
        sampler, rows=rows, policy="row_uniform", groups=groups
    ) < len(rows)
    with pytest.raises(ValueError, match="base_problem_id"):
        build_base_problem_groups([{"id": "missing"}])
    with pytest.raises(ValueError, match="unknown"):
        sample_phase_g_row_index(sampler, rows=rows, policy="unknown", groups=groups)
