from __future__ import annotations

from training.abductive_injective_task import (
    AbductiveInjectiveConfig,
    build_injective_abduction_instance,
    build_multimodal_abduction_instance,
    build_row,
    build_rows,
    exact_depth_preimages,
    row_manifest,
    validate_rows,
)


def test_injective_control_has_exactly_one_depth_preimage() -> None:
    instance = build_injective_abduction_instance(
        instance_id="injective",
        n_symbols=16,
        depth=6,
        seed=17,
    )

    assert exact_depth_preimages(instance.mapping, instance.target, instance.depth) == [instance.selected_start]


def test_multimodal_fan_has_exact_requested_preimages() -> None:
    instance = build_multimodal_abduction_instance(
        instance_id="abductive",
        n_symbols=20,
        depth=8,
        seed=23,
        solution_count=4,
    )

    assert exact_depth_preimages(instance.mapping, instance.target, instance.depth) == instance.valid_starts
    assert len(instance.valid_starts) == 4
    assert instance.selected_orbit[-1] == instance.target


def test_training_row_supervises_one_valid_reverse_chain_but_keeps_full_set() -> None:
    instance = build_multimodal_abduction_instance(
        instance_id="abductive",
        n_symbols=12,
        depth=4,
        seed=29,
        solution_count=3,
    )
    row = build_row(instance)

    assert row["completion"] in row["valid_starts"]
    assert row["coverage_denominator"] == 3
    assert row["loop_completions"][-1] == row["completion"]
    assert len(row["loop_completions"]) == row["depth"]
    assert set(row["valid_orbits"]) == set(row["valid_starts"])


def test_dataset_generation_is_deterministic_and_balances_solution_counts() -> None:
    config = AbductiveInjectiveConfig(
        n_symbols=16,
        max_depth=5,
        rows_per_depth=6,
        seed=31,
        min_solutions=2,
        max_solutions=4,
    )
    first = build_rows(config, split="test", mode="abductive")
    second = build_rows(config, split="test", mode="abductive")

    assert row_manifest(first) == row_manifest(second)
    assert set(row_manifest(first)["solution_counts"]) == {"2", "3", "4"}
    assert validate_rows(first, expected_mode="abductive")["status"] == "passed"


def test_validation_rejects_corrupted_solution_set() -> None:
    config = AbductiveInjectiveConfig(n_symbols=12, max_depth=2, rows_per_depth=1, seed=37)
    rows = build_rows(config, split="test", mode="abductive")
    rows[0]["valid_starts"] = rows[0]["valid_starts"][:-1]

    validation = validate_rows(rows, expected_mode="abductive")

    assert validation["status"] == "failed"
    assert any("exact preimages" in error for error in validation["errors"])

