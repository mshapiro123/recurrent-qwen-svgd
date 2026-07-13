from __future__ import annotations

from training.abductive_injective_task import (
    AbductiveInjectiveConfig,
    PhaseGFrozenEvalConfig,
    build_injective_abduction_instance,
    build_multimodal_abduction_instance,
    build_phase_g_frozen_rows,
    build_row,
    build_rows,
    build_stratified_random_abduction_instance,
    exact_depth_preimages,
    preimage_stratum,
    row_manifest,
    validate_phase_g_frozen_rows,
    validate_rows,
    with_inverse_table_prompt,
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

    assert row["completion"].strip() in row["valid_starts"]
    assert row["coverage_denominator"] == 3
    assert row["loop_completions"][-1] == row["completion"]
    assert not row["prompt"].endswith(" ")
    assert row["completion"].startswith(" ")
    assert all(completion.startswith(" ") for completion in row["loop_completions"])
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


def test_arbitrary_random_mapping_hits_each_locked_preimage_stratum() -> None:
    for index, stratum in enumerate(("unique", "small", "large")):
        instance = build_stratified_random_abduction_instance(
            instance_id=stratum,
            n_symbols=24,
            depth=3,
            seed=100 + index,
            stratum=stratum,
        )

        exact = exact_depth_preimages(instance.mapping, instance.target, instance.depth)
        assert preimage_stratum(len(exact)) == stratum
        assert len(set(instance.mapping.values())) < 24
        assert instance.selected_start in exact
        assert instance.generator_kind == "arbitrary_random_mapping_conditioned_on_preimage_stratum"


def test_phase_g_frozen_rows_are_balanced_exact_and_split_disjoint() -> None:
    config = PhaseGFrozenEvalConfig(
        n_symbols=24,
        depths=(1, 2, 3, 4),
        rows_per_stratum=8,
        seed=7194203,
    )
    calibration = build_phase_g_frozen_rows(config, split="calibration")
    test = build_phase_g_frozen_rows(config, split="test")

    validation = validate_phase_g_frozen_rows(calibration, rows_per_stratum=8)
    assert validation["status"] == "passed"
    assert validation["stratum_counts"] == {"unique": 8, "small": 8, "large": 8}
    assert {row["id"] for row in calibration}.isdisjoint({row["id"] for row in test})
    assert all(row["posterior_chain_sampling"] == "uniform_over_exact_valid_preimages" for row in calibration)


def test_inverse_table_prompt_preserves_targets_and_makes_each_step_forward_lookup() -> None:
    config = AbductiveInjectiveConfig(n_symbols=8, max_depth=3, rows_per_depth=1, seed=71)
    row = next(item for item in build_rows(config, split="train", mode="injective") if item["depth"] == 3)

    inverse_row = with_inverse_table_prompt(row)

    assert inverse_row["table_direction"] == "inverse_given"
    assert inverse_row["completion"] == row["completion"]
    assert inverse_row["loop_completions"] == row["loop_completions"]
    assert "Starting with" in inverse_row["question"]
    assert "reverse handoffs" in inverse_row["question"]

    inverse_mapping = inverse_row["display_mapping"]
    current = row["observed_target"]
    observed_chain = []
    for _ in range(row["depth"]):
        current = inverse_mapping[current]
        observed_chain.append(current)
    assert observed_chain == [value.strip() for value in row["loop_completions"]]
