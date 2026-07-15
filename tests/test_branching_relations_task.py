from training.branching_relations_task import (
    BranchingRelationsConfig,
    assess_validity_gate,
    build_rows,
    reachable_values,
    validate_rows,
)


def test_branching_rows_store_exact_reachable_sets_and_sampled_chains() -> None:
    rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=16, seed=101),
        split="calibration",
        rendering="symbolic",
        n_symbols=24,
    )
    validation = validate_rows(rows)
    assert validation["status"] == "passed", validation["errors"]
    assert validation["depth_counts"] == {"1": 16, "2": 16, "3": 16, "4": 16}
    for row in rows:
        mapping = {int(k): tuple(v) for k, v in row["successor_values"].items()}
        assert all(len(set(values)) == 2 for values in mapping.values())
        exact = reachable_values(mapping, int(row["start_value"]), int(row["depth"]))
        assert sorted(exact) == sorted(row["reachable_values"])
        chain = row["sampled_chain_values"]
        assert len(chain) == int(row["depth"]) + 1
        assert all(chain[index + 1] in mapping[chain[index]] for index in range(len(chain) - 1))


def test_reachable_size_strata_are_balanced_where_mathematically_feasible() -> None:
    rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=128, seed=202),
        split="test",
        rendering="verbal",
        n_symbols=20,
    )
    counts = validate_rows(rows)["stratum_counts_by_depth"]
    assert counts["1"] == {"2": 128}
    assert counts["2"] == {"2": 64, "3-4": 64}
    assert set(counts["3"]) == {"2", "3-4", "5-8"}
    assert max(counts["3"].values()) - min(counts["3"].values()) <= 1
    assert counts["4"] == {"2": 32, "3-4": 32, "5-8": 32, "9-16": 32}


def test_validity_gate_requires_pooled_and_every_depth() -> None:
    green = [{"depth": depth, "valid": index < 90} for depth in range(1, 5) for index in range(128)]
    assert assess_validity_gate(green)["passed"] is True
    red = [
        {"depth": depth, "valid": index < (60 if depth == 4 else 100)}
        for depth in range(1, 5)
        for index in range(128)
    ]
    verdict = assess_validity_gate(red)
    assert verdict["pooled_accuracy"] >= 0.70
    assert verdict["by_depth"]["4"]["accuracy"] < 0.55
    assert verdict["passed"] is False
