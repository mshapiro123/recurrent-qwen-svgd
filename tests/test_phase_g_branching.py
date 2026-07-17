from __future__ import annotations

from eval.phase_g_branching import (
    exact_branching_coverage,
    solve_global_temperature,
    summarize_coverage_rows,
)


def test_exact_branching_coverage_uses_distinct_valid_predictions() -> None:
    result = exact_branching_coverage(["A", "A", "B", "X"], ["A", "B", "C"])

    assert result["valid_samples"] == 3
    assert result["unique_valid"] == ["A", "B"]
    assert result["coverage"] == 2 / 3
    assert result["full_coverage"] is False
    assert result["duplicate_rate"] == 0.25


def test_entropy_temperature_solver_hits_global_target() -> None:
    result = solve_global_temperature(
        [{"A": 3.0, "B": 0.0}, {"A": 1.0, "B": 0.0}],
        target_mean_entropy=0.3,
    )

    assert result["temperature"] > 0.0
    assert abs(result["achieved_mean_entropy"] - 0.3) < 1e-8


def test_coverage_summary_stratifies_identical_rows() -> None:
    rows = [
        {
            "depth": 1,
            "reachable_set_stratum": "2",
            **exact_branching_coverage(["A"], ["A", "B"]),
        },
        {
            "depth": 2,
            "reachable_set_stratum": "3-4",
            **exact_branching_coverage(["A", "B"], ["A", "B"]),
        },
    ]
    summary = summarize_coverage_rows(rows)

    assert summary["overall"]["rows"] == 2
    assert summary["overall"]["mean_coverage"] == 0.75
    assert summary["by_depth"]["1"]["mean_coverage"] == 0.5
