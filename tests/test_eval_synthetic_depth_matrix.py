from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eval_synthetic_depth_matrix import (
    MatrixCell,
    build_accuracy_matrix,
    largest_depth_at_threshold,
    summarize_matrix,
)


def test_accuracy_matrix_groups_by_depth_and_loop() -> None:
    rows = [
        {"depth": 1, "forced_loop_count": 1, "hit": True},
        {"depth": 1, "forced_loop_count": 1, "hit": False},
        {"depth": 2, "forced_loop_count": 1, "hit": False},
        {"depth": 2, "forced_loop_count": 2, "hit": True},
    ]

    matrix = build_accuracy_matrix(rows)

    assert matrix[(1, 1)] == MatrixCell(correct=1, total=2)
    assert matrix[(2, 1)] == MatrixCell(correct=0, total=1)
    assert matrix[(2, 2)] == MatrixCell(correct=1, total=1)


def test_largest_depth_at_threshold_reports_frontier() -> None:
    matrix = {
        (1, 1): MatrixCell(correct=10, total=10),
        (2, 1): MatrixCell(correct=8, total=10),
        (3, 1): MatrixCell(correct=4, total=10),
        (1, 2): MatrixCell(correct=10, total=10),
        (2, 2): MatrixCell(correct=10, total=10),
        (3, 2): MatrixCell(correct=8, total=10),
    }

    assert largest_depth_at_threshold(matrix, loop=1, threshold=0.75) == 2
    assert largest_depth_at_threshold(matrix, loop=2, threshold=0.75) == 3
    assert largest_depth_at_threshold(matrix, loop=3, threshold=0.75) == 0


def test_summarize_matrix_marks_staircase_when_frontier_grows(tmp_path: Path) -> None:
    rows = []
    for depth in range(1, 5):
        for loop in (1, 2):
            for idx in range(4):
                rows.append(
                    {
                        "id": f"d{depth}_k{loop}_{idx}",
                        "depth": depth,
                        "forced_loop_count": loop,
                        "hit": depth <= loop * 2,
                    }
                )

    summary = summarize_matrix(rows, threshold=0.75)

    assert summary["depths"] == [1, 2, 3, 4]
    assert summary["loops"] == [1, 2]
    assert summary["frontier_by_loop"] == {"1": 2, "2": 4}
    assert summary["frontier_is_non_decreasing"] is True
    assert summary["frontier_strictly_expands"] is True
    assert summary["matrix"]["4"]["1"]["accuracy"] == pytest.approx(0.0)
    assert summary["matrix"]["4"]["2"]["accuracy"] == pytest.approx(1.0)

    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary), encoding="utf-8")
    assert json.loads(out.read_text())["frontier_strictly_expands"] is True
