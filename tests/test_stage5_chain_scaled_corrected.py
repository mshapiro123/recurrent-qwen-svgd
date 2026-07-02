from __future__ import annotations

import json
from pathlib import Path

import colab.run_stage5_chain_scaled_corrected as runner


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_limit_rows_per_depth_keeps_first_n_per_depth(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "rows.jsonl"
    dest = tmp_path / "limited.jsonl"
    write_rows(
        source,
        [
            {"id": "d1a", "depth": 1},
            {"id": "d1b", "depth": 1},
            {"id": "d2a", "depth": 2},
            {"id": "d2b", "depth": 2},
            {"id": "d2c", "depth": 2},
        ],
    )

    result = runner.limit_rows_per_depth(source, dest, rows_per_depth=2)
    kept = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]

    assert [row["id"] for row in kept] == ["d1a", "d1b", "d2a", "d2b"]
    assert result["rows"] == 4
    assert result["depth_counts"] == {"1": 2, "2": 2}


def test_filter_depth_jsonl_keeps_requested_depth_prefix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "rows.jsonl"
    dest = tmp_path / "filtered.jsonl"
    write_rows(
        source,
        [
            {"id": "d1", "synthetic_depth": 1},
            {"id": "d2", "synthetic_depth": 2},
            {"id": "d3", "synthetic_depth": 3},
        ],
    )

    result = runner.filter_depth_jsonl(source, dest, max_depth=2)
    kept = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]

    assert [row["id"] for row in kept] == ["d1", "d2"]
    assert result["rows"] == 2
    assert result["depth_counts"] == {"1": 1, "2": 1}


def test_active_diag_min_reports_lowest_depth_accuracy() -> None:
    summary = {"active_diagonal": {"1": 1.0, "2": 0.75, "3": 0.5}}

    assert runner.active_diag(summary) == {"1": 1.0, "2": 0.75, "3": 0.5}
    assert runner.active_diag_min(summary) == 0.5


def test_matrix_total_hits_sums_all_cells() -> None:
    matrix = {
        "matrix": {
            "1": {"1": {"correct": 3, "total": 4}},
            "2": {
                "1": {"correct": 1, "total": 4},
                "2": {"correct": 2, "total": 4},
            },
        }
    }

    assert runner.matrix_total_hits(matrix) == {"correct": 6, "total": 12}
