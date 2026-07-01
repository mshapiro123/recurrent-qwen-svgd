from __future__ import annotations

import json
from pathlib import Path

from colab.summarize_synthetic_depth_primitive_curve import summarize_curve


def write_run(tmp_path: Path, *, run_id: str, n_symbols: int, base_acc: float, recurrent_acc: float) -> Path:
    run_dir = tmp_path / run_id
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "summary.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_depth_dataset",
                "config": {
                    "n_symbols": n_symbols,
                    "max_depth": 1,
                    "rows_per_depth": 100,
                    "seed": 1,
                    "num_choices": 4,
                    "max_target_loops": 1,
                    "value_prefix": "",
                },
            }
        ),
        encoding="utf-8",
    )
    total = 100
    summary = {
        "run_id": run_id,
        "train_format": "mcq_option_text",
        "checkpoint": f"outputs/stage5/{run_id}/train/unfrozen/unfrozen_recurrent_step_500.pt",
        "dataset_summary": f"outputs/stage5/{run_id}/data/summary.json",
        "base_matrix": {
            "matrix": {
                "1": {
                    "0": {
                        "correct": int(base_acc * total),
                        "total": total,
                        "accuracy": base_acc,
                    }
                }
            }
        },
        "matrix": {
            "matrix": {
                "1": {
                    "1": {
                        "correct": int(recurrent_acc * total),
                        "total": total,
                        "accuracy": recurrent_acc,
                    }
                }
            }
        },
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_summarize_curve_recommends_largest_n_over_bar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    paths = [
        write_run(tmp_path, run_id="curve_N8", n_symbols=8, base_acc=0.24, recurrent_acc=0.99),
        write_run(tmp_path, run_id="curve_N12", n_symbols=12, base_acc=0.23, recurrent_acc=0.74),
        write_run(tmp_path, run_id="curve_N16", n_symbols=16, base_acc=0.22, recurrent_acc=0.62),
    ]

    summary = summarize_curve(paths, primitive_bar=0.71, strong_bar=0.9)

    assert [row["n_symbols"] for row in summary["runs"]] == [8, 12, 16]
    assert summary["largest_n_clearing_primitive_bar"] == 12
    assert summary["largest_n_clearing_strong_bar"] == 8
    assert summary["recommended_phase2_n_symbols"] == 12
    assert summary["all_runs_clear_primitive_bar"] is False
