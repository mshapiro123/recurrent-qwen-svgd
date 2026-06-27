from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.eval_tail_damper_depth_sweep import fixed_damper_components
from eval.eval_tail_damper_depth_sweep import write_markdown


def test_tail_damper_sweep_markdown_reports_tradeoff(tmp_path: Path) -> None:
    summary = {
        "run_id": "tail_damper_test",
        "checkpoint": "ckpt.pt",
        "examples": 10,
        "score_loops": [1, 2, 3],
        "tail_loop_counts": [1, 2, 3, 4, 8],
        "reentry_rescale_mode": "none",
        "calibration": {
            "correction_class": {"action": "tail_damper"},
            "tail_decomposition_loop1": {
                "tail_mismatch": 2.0,
                "after_damper": 0.4,
                "damper_scale": [0.5, 0.75],
            },
        },
        "strength_summaries": [
            {
                "strength": 0.5,
                "tail_trace": {"loop8": {"ratio_vs_entry": 4.2}},
                "score_summary": {
                    "examples": 10,
                    "loop_results": {
                        "1": {"correct": 3},
                        "2": {"correct": 4},
                        "3": {"correct": 5},
                    },
                    "oracle_correct": 6,
                    "oracle_gap_vs_loop1": 3,
                    "rescued_vs_loop1": 4,
                    "harmed_vs_loop1": 1,
                },
            }
        ],
    }
    out = tmp_path / "summary.md"

    write_markdown(summary, out)
    text = out.read_text(encoding="utf-8")

    assert "loop8 tail ratio" in text
    assert "0.50" in text
    assert "6/10" in text
    assert "rescued" in text


def test_fixed_damper_components_normalize_covariance_dtype() -> None:
    mean, basis, decomp = fixed_damper_components(
        {
            "mean": torch.zeros(4, dtype=torch.float32),
            "basis": torch.eye(4, 2, dtype=torch.float32),
            "damper_scale": torch.tensor([0.5, 0.25], dtype=torch.float32),
        }
    )

    assert mean.dtype == torch.float64
    assert basis.dtype == torch.float64
    assert decomp["after_rotation"] == 0.0
    assert decomp["after_rotation_then_damper"] == 0.0
