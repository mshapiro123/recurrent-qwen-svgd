from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.eval_reentry_tail_diagnostic import (
    correction_class,
    forced_depth_patterns,
    group_summary,
    tail_decomposition,
)


def rotation(theta: float, dim: int = 4) -> torch.Tensor:
    mat = torch.eye(dim, dtype=torch.double)
    c = torch.cos(torch.tensor(theta, dtype=torch.double))
    s = torch.sin(torch.tensor(theta, dtype=torch.double))
    mat[1, 1] = c
    mat[1, 2] = -s
    mat[2, 1] = s
    mat[2, 2] = c
    return mat


def test_tail_decomposition_detects_tail_inflation() -> None:
    entry = torch.diag(torch.tensor([1000.0, 4.0, 2.0, 1.0], dtype=torch.double))
    exit_cov = torch.diag(torch.tensor([1000.0, 16.0, 8.0, 4.0], dtype=torch.double))

    result = tail_decomposition(entry, exit_cov, n_tail=3)
    rec = correction_class(result)

    assert result["tail_mismatch"] > 2.0
    assert result["after_damper"] < 1e-6
    assert rec["action"] == "tail_damper"


def test_tail_decomposition_detects_tail_rotation() -> None:
    entry = torch.diag(torch.tensor([1000.0, 4.0, 2.0, 1.0], dtype=torch.double))
    rot = rotation(0.8, dim=4)
    exit_cov = rot.T @ entry @ rot

    result = tail_decomposition(entry, exit_cov, n_tail=3)
    rec = correction_class(result)

    assert result["tail_mismatch"] > 0.1
    assert result["after_rotation"] < 1e-6
    assert rec["action"] == "tail_rotation"


def test_forced_depth_patterns_classify_harmed_and_rescued(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    summary = root / "summary.json"
    summary.write_text(
        json.dumps({"loop_run_ids": ["run_loop1", "run_loop2", "run_loop3"], "loops": [1, 2, 3]}),
        encoding="utf-8",
    )
    for run_id, hits in {
        "run_loop1": {"h": True, "r": False, "s": True, "w": False},
        "run_loop2": {"h": False, "r": True, "s": True, "w": False},
        "run_loop3": {"h": True, "r": True, "s": True, "w": False},
    }.items():
        out = root / "outputs" / "stage5" / run_id
        out.mkdir(parents=True)
        out.joinpath("arc_challenge_recurrent_content_question_only.jsonl").write_text(
            "".join(
                json.dumps({"id": key, "hit": value, "answer": "A", "prediction": "A"}) + "\n"
                for key, value in hits.items()
            ),
            encoding="utf-8",
        )
    import eval.eval_reentry_tail_diagnostic as module

    monkeypatch.setattr(module, "ROOT", root)
    patterns = forced_depth_patterns(summary, benchmark="arc_challenge", score_target="content_question_only")

    assert patterns["h"]["pattern"] == "101"
    assert patterns["h"]["group"] == "harmed"
    assert patterns["h"]["tipping_loop"] == 2
    assert patterns["r"]["pattern"] == "011"
    assert patterns["r"]["group"] == "rescued"
    assert patterns["r"]["tipping_loop"] == 2
    assert patterns["s"]["group"] == "stable_correct"
    assert patterns["w"]["group"] == "stable_wrong"


def test_group_summary_reports_harmed_minus_rescued_delta() -> None:
    rows = [
        {"group": "harmed", "pattern": "100", "tipping_tail_energy_ratio": 3.0, "loop_tail_energy_ratio": {"2": 3.0}},
        {"group": "rescued", "pattern": "011", "tipping_tail_energy_ratio": 1.5, "loop_tail_energy_ratio": {"2": 1.5}},
    ]

    summary = group_summary(rows, [2])

    assert summary["harmed"]["n"] == 1
    assert summary["rescued"]["n"] == 1
    assert summary["harmed_minus_rescued"]["mean_tipping_tail_energy_ratio_delta"] == 1.5
