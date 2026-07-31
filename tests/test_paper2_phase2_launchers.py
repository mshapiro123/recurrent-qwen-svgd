from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_exposes_all_prewindow_targets() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    for target in (
        "paper2_phase2_oracle_overlap",
        "paper2_phase2_v1_v2",
        "paper2_phase2_eval_de_freeze",
    ):
        assert f'"{target}"' in bootstrap


def test_prewindow_cells_keep_training_closed() -> None:
    expectations = {
        "STAGE5_PAPER2_PHASE2_ORACLE_OVERLAP_CELL.py": (
            "no model no scoring no training",
            "read-once scoring remains spent",
        ),
        "STAGE5_PAPER2_PHASE2_V1_V2_CELL.py": (
            "no optimizer no training",
            "not a certified Lipschitz upper bound",
        ),
        "STAGE5_PAPER2_PHASE2_EVAL_DE_FREEZE_CELL.py": (
            "no optimizer no training",
            "read-once scoring remains unspent",
        ),
    }
    for filename, markers in expectations.items():
        text = (ROOT / "colab" / filename).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text


def test_eval_de_runner_requires_all_prior_frozen_partitions() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_eval_de_freeze.py").read_text(
        encoding="utf-8"
    )
    assert "private/eval_b/eval_b.jsonl" in runner
    assert "private/dev_c/dev_c.jsonl" in runner
    assert "private/eval_c/eval_c.jsonl" in runner
    assert "scores_exposed" in runner
    assert "read_once_scoring_spent" in runner
