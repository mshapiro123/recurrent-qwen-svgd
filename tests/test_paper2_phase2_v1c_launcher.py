from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v1c_bootstrap_target_is_separate_and_safety_marked() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_v1c",
        "STAGE5_PAPER2_PHASE2_V1C_CELL.py",
        "paper2_phase2_v1c_v1",
        "c 0.075 0.10 0.15 matched neutral no optimizer no training",
        "same V1b sample seed and cohorts paired radius extension",
    ):
        assert marker in bootstrap


def test_v1c_launcher_locks_radius_set_and_requires_v1b() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_v1c.py").read_text(
        encoding="utf-8"
    )
    assert '"0.075,0.10,0.15"' in runner
    assert "v1b/summary.json" in runner
    assert "v1c_radius_v1" in runner
    assert "torch.optim" not in runner
    assert "training/train_" not in runner
