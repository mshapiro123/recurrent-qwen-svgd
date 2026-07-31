from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v1b_bootstrap_target_is_separate_and_safety_marked() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_v1b",
        "STAGE5_PAPER2_PHASE2_V1B_CELL.py",
        "paper2_phase2_v1b_v1",
        "DEV-only finite perturbation causal check no optimizer no training",
        "teacher flip pair crossing and collateral damage reported separately",
    ):
        assert marker in bootstrap


def test_v1b_launcher_requires_completed_v1_and_no_optimizer() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_v1b.py").read_text(
        encoding="utf-8"
    )
    assert "v1_v2/summary.json" in runner
    assert 'v1_v2["status"] == "complete_no_training_dev_only"' in runner
    assert "--sample_size" in runner
    assert 'STAGE5_PHASE2_V1B_SAMPLE_SIZE", "2000"' in runner
    assert "torch.optim" not in runner
    assert "training/train_" not in runner
