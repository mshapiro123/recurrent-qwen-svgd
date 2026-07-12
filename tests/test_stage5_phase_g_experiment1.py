from __future__ import annotations

from pathlib import Path

import yaml

from colab.run_stage5_phase_g_experiment1 import (
    assess_task,
    restore_stage_checkpoint,
    subset_by_depth,
    write_arm_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_subset_by_depth_takes_equal_prefix_per_depth() -> None:
    rows = [
        {"id": f"d{depth}-{index}", "depth": depth}
        for depth in (1, 2, 3)
        for index in range(5)
    ]

    subset = subset_by_depth(rows, 2)

    assert len(subset) == 6
    assert [row["id"] for row in subset] == ["d1-0", "d1-1", "d2-0", "d2-1", "d3-0", "d3-1"]


def test_task_gate_requires_pooled_and_every_depth() -> None:
    summary = {
        "overall": {"greedy_valid_rate": 0.95},
        "by_depth": {
            "1": {"greedy_valid_rate": 1.0},
            "2": {"greedy_valid_rate": 0.79},
        },
    }

    result = assess_task(summary, pooled_floor=0.90, depth_floor=0.80)

    assert result["passed"] is False
    assert result["min_depth_greedy_valid_rate"] == 0.79


def test_arm_config_locks_seed_and_disables_stochastic_modules(tmp_path) -> None:
    keeper = tmp_path / "keeper.pt"
    keeper.write_bytes(b"checkpoint")
    config_path = write_arm_config(
        tmp_path,
        arm="injective_control",
        keeper=keeper,
        max_steps=1000,
        seed=81001,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["seed"] == 81001
    assert config["max_steps"] == 1000
    assert config["loop_loss_mode"] == "per_loop_labels"
    assert config["train_auxiliary"]["latent"] is False
    assert config["train_auxiliary"]["halting"] is False
    assert config["recurrence_curriculum"]["enabled"] is False


def test_restore_stage_checkpoint_rejects_wrong_sha(tmp_path, monkeypatch) -> None:
    restored = tmp_path / "restored.pt"
    restored.write_bytes(b"restored")

    def fake_restore(*args, **kwargs):
        return restored, {"selected_checkpoint_sha256": "actual"}

    monkeypatch.setattr("colab.run_stage5_phase_g_experiment1.restore_checkpoint", fake_restore)

    try:
        restore_stage_checkpoint(
            tmp_path,
            {"checkpoint": "source.pt", "checkpoint_sha256": "expected"},
            arm="injective_control",
        )
    except RuntimeError as exc:
        assert "SHA mismatch" in str(exc)
    else:
        raise AssertionError("Expected a checkpoint SHA mismatch")


def test_bootstrap_exposes_phase_g_experiment1_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"phase_g_experiment1"' in text
    assert "STAGE5_PHASE_G_EXPERIMENT1_CELL_VERSION" in text
    assert "colab/run_stage5_phase_g_experiment1.py" in text
