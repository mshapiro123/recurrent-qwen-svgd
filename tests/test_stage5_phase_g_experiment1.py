from __future__ import annotations

from pathlib import Path

import yaml

from colab.run_stage5_phase_g_experiment1 import (
    assess_task,
    resolve_training_initialization,
    restore_stage_checkpoint,
    subset_by_depth,
    write_arm_config,
)
from training.abductive_injective_task import AbductiveInjectiveConfig, build_rows, row_manifest


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
    assert config["require_active_supervision"] is True
    assert config["require_nonzero_train_gradient"] is True


def test_arm_config_supports_explicit_recurrence_curriculum(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_CURRICULUM_ENABLED", "1")
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_CURRICULUM_START", "2")
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_CURRICULUM_END", "8")
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_CURRICULUM_RAMP_COMPUTE", "1")

    path = write_arm_config(
        tmp_path,
        arm="injective_curriculum_recovery",
        keeper=tmp_path / "parent.pt",
        max_steps=2000,
        seed=81001,
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["resume_from"].endswith("parent.pt")
    assert config["recurrence_curriculum"] == {
        "enabled": True,
        "start_loop": 2,
        "end_loop": 8,
        "schedule": "linear",
        "target_source": "row_capped",
        "ramp_compute": True,
    }


def test_training_initialization_requires_matching_override_sha(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent-checkpoint")
    expected_sha = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_INIT_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("STAGE5_PHASE_G_EXP1_INIT_SHA256", expected_sha)

    resolved, receipt = resolve_training_initialization(
        tmp_path / "run",
        default_checkpoint=tmp_path / "keeper.pt",
    )

    assert resolved.read_bytes() == checkpoint.read_bytes()
    assert receipt["kind"] == "explicit_continuation"
    assert receipt["checkpoint_sha256"] == expected_sha


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
    assert '"phase_g_injective_curriculum_recovery"' in text
    assert "STAGE5_PHASE_G_EXPERIMENT1_CELL_VERSION" in text
    assert "colab/run_stage5_phase_g_experiment1.py" in text
    assert '"STAGE5_PHASE_G_EXP1_CURRICULUM_START": "2"' in text
    assert '"STAGE5_PHASE_G_EXP1_CURRICULUM_END": "8"' in text
    assert '"STAGE5_PHASE_G_EXP1_GATE_SAMPLE_COUNTS": "1"' in text


def test_experiment1_dataset_hashes_remain_resume_compatible() -> None:
    expected = {
        "train_injective": "4ab6377a15d64cf5e07c8855ed05f432feed75e512e196cbd53f648dc9fcb4a5",
        "train_abductive": "22bfc8258a7ae6f0433b199dff6531b82afaa9577704ddfdca339d7f5c203a6a",
        "test_injective": "4dd29d9fb7b4170390234646c7c1773377eea56145f6ae659e38f3ae443f2068",
        "test_abductive": "db80c1c0a07264b0cd99aef5db302dd8184d5181f01e03edc2f722f34e356965",
    }
    for split, rows_per_depth in (("train", 256), ("test", 128)):
        for mode in ("injective", "abductive"):
            rows = build_rows(
                AbductiveInjectiveConfig(
                    n_symbols=20,
                    max_depth=8,
                    rows_per_depth=rows_per_depth,
                    seed=1_104_729,
                    min_solutions=2,
                    max_solutions=4,
                ),
                split=split,
                mode=mode,
            )
            assert row_manifest(rows)["row_sha256"] == expected[f"{split}_{mode}"]
