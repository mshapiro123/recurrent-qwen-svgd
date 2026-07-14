from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from colab.run_stage5_inverse_table_rehearsal import (
    _write_config,
    build_rehearsal_mix,
    fixed_schedule_dose,
    rehearsal_optimizer_steps,
    rehearsal_weight_profiles,
)


def test_rehearsal_config_uses_interval_checkpoints_and_drive_receipts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        checkpoint=tmp_path / "init.pt",
        output_dir=tmp_path / "train",
        max_steps=334,
        seed=17,
        checkpoint_backup_dir=tmp_path / "drive" / "checkpoints",
        progress_backup_path=tmp_path / "drive" / "progress.json",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["save_every"] == 25
    assert config["checkpoint_backup_every"] == 100
    assert Path(config["checkpoint_backup_dir"]) == tmp_path / "drive" / "checkpoints"
    assert Path(config["progress_backup_path"]) == tmp_path / "drive" / "progress.json"


def test_rehearsal_mix_adds_rows_without_reducing_original_task_dose() -> None:
    task_rows = [{"id": f"task-{index}", "depth": 1 + index % 3} for index in range(12)]
    rehearsal_rows = [{"id": f"rehearsal-{index}", "depth": 1 + index % 12} for index in range(24)]
    steps = rehearsal_optimizer_steps(
        baseline_steps=250,
        effective_batch_size=8,
        rehearsal_fraction=0.25,
    )
    mixed, receipt = build_rehearsal_mix(
        task_rows,
        rehearsal_rows,
        optimizer_steps=steps,
        effective_batch_size=8,
        rehearsal_fraction=0.25,
        seed=17,
    )

    assert steps == 334
    assert receipt["task_rows"] >= 250 * 8
    assert receipt["rehearsal_rows"] == len(mixed) - receipt["task_rows"]
    assert abs(receipt["realized_rehearsal_fraction"] - 0.25) < 0.001


def test_rehearsal_weight_profiles_preserve_task_weights_and_bound_rehearsal_scale() -> None:
    profiles = rehearsal_weight_profiles(
        task_weights=[0.35, 0.53, 2.12],
        rehearsal_depths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        max_loops=12,
    )

    assert profiles["task"][:3] == [0.35, 0.53, 2.12]
    assert profiles["task"][3:] == [0.0] * 9
    assert abs(sum(profiles["rehearsal"]) - sum(profiles["task"])) < 1e-6
    assert profiles["rehearsal"][-1] > profiles["rehearsal"][0]


def test_inverse_table_rehearsal_target_is_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_INVERSE_TABLE_REHEARSAL_CELL.py").read_text(encoding="utf-8")
    runner = Path("colab/run_stage5_inverse_table_rehearsal.py").read_text(encoding="utf-8")

    assert '"inverse_table_cap3_rehearsal"' in bootstrap
    assert "STAGE5_INVERSE_TABLE_REHEARSAL_CELL_VERSION" in cell
    assert "row_specific_forward_loops" in cell
    assert "accepted_returncodes={0, 2}" in cell
    assert 'os.environ.get("STAGE5_BOOTSTRAP_REF", "main")' in cell
    assert "Pinned checkout verified" in cell
    assert runner.index("sys.path.insert(0, str(REPO_ROOT))") < runner.index("from colab.")


def test_inverse_table_rehearsal_runner_resolves_repo_packages_when_run_by_path(tmp_path: Path) -> None:
    env = {**os.environ, "STAGE5_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "colab/run_stage5_inverse_table_rehearsal.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stdout
    assert "FileNotFoundError" in result.stdout


def test_fixed_schedule_dose_reports_source_by_loop() -> None:
    rows = [
        {
            "training_source": "task",
            "depth": 2,
            "loop_label_weights": [1.0, 2.0, 0.0],
        },
        {
            "training_source": "rehearsal",
            "depth": 3,
            "loop_label_weights": [0.5, 0.5, 1.0],
        },
    ]

    receipt = fixed_schedule_dose(rows, max_loops=3)

    assert receipt["task"]["weighted_active_labels_by_loop"] == [1.0, 2.0, 0.0]
    assert receipt["rehearsal"]["weighted_active_labels_by_loop"] == [0.5, 0.5, 1.0]
