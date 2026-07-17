from __future__ import annotations

from pathlib import Path

from colab.stage5_path_utils import repo_relative_text, resolve_repo_path


def test_accepts_relative_training_checkpoint_paths(tmp_path: Path) -> None:
    relative = "outputs/stage5/run/train/phase_g_raw_step_1000.pt"

    assert resolve_repo_path(tmp_path, relative) == tmp_path / relative
    assert repo_relative_text(tmp_path, relative) == relative


def test_normalizes_absolute_training_checkpoint_paths(tmp_path: Path) -> None:
    relative = "outputs/stage5/run/train/phase_g_ema_step_1000.pt"
    absolute = resolve_repo_path(tmp_path, relative)

    assert repo_relative_text(tmp_path, absolute) == relative
