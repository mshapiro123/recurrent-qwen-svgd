from __future__ import annotations

from pathlib import Path

import yaml

from colab.run_stage5_inverse_rendered_continuation import (
    EFFECTIVE_BATCH_SIZE,
    MAX_LOOPS,
    MAX_STEPS,
    loop_weights,
    prepare_training_rows,
    write_config,
)


def test_continuation_training_rows_are_inverse_rendered_and_disjoint(tmp_path: Path, monkeypatch) -> None:
    import colab.run_stage5_inverse_rendered_continuation as runner

    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    for split in ("calibration", "test"):
        (frozen_root / f"{split}_n24.jsonl").write_text(
            '{"id":"phase_g_' + split + '_unique_d01_00000"}\n', encoding="utf-8"
        )
    monkeypatch.setattr(runner, "DATA_ROOT", frozen_root)

    receipt = prepare_training_rows(tmp_path / "run", rows_per_stratum=2)

    assert receipt["validation"]["status"] == "passed"
    assert receipt["rendering"] == "inverse_relation_given"
    assert receipt["forward_rehearsal_fraction"] == 0.0
    assert receipt["frozen_eval_id_overlap"] == 0


def test_continuation_loop_weights_equalize_exposure_with_newest_emphasis() -> None:
    weights = loop_weights(
        [{"depth": 1}, {"depth": 2}, {"depth": 3}, {"depth": 4}] * 3
    )

    assert len(weights) == MAX_LOOPS
    assert all(value > 0.0 for value in weights)
    assert weights[-1] > weights[0]


def test_continuation_config_locks_deterministic_objective(tmp_path: Path) -> None:
    config_path = tmp_path / "continuation.yaml"
    write_config(
        config_path,
        checkpoint=tmp_path / "source.pt",
        output_dir=tmp_path / "train",
        checkpoint_backup_dir=tmp_path / "drive" / "checkpoints",
        progress_backup_path=tmp_path / "drive" / "progress.json",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["max_loops"] == MAX_LOOPS
    assert config["max_steps"] == MAX_STEPS
    assert config["gradient_accumulation_steps"] == EFFECTIVE_BATCH_SIZE
    assert config["optimizer"] == "adamw"
    assert config["beta"] == 0.0
    assert config["halt_target_nll_weight"] == 0.0
    assert config["loop_control_ce_weight"] == 0.0
    assert config["train_auxiliary"]["latent"] is False
    assert config["train_auxiliary"]["halting"] is False


def test_continuation_target_is_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_INVERSE_RENDERED_CONTINUATION_CELL.py").read_text(encoding="utf-8")
    runner = Path("colab/run_stage5_inverse_rendered_continuation.py").read_text(encoding="utf-8")

    assert '"inverse_rendered_n24_continuation"' in bootstrap
    assert "STAGE5_INVERSE_RENDERED_CONTINUATION_CELL_VERSION" in cell
    assert "tests/test_stage5_inverse_rendered_continuation.py" in cell
    assert "forward_rehearsal_fraction" in runner
    assert "test_summary is not None" in runner
    assert "phase_g_alpha_status" in runner
