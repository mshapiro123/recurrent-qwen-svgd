from __future__ import annotations

import os

def test_candidate_drive_checkpoints_prefers_exact_run_dir(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    run_id = "stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231"
    exact = tmp_path / run_id / "run_dir" / "phase1" / "phase1_step_125.pt"
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"checkpoint")

    candidates = module.candidate_drive_checkpoints(run_id, "phase1_step_125.pt")

    assert candidates[0] == exact
    assert exact in candidates


def test_candidate_drive_checkpoints_finds_repo_output_shape(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    root_a = tmp_path / "recurrent-qwen-svgd-artifacts"
    root_b = tmp_path / "recurrent-qwen-svgd"
    monkeypatch.setenv("DRIVE_BACKUP_DIRS", os.pathsep.join([str(root_a), str(root_b)]))
    monkeypatch.delenv("DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    run_id = "stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231"
    exact = root_b / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_125.pt"
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"checkpoint")

    candidates = module.candidate_drive_checkpoints(run_id, "phase1_step_125.pt")

    assert exact in candidates
    assert candidates.index(exact) < 12


def test_candidate_drive_checkpoints_finds_arc_mix_arm_shape(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path / "drive"))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    run_id = "stage5_arc_agi_next_action_20260622_145746_plan_arc_mix_probe"
    exact = (
        tmp_path
        / "drive"
        / run_id
        / "run_dir"
        / "arc_mix_response_w01_lr2e6"
        / "phase1"
        / "phase1_step_50.pt"
    )
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"checkpoint")

    candidates = module.candidate_drive_checkpoints(run_id, "phase1_step_50.pt")

    assert exact in candidates


def test_restore_checkpoint_copies_from_drive(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path / "drive"))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run_id = "run"
    source = tmp_path / "drive" / run_id / "run_dir" / "phase1" / "phase1_step_125.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    target = tmp_path / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_125.pt"

    module.restore_checkpoint_if_needed(target, run_id=run_id)

    assert target.read_bytes() == b"checkpoint"


def test_restore_checkpoint_copies_from_repo_output_drive_shape(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    drive_root = tmp_path / "drive-project"
    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(drive_root))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run_id = "run"
    source = drive_root / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_125.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    target = tmp_path / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_125.pt"

    module.restore_checkpoint_if_needed(target, run_id=run_id)

    assert target.read_bytes() == b"checkpoint"


def test_restore_checkpoint_failure_mentions_drive_reauth(monkeypatch, tmp_path) -> None:
    import pytest
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path / "missing-drive"))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    target = tmp_path / "outputs" / "stage5" / "run" / "phase1" / "phase1_step_125.pt"

    with pytest.raises(FileNotFoundError) as exc:
        module.restore_checkpoint_if_needed(target, run_id="run")

    message = str(exc.value)
    assert "Drive visibility:" in message
    assert "drive.mount('/content/drive', force_remount=True)" in message
    assert "FORCE_DRIVE_REMOUNT=1" in message


def test_default_run_id_uses_full_label() -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    run_id = module.default_run_id("all")

    assert "arcfull" in run_id
