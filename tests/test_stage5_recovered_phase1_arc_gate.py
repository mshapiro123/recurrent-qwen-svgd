from __future__ import annotations

def test_candidate_drive_checkpoints_prefers_exact_run_dir(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path))
    run_id = "stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231"
    exact = tmp_path / run_id / "run_dir" / "phase1" / "phase1_step_125.pt"
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"checkpoint")

    candidates = module.candidate_drive_checkpoints(run_id, "phase1_step_125.pt")

    assert candidates[0] == exact
    assert exact in candidates


def test_restore_checkpoint_copies_from_drive(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path / "drive"))
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run_id = "run"
    source = tmp_path / "drive" / run_id / "run_dir" / "phase1" / "phase1_step_125.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    target = tmp_path / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_125.pt"

    module.restore_checkpoint_if_needed(target, run_id=run_id)

    assert target.read_bytes() == b"checkpoint"


def test_restore_checkpoint_failure_mentions_drive_reauth(monkeypatch, tmp_path) -> None:
    import pytest
    import colab.run_stage5_recovered_phase1_arc_gate as module

    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(tmp_path / "missing-drive"))
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
