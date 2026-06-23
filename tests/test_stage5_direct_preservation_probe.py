from pathlib import Path

import pytest


def test_restore_checkpoint_prefers_matching_stage5_run(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_direct_preservation_probe as module

    drive_root = tmp_path / "drive"
    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(drive_root))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    target = tmp_path / "outputs" / "stage5" / "scale64_run" / "phase1" / "phase1_step_200.pt"
    wrong = drive_root / "outputs" / "stage5" / "other_run" / "phase1" / "phase1_step_200.pt"
    correct = drive_root / "outputs" / "stage5" / "scale64_run" / "phase1" / "phase1_step_200.pt"
    wrong.parent.mkdir(parents=True)
    correct.parent.mkdir(parents=True)
    wrong.write_bytes(b"wrong")
    correct.write_bytes(b"correct")

    module.restore_checkpoint_if_needed(target)

    assert target.read_bytes() == b"correct"


def test_restore_checkpoint_refuses_ambiguous_same_name_fallback(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_direct_preservation_probe as module

    drive_root = tmp_path / "drive"
    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(drive_root))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    for run_id in ("other_a", "other_b"):
        candidate = drive_root / "outputs" / "stage5" / run_id / "phase1" / "phase1_step_200.pt"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(run_id.encode("utf-8"))

    target = tmp_path / "outputs" / "stage5" / "scale64_run" / "phase1" / "phase1_step_200.pt"

    with pytest.raises(FileNotFoundError) as exc:
        module.restore_checkpoint_if_needed(target)

    assert "Refusing ambiguous restore" in str(exc.value)
    assert "scale64_run" in str(exc.value)


def test_candidate_drive_checkpoints_includes_run_dir_backup_shapes(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_direct_preservation_probe as module

    drive_root = tmp_path / "drive"
    monkeypatch.setenv("DRIVE_BACKUP_DIR", str(drive_root))
    monkeypatch.delenv("DRIVE_BACKUP_DIRS", raising=False)
    monkeypatch.delenv("STAGE5_DRIVE_BACKUP_DIR", raising=False)

    candidates = module.candidate_drive_checkpoints(
        "scale64_run",
        "outputs/stage5/scale64_run/phase1/phase1_step_200.pt",
        "phase1_step_200.pt",
    )

    expected = drive_root / "stage5" / "scale64_run" / "run_dir" / "phase1" / "phase1_step_200.pt"
    assert expected in candidates


def test_infer_stage5_run_id_from_checkpoint_path() -> None:
    import colab.run_stage5_direct_preservation_probe as module

    checkpoint = Path("outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/phase1/phase1_step_200.pt")

    assert module.infer_stage5_run_id(checkpoint) == "stage5_local_hf_traced_capability_sft_20260623_194543"
